/*
  ESP32-CAM cube-face classifier: camera -> find the face (facefind.cpp) -> int8 CNN (TFLite Micro + esp-nn)
  -> vote over VOTE_N_FRAMES frames -> result over the USB serial cable. PC side: camserial.py,
  viewer: python -m classifier.device_live --url serial:/dev/ttyUSB0

  Pipeline per frame
    raw RGB565 VGA capture -> only the middle third (belt band, or the full frame with "set crop 0") is converted
    to BGR888 (no JPEG decode: the full-frame decode cost ~410 ms) -> ff_process (same as preprocess.py)
    -> 64x64 face, int8 input = pixel - 128 -> Invoke -> class probabilities -> Voter -> decision + destination
  Same packet protocol as firmware/esp32_cam_serial (see camserial.py) plus:
    'R' JSON result for every processed frame (none-frames: at most 2 per second)
        {"n":frame,"t":cam_s,"st":"ok|partial|none","box":[x1,y1,x2,y2],"c":[8 corner coords],"cm":side,
         "p":[probs],"cls":"stripes","votes":["stripes",..],"dec":"stripes|dudoso|null","why":"..","dest":"servo_2",
         "ms":{"cap":..,"rgb":..,"pre":..,"cnn":..}}         (coordinates in FRAME pixels)
    'F' the 64x64 face the model saw, raw RGB (12288 bytes), after every classified frame ("face 0" to stop)
    'J' optional preview JPEG of the band (software-encoded, ~100 ms each: "preview 1", "fps <n>", default off)
  Commands (one line each, answered with 'A'):
    set <camvar> <int>   crop quality framesize brightness contrast saturation exposure_ctrl aec_value gain_ctrl agc_gain
    set tau <percent>    decision confidence threshold (default from params.json)
    classify <0|1>       run the classifier (default 1)
    preview <0|1>        send preview frames (default 0)     fps <n>   preview frames per second
    face <0|1>           send the face crop (default 1)       reset      forget the current vote
    stream <0|1>         same as preview (compatibility with camserial.py)
    ping                 -> "ok pong"

  Files: facefind.cpp/.h (portable, parity-tested on the PC), params.h + model_data.h (generated:
         python -m classifier.export && python -m classifier.fwgen).
  Arduino IDE: board "AI Thinker ESP32-CAM" (PSRAM enabled), esp32 core 3.x - TFLite Micro + esp-nn come with
  the core (no extra library). Close any program holding the port before uploading.
*/

#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <stdarg.h>

#include "facefind.h"
#include "params.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ===================== USER SETTINGS =====================
#define BAUD          2000000
#define FRAME_SIZE    FRAMESIZE_VGA
#define JPEG_QUALITY  12
#define XCLK_HZ       20000000
#define FLIP_VERTICAL 1
#define ARENA_BYTES   (72 * 1024)   // width-8 model: ~45 KB of activations (classifier/export.py) + interpreter
#define VOTE_LOST_FRAMES 3          // frames without the cube before the vote is reset (next cube)
#define VOTE_JUMP_PX  120           // centre jump (frame px) that means "a different cube"
// =========================================================

#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 'C', 'M'};
static uint16_t seq = 0;
bool cropEnabled = true, classifyOn = true, previewOn = false, faceOn = true;
int previewFps = 2;
float tau = VOTE_TAU;
int64_t lastPreviewUs = 0, lastNoneUs = 0;
uint32_t frameNo = 0;

uint8_t *rgb = nullptr;                       // band (or frame) converted to BGR888, PSRAM
uint8_t face[64 * 64 * 3];                    // model input as BGR (input_size must be 64, checked in setup)
uint8_t *arena = nullptr;
tflite::MicroInterpreter *interp = nullptr;
TfLiteTensor *tin = nullptr, *tout = nullptr;
const int NC = g_model_n_classes;

// timing statistics (5 s windows)
uint32_t stN = 0;
int64_t stCap = 0, stJpg = 0, stPre = 0, stCnn = 0, stT0 = 0;
int64_t stFF[FF_N_TIMING] = {0};

// RGB565 (big-endian, as the camera delivers it) rows [y0, y0+rows) -> BGR888, expanding 5/6 bits to 8
static void rgb565ToBgr(const uint8_t *src, int w, int y0, int rows, uint8_t *dst) {
  const uint8_t *p = src + (size_t)y0 * w * 2;
  for (int i = 0; i < w * rows; i++, p += 2, dst += 3) {
    uint16_t v = (uint16_t)(p[0] << 8 | p[1]);
    uint8_t r = v >> 11, g = (v >> 5) & 0x3F, b = v & 0x1F;
    dst[0] = (b << 3) | (b >> 2);
    dst[1] = (g << 2) | (g >> 4);
    dst[2] = (r << 3) | (r >> 2);
  }
}

// ---------------------------------------------------------------- packets
struct Fletcher {
  uint32_t a = 0, b = 0;
  void add(const uint8_t *d, size_t n) {
    for (size_t i = 0; i < n; i++) { a = (a + d[i]) % 255; b = (b + a) % 255; }
  }
};

static void sendPacket(char type, uint64_t tsUs, const uint8_t *payload, uint32_t len) {
  uint8_t head[15];
  head[0] = (uint8_t)type;
  memcpy(head + 1, &seq, 2);
  memcpy(head + 3, &tsUs, 8);
  memcpy(head + 11, &len, 4);
  Fletcher f;
  f.add(head, sizeof(head));
  f.add(payload, len);
  uint16_t chk = (uint16_t)((f.b << 8) | f.a);
  Serial.write(MAGIC, 4);
  Serial.write(head, sizeof(head));
  Serial.write(payload, len);
  Serial.write((const uint8_t *)&chk, 2);
  seq++;
}

static void sendText(char type, const char *fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  sendPacket(type, esp_timer_get_time(), (const uint8_t *)buf, n < 0 ? 0 : min(n, (int)sizeof(buf) - 1));
}
#define logmsg(...) sendText('L', __VA_ARGS__)
#define answer(...) sendText('A', __VA_ARGS__)

// ---------------------------------------------------------------- voter (same rule as classifier/live.py)
struct Voter {
  float probs[VOTE_N_FRAMES][8];
  int n = 0, misses = 0, decision = -2;     // -2 none yet, -1 dudoso, >= 0 class
  float cx = -1e9f, cy = -1e9f;
  char why[48] = "";
  void reset() { n = 0; misses = 0; decision = -2; cx = cy = -1e9f; why[0] = 0; }
  void miss() { if (++misses > VOTE_LOST_FRAMES) reset(); }
  bool add(const float *p, float x, float y) {          // returns true when a decision was just taken
    misses = 0;
    if (cx > -1e8f && hypotf(x - cx, y - cy) > VOTE_JUMP_PX) reset();
    cx = x; cy = y;
    if (decision != -2) return false;
    memcpy(probs[n++], p, sizeof(float) * NC);
    if (n < VOTE_N_FRAMES) return false;
    int cnt[8] = {0};
    for (int i = 0; i < n; i++) cnt[argmax(probs[i])]++;
    int win = 0;
    for (int c = 1; c < NC; c++) if (cnt[c] > cnt[win]) win = c;
    float conf = 0;
    for (int i = 0; i < n; i++) if (argmax(probs[i]) == win) conf += probs[i][win];
    conf /= cnt[win];
    const char *name = g_model_classes[win];
    if (cnt[win] < VOTE_MIN_AGREE) { decision = -1; snprintf(why, sizeof(why), "only %d/%d agree", cnt[win], n); }
    else if (conf < tau) { decision = -1; snprintf(why, sizeof(why), "%s conf %.2f < %.2f", name, conf, tau); }
    else if (!strcmp(name, "other")) { decision = -1; snprintf(why, sizeof(why), "other (%d/%d, %.2f)", cnt[win], n, conf); }
    else { decision = win; snprintf(why, sizeof(why), "%d/%d votes, conf %.2f", cnt[win], n, conf); }
    return true;
  }
  static int argmax(const float *p) {
    int b = 0;
    for (int c = 1; c < NC; c++) if (p[c] > p[b]) b = c;
    return b;
  }
} voter;

// ---------------------------------------------------------------- classification
static void classifyFrame(camera_fb_t *fb, int64_t tCap) {
  int64_t t0 = esp_timer_get_time();
  const int W = fb->width, H = fb->height;
  const int y0 = cropEnabled ? H / 3 : 0, bh = cropEnabled ? (2 * H / 3 - H / 3) : H;   // = camserial.middle_third
  rgb565ToBgr(fb->buf, W, y0, bh, rgb);
  int64_t t1 = esp_timer_get_time();
  ff_result r;
  int st = ff_process(rgb, W, bh, W * 3, &FF_PARAMS, &r, face);
  for (int k = 0; k < FF_N_TIMING; k++) stFF[k] += ff_timing_us[k];
  int64_t t2 = esp_timer_get_time(), t3 = t2;
  uint64_t ts = (uint64_t)fb->timestamp.tv_sec * 1000000ULL + fb->timestamp.tv_usec;
  frameNo++;

  float prob[8] = {0};
  bool decided = false;
  if (st == FF_OK) {
    const float si = tin->params.scale;
    const int zi = tin->params.zero_point;
    for (int i = 0; i < 64 * 64; i++)                  // face is BGR, the model wants RGB
      for (int c = 0; c < 3; c++) {
        int q = (int)lroundf(face[i * 3 + (2 - c)] / si) + zi;
        tin->data.int8[i * 3 + c] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
      }
    if (interp->Invoke() != kTfLiteOk) { logmsg("invoke failed"); return; }
    t3 = esp_timer_get_time();
    for (int c = 0; c < NC; c++) prob[c] = (tout->data.int8[c] - tout->params.zero_point) * tout->params.scale;
    decided = voter.add(prob, (r.bbox[0] + r.bbox[2]) / 2.0f, y0 + (r.bbox[1] + r.bbox[3]) / 2.0f);
  } else {
    voter.miss();
  }
  stN++; stCap += t0 - tCap; stJpg += t1 - t0; stPre += t2 - t1; stCnn += t3 - t2;

  // ---- report
  if (st == FF_NONE && voter.n == 0 && t3 - lastNoneUs < 500000) return;      // idle: at most 2 per second
  if (st == FF_NONE) lastNoneUs = t3;
  static char js[900];
  int k = snprintf(js, sizeof(js), "{\"n\":%u,\"t\":%.3f,\"st\":\"%s\"", (unsigned)frameNo, ts / 1e6,
                   st == FF_OK ? "ok" : st == FF_PARTIAL ? "partial" : "none");
  if (st != FF_NONE)
    k += snprintf(js + k, sizeof(js) - k, ",\"box\":[%d,%d,%d,%d]", r.bbox[0], y0 + r.bbox[1], r.bbox[2], y0 + r.bbox[3]);
  if (st == FF_OK) {
    k += snprintf(js + k, sizeof(js) - k, ",\"c\":[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f],\"cm\":%.1f,\"p\":[",
                  r.corners[0], y0 + r.corners[1], r.corners[2], y0 + r.corners[3], r.corners[4], y0 + r.corners[5],
                  r.corners[6], y0 + r.corners[7], (r.size_cm[0] + r.size_cm[1]) / 2);
    for (int c = 0; c < NC; c++) k += snprintf(js + k, sizeof(js) - k, "%s%.3f", c ? "," : "", prob[c]);
    k += snprintf(js + k, sizeof(js) - k, "],\"cls\":\"%s\"", g_model_classes[Voter::argmax(prob)]);
  }
  k += snprintf(js + k, sizeof(js) - k, ",\"votes\":[");
  for (int i = 0; i < voter.n; i++)
    k += snprintf(js + k, sizeof(js) - k, "%s\"%s\"", i ? "," : "", g_model_classes[Voter::argmax(voter.probs[i])]);
  if (voter.decision == -2) k += snprintf(js + k, sizeof(js) - k, "],\"dec\":null");
  else {
    const char *d = voter.decision < 0 ? "dudoso" : g_model_classes[voter.decision];
    const char *dest = DESTINATIONS[voter.decision < 0 ? NC : voter.decision];
    k += snprintf(js + k, sizeof(js) - k, "],\"dec\":\"%s\",\"why\":\"%s\",\"dest\":\"%s\",\"new\":%s", d, voter.why,
                  dest, decided ? "true" : "false");
  }
  k += snprintf(js + k, sizeof(js) - k, ",\"ms\":{\"cap\":%.0f,\"rgb\":%.0f,\"pre\":%.0f,\"cnn\":%.0f}}",
                (t0 - tCap) / 1000.0, (t1 - t0) / 1000.0, (t2 - t1) / 1000.0, (t3 - t2) / 1000.0);
  sendPacket('R', ts, (const uint8_t *)js, k);
  if (st == FF_OK && faceOn) {                          // the face as RGB for the PC viewer
    static uint8_t rgbFace[64 * 64 * 3];
    for (int i = 0; i < 64 * 64; i++) {
      rgbFace[i * 3] = face[i * 3 + 2]; rgbFace[i * 3 + 1] = face[i * 3 + 1]; rgbFace[i * 3 + 2] = face[i * 3];
    }
    sendPacket('F', ts, rgbFace, sizeof(rgbFace));
  }
}

static void reportStats() {
  int64_t now = esp_timer_get_time();
  if (now - stT0 < 5000000) return;
  if (stN)
  if (stN) {
    logmsg("stats %.1f frames/s  cap %.0f rgb %.0f pre %.0f cnn %.0f ms  heap %u psram %u",
           stN * 1e6f / (now - stT0), stCap / 1000.0f / stN, stJpg / 1000.0f / stN, stPre / 1000.0f / stN,
           stCnn / 1000.0f / stN, (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
    logmsg("stats face-finder ms: gradient %.0f nms+hyst %.0f mask %.0f comps %.0f merge %.0f warp %.0f",
           stFF[0] / 1000.0f / stN, stFF[1] / 1000.0f / stN, stFF[2] / 1000.0f / stN, stFF[3] / 1000.0f / stN,
           stFF[4] / 1000.0f / stN, stFF[5] / 1000.0f / stN);
  }
  stN = 0; stCap = stJpg = stPre = stCnn = 0; stT0 = now;
  for (int k = 0; k < FF_N_TIMING; k++) stFF[k] = 0;
}

// ---------------------------------------------------------------- commands
static int setVar(const char *var, int v) {
  if (!strcmp(var, "crop")) { cropEnabled = v != 0; voter.reset(); return 0; }
  if (!strcmp(var, "tau")) { tau = v / 100.0f; return 0; }
  if (!strcmp(var, "raw")) return 0;                    // camserial compatibility: always hardware JPEG here
  sensor_t *s = esp_camera_sensor_get();
  if (!s) return -1;
  if (!strcmp(var, "framesize"))     return s->set_framesize(s, (framesize_t)v);
  if (!strcmp(var, "quality"))       return s->set_quality(s, v);
  if (!strcmp(var, "brightness"))    return s->set_brightness(s, v);
  if (!strcmp(var, "contrast"))      return s->set_contrast(s, v);
  if (!strcmp(var, "saturation"))    return s->set_saturation(s, v);
  if (!strcmp(var, "vflip"))         return s->set_vflip(s, v);
  if (!strcmp(var, "hmirror"))       return s->set_hmirror(s, v);
  if (!strcmp(var, "exposure_ctrl")) return s->set_exposure_ctrl(s, v);
  if (!strcmp(var, "aec_value"))     return s->set_aec_value(s, v);
  if (!strcmp(var, "gain_ctrl"))     return s->set_gain_ctrl(s, v);
  if (!strcmp(var, "agc_gain"))      return s->set_agc_gain(s, v);
  return -1;
}

static void handleLine(char *line) {
  char cmd[16] = "", var[24] = "";
  int v = 0;
  if (sscanf(line, "%15s", cmd) != 1) return;
  bool one = sscanf(line, "%*s %d", &v) == 1;
  if (!strcmp(cmd, "set") && sscanf(line, "%*s %23s %d", var, &v) == 2) {
    if (setVar(var, v) < 0) answer("err set %s %d", var, v); else answer("ok set %s %d", var, v);
  } else if (!strcmp(cmd, "classify") && one) { classifyOn = v; voter.reset(); answer("ok classify %d", v); }
  else if ((!strcmp(cmd, "preview") || !strcmp(cmd, "stream")) && one) { previewOn = v; answer("ok preview %d", v); }
  else if (!strcmp(cmd, "fps") && one) { previewFps = max(1, v); answer("ok fps %d", previewFps); }
  else if (!strcmp(cmd, "face") && one) { faceOn = v; answer("ok face %d", v); }
  else if (!strcmp(cmd, "reset")) { voter.reset(); answer("ok reset"); }
  else if (!strcmp(cmd, "ping")) answer("ok pong");
  else answer("err unknown: %s", line);
}

static void pollCommands() {
  static char buf[64];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') { if (n) { buf[n] = 0; handleLine(buf); } n = 0; }
    else if (n < sizeof(buf) - 1) buf[n++] = c;
  }
}

// ---------------------------------------------------------------- setup / loop
static bool initModel() {
  if (FF_PARAMS.input_size != 64) { logmsg("params.h input_size %d != 64", FF_PARAMS.input_size); return false; }
  arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  bool internal = arena != nullptr;
  if (!arena) arena = (uint8_t *)heap_caps_malloc(ARENA_BYTES, MALLOC_CAP_SPIRAM);   // slower, but works
  if (!arena) { logmsg("no memory for the tensor arena"); return false; }
  const tflite::Model *model = tflite::GetModel(g_model);
  static tflite::MicroMutableOpResolver<5> ops;
  ops.AddConv2D(); ops.AddMaxPool2D(); ops.AddMean(); ops.AddFullyConnected(); ops.AddSoftmax();
  static tflite::MicroInterpreter it(model, ops, arena, ARENA_BYTES);
  interp = &it;
  if (interp->AllocateTensors() != kTfLiteOk) { logmsg("AllocateTensors failed (ARENA_BYTES too small?)"); return false; }
  tin = interp->input(0);
  tout = interp->output(0);
  logmsg("model %u bytes, arena used %u of %u (%s RAM), input scale %.3f zp %d, %d classes",
       g_model_len, (unsigned)interp->arena_used_bytes(), ARENA_BYTES, internal ? "internal" : "PSRAM",
       tin->params.scale, tin->params.zero_point, NC);
  return true;
}

static bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM; config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM; config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM; config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM; config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = XCLK_HZ;
  config.frame_size = FRAME_SIZE;
  config.pixel_format = PIXFORMAT_RGB565;               // raw: no JPEG decode on the ESP32
  config.jpeg_quality = JPEG_QUALITY;                   // (used by the preview encoder)
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_count = 1;                                  // 614 KB per raw VGA frame
  config.fb_location = CAMERA_FB_IN_PSRAM;
  if (esp_camera_init(&config) != ESP_OK) { logmsg("camera init failed"); return false; }
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) { s->set_vflip(s, FLIP_VERTICAL); s->set_brightness(s, 0); s->set_saturation(s, 0); }
  return true;
}

void setup() {
  WiFi.mode(WIFI_OFF);
  btStop();
  Serial.setRxBufferSize(1024);
  Serial.setTxBufferSize(32768);
  Serial.begin(BAUD);
  if (!psramFound()) { logmsg("no PSRAM: enable it in the board menu"); }
  rgb = (uint8_t *)heap_caps_malloc(640 * 480 * 3, MALLOC_CAP_SPIRAM);
  ff_clock_us = esp_timer_get_time;                     // face-finder stage timing (stats line)
  bool ok = rgb && initCamera() && initModel();
  if (!ok) {
    logmsg("init failed - restarting in 5 s");
    delay(5000);
    ESP.restart();
  }
  logmsg("ready esp32_cam_classifier baud=%d vote %d/%d tau %.2f classes %s,%s,%s,%s", BAUD, VOTE_MIN_AGREE,
       VOTE_N_FRAMES, tau, g_model_classes[0], g_model_classes[1], g_model_classes[2], g_model_classes[3]);
}

void loop() {
  pollCommands();
  int64_t now = esp_timer_get_time();
  bool previewDue = previewOn && now - lastPreviewUs >= 1000000LL / previewFps;
  if (!classifyOn && !previewDue) { delay(2); reportStats(); return; }
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { logmsg("capture failed"); delay(50); return; }
  if (previewDue) {
    lastPreviewUs = now;
    uint64_t ts = (uint64_t)fb->timestamp.tv_sec * 1000000ULL + fb->timestamp.tv_usec;
    camera_fb_t crop = *fb;                             // preview: JPEG-encode the band only
    if (cropEnabled) {
      crop.height = fb->height / 3;
      crop.buf = fb->buf + (size_t)crop.height * fb->width * 2;
      crop.len = (size_t)crop.height * fb->width * 2;
    }
    uint8_t *jpg = nullptr;
    size_t jlen = 0;
    if (frame2jpg(&crop, JPEG_QUALITY, &jpg, &jlen)) {
      sendPacket('J', ts, jpg, jlen);
      free(jpg);
    }
  }
  if (classifyOn) classifyFrame(fb, now);
  esp_camera_fb_return(fb);
  reportStats();
}
