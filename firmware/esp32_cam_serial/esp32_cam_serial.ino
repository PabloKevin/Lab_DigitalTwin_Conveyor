/*
  ESP32-CAM (AI Thinker) + OV3660  ->  JPEG frames over the USB serial cable (no WiFi)
  PC side: camserial.py  (used by vision.py and classifier/capture.py).

  Why: the lab WiFi is too slow / unreliable for the image stream. The ESP32-CAM-MB programmer board (CH340) is a
  USB-serial link, 2 Mbaud ~ 200 KB/s.

  Two capture modes (command "set raw 0|1", re-initialises the camera):
    raw 0 (default) HARDWARE JPEG from the sensor: fast (no encoding on the ESP32) and the sensor's own colour
                    pipeline. Full frame; the belt band crop is done on the PC (camserial.middle_third).
                    VGA q12 ~ 25-35 KB -> ~6-8 fps at 2 Mbaud; raise "quality" (e.g. 20) or use QVGA for more fps.
    raw 1           RGB565 capture + band crop + SOFTWARE JPEG on the ESP32: fewer bytes on the wire but the
                    encoding costs ~100-300 ms per frame and RGB565 has poorer colours (5-6 bits per channel).
  Every 5 s a log packet reports fps and where the time goes: "stats fps .. cap .. enc .. send .. ms, .. KB/frame".

  Packet (ESP32 -> PC), little-endian:
     A5 5A 'C' 'M' | type(1) | seq(u16) | ts_us(u64) | len(u32) | payload(len) | fletcher16(u16) of type..payload
     type 'J' JPEG frame (ts = capture time), 'L' log text, 'A' answer to a command.
  Commands (PC -> ESP32), one line each, answered with an 'A' packet "ok ..." / "err ...":
     set <var> <int>   raw crop quality framesize brightness contrast saturation vflip hmirror
                       exposure_ctrl aec_value gain_ctrl agc_gain      (same names as the WiFi firmware's /control)
     fps <n>           max frames per second (0 = as fast as the link allows)
     stream <0|1>      start / stop sending frames (starts stopped: the PC sends "stream 1")
     snap              send one frame now
     ping              -> "ok pong"
  Nothing else may be printed on Serial: every byte has to be inside a packet (use logf()).

  Arduino IDE:
    Board : "AI Thinker ESP32-CAM" (PSRAM enabled), esp32 core 3.x
    Upload: ESP32-CAM-MB board (auto-reset) - close any program holding the port (app.py, capture.py) first.
*/

#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <stdarg.h>

// ===================== USER SETTINGS =====================
#define BAUD          921600     // CH340 on the ESP32-CAM-MB handles 2 Mbaud; use 921600 if you see many bad packets 2000000
                                  // (and add "@921600" to the PC url: "serial:/dev/ttyUSB0@921600")
#define FRAME_SIZE    FRAMESIZE_VGA
#define JPEG_QUALITY  12          // 10 = best/biggest ... 63 = worst/smallest
#define XCLK_HZ       20000000
#define FLIP_VERTICAL 1           // the OV3660 on most boards comes out upside down
#define DEFAULT_FPS   15
#define RAW_DEFAULT   0           // 0 = hardware JPEG (fast, good colours), 1 = RGB565 + crop + software JPEG
#define CROP_DEFAULT  1           // raw mode only: send just the middle third (the belt band)
// =========================================================

// AI Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

static const uint8_t MAGIC[4] = {0xA5, 0x5A, 'C', 'M'};
static uint16_t seq = 0;
volatile bool cropEnabled = CROP_DEFAULT;
bool rawMode = RAW_DEFAULT, psram = false;
// timing statistics, reported every 5 s
uint32_t stFrames = 0, stBytes = 0;
int64_t stCap = 0, stEnc = 0, stSend = 0, stT0 = 0;
bool streaming = false, snapPending = false;
int maxFps = DEFAULT_FPS;
int64_t lastFrameUs = 0;

// ---------------------------------------------------------------- packets
struct Fletcher {
  uint32_t a = 0, b = 0;
  void add(const uint8_t *d, size_t n) {
    for (size_t i = 0; i < n; i++) { a = (a + d[i]) % 255; b = (b + a) % 255; }
  }
  uint16_t value() const { return (uint16_t)((b << 8) | a); }
};

static void sendPacket(char type, uint64_t tsUs, const uint8_t *payload, uint32_t len) {
  uint8_t head[1 + 2 + 8 + 4];
  head[0] = (uint8_t)type;
  memcpy(head + 1, &seq, 2);                 // ESP32 is little-endian, same as the PC struct format "<cHQI"
  memcpy(head + 3, &tsUs, 8);
  memcpy(head + 11, &len, 4);
  Fletcher f;
  f.add(head, sizeof(head));
  f.add(payload, len);
  uint16_t chk = f.value();
  Serial.write(MAGIC, 4);
  Serial.write(head, sizeof(head));
  Serial.write(payload, len);                // blocks while the TX buffer is full = natural back-pressure
  Serial.write((const uint8_t *)&chk, 2);
  seq++;
}

static void logf(const char *fmt, ...) {
  char buf[200];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  sendPacket('L', esp_timer_get_time(), (const uint8_t *)buf, n < 0 ? 0 : min(n, (int)sizeof(buf) - 1));
}

static void answer(const char *fmt, ...) {
  char buf[120];
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  sendPacket('A', esp_timer_get_time(), (const uint8_t *)buf, n < 0 ? 0 : min(n, (int)sizeof(buf) - 1));
}

// ---------------------------------------------------------------- frames
// Crops a raw RGB565 frame to its middle third (rows are contiguous: pointer arithmetic, no copy) and JPEG-encodes
// it in software. Returns false for a hardware-JPEG frame (no PSRAM), which is then sent as is.
static bool cropAndEncode(camera_fb_t *fb, uint8_t **out, size_t *outLen) {
  if (fb->format != PIXFORMAT_RGB565) return false;
  camera_fb_t crop = *fb;
  if (cropEnabled) {
    crop.height = fb->height / 3;
    crop.buf    = fb->buf + crop.height * fb->width * 2;
    crop.len    = crop.height * fb->width * 2;
  }
  sensor_t *s = esp_camera_sensor_get();
  return frame2jpg(&crop, s ? s->status.quality : JPEG_QUALITY, out, outLen);
}

static void sendFrame() {
  int64_t t0 = esp_timer_get_time();
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    logf("camera capture failed");
    return;
  }
  int64_t t1 = esp_timer_get_time();
  uint64_t ts = (uint64_t)fb->timestamp.tv_sec * 1000000ULL + fb->timestamp.tv_usec;
  uint8_t *jpg = NULL;
  size_t jpgLen = 0;
  bool encoded = cropAndEncode(fb, &jpg, &jpgLen);
  int64_t t2 = esp_timer_get_time();
  if (encoded) {
    sendPacket('J', ts, jpg, jpgLen);
    free(jpg);
  } else {
    sendPacket('J', ts, fb->buf, fb->len);
    jpgLen = fb->len;
  }
  esp_camera_fb_return(fb);
  int64_t t3 = esp_timer_get_time();
  stFrames++;
  stBytes += jpgLen;
  stCap += t1 - t0;
  stEnc += t2 - t1;
  stSend += t3 - t2;
}

static void reportStats() {
  int64_t now = esp_timer_get_time();
  if (now - stT0 < 5000000) return;
  if (stFrames) {
    logf("stats fps %.1f cap %.0f enc %.0f send %.0f ms, %.1f KB/frame, mode %s",
         stFrames * 1e6f / (now - stT0), stCap / 1000.0f / stFrames, stEnc / 1000.0f / stFrames,
         stSend / 1000.0f / stFrames, stBytes / 1024.0f / stFrames, rawMode ? "raw" : "jpeg");
  }
  stFrames = stBytes = 0;
  stCap = stEnc = stSend = 0;
  stT0 = now;
}

// ---------------------------------------------------------------- commands
static bool initCamera(bool raw);

static int setVar(const char *var, int v) {
  if (!strcmp(var, "crop")) { cropEnabled = v != 0; return 0; }          // raw mode only (jpeg: the PC crops)
  if (!strcmp(var, "raw")) {
    if ((v != 0) == rawMode) return 0;                                    // no change: no re-init
    esp_camera_deinit();
    return initCamera(v != 0) ? 0 : -1;                                   // logs "ready ..." -> PC re-sends settings
  }
  sensor_t *s = esp_camera_sensor_get();
  if (!s) return -1;
  if (!strcmp(var, "framesize"))     return s->set_framesize(s, (framesize_t)v);
  if (!strcmp(var, "quality"))       return s->set_quality(s, v);          // 10..63
  if (!strcmp(var, "brightness"))    return s->set_brightness(s, v);       // -2..2
  if (!strcmp(var, "contrast"))      return s->set_contrast(s, v);         // -2..2
  if (!strcmp(var, "saturation"))    return s->set_saturation(s, v);       // -2..2
  if (!strcmp(var, "vflip"))         return s->set_vflip(s, v);
  if (!strcmp(var, "hmirror"))       return s->set_hmirror(s, v);
  if (!strcmp(var, "exposure_ctrl")) return s->set_exposure_ctrl(s, v);    // 1 = auto, 0 = manual
  if (!strcmp(var, "aec_value"))     return s->set_aec_value(s, v);        // manual exposure 0..1200
  if (!strcmp(var, "gain_ctrl"))     return s->set_gain_ctrl(s, v);        // 1 = auto, 0 = manual
  if (!strcmp(var, "agc_gain"))      return s->set_agc_gain(s, v);         // manual gain 0..30
  return -1;
}

static void handleLine(char *line) {
  char cmd[16] = "", var[24] = "";
  int v = 0;
  if (sscanf(line, "%15s", cmd) != 1) return;
  if (!strcmp(cmd, "set") && sscanf(line, "%*s %23s %d", var, &v) == 2) {
    int r = setVar(var, v);
    if (r < 0) answer("err set %s %d", var, v);
    else answer("ok set %s %d", var, v);
  } else if (!strcmp(cmd, "fps") && sscanf(line, "%*s %d", &v) == 1) {
    maxFps = max(0, v);
    answer("ok fps %d", maxFps);
  } else if (!strcmp(cmd, "stream") && sscanf(line, "%*s %d", &v) == 1) {
    streaming = v != 0;
    answer("ok stream %d", streaming);
  } else if (!strcmp(cmd, "snap")) {
    snapPending = true;
    answer("ok snap");
  } else if (!strcmp(cmd, "ping")) {
    answer("ok pong");
  } else {
    answer("err unknown: %s", line);
  }
}

static void pollCommands() {
  static char buf[64];
  static uint8_t n = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (n) { buf[n] = 0; handleLine(buf); }
      n = 0;
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = c;
    }
  }
}

// ---------------------------------------------------------------- setup / loop
static bool initCamera(bool raw) {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk  = XCLK_GPIO_NUM;
  config.pin_pclk  = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = XCLK_HZ;
  config.frame_size   = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  rawMode = raw && psram;                     // raw VGA (614 KB) needs PSRAM
  if (rawMode) {
    config.pixel_format = PIXFORMAT_RGB565;   // raw, so rows can be cropped before (software) JPEG encoding
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_PSRAM;
  } else {
    config.pixel_format = PIXFORMAT_JPEG;     // hardware JPEG
    if (psram) {
      config.fb_count    = 2;                 // capture the next frame while this one is being sent
      config.fb_location = CAMERA_FB_IN_PSRAM;
    } else {                                  // no PSRAM: small frames in internal RAM
      config.frame_size  = FRAMESIZE_QVGA;
      config.fb_count    = 1;
      config.fb_location = CAMERA_FB_IN_DRAM;
    }
  }
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    logf("camera init failed 0x%x (ribbon cable, power, board selection)", err);
    return false;
  }
  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    s->set_vflip(s, FLIP_VERTICAL);
    s->set_brightness(s, 0);
    s->set_saturation(s, 0);
  }
  logf("ready esp32_cam_serial baud=%d psram=%d sensor=0x%x mode=%s frame=%s (send 'stream 1')",
       BAUD, psram, s->id.PID, rawMode ? "raw" : "jpeg",
       config.frame_size == FRAMESIZE_VGA ? "VGA 640x480" : config.frame_size == FRAMESIZE_QVGA ? "QVGA 320x240" : "other");
  return true;
}

void setup() {
  WiFi.mode(WIFI_OFF);                      // no radio: less power, less noise, more CPU for JPEG
  btStop();
  Serial.setRxBufferSize(1024);
  Serial.setTxBufferSize(32768);            // must be set before begin()
  Serial.begin(BAUD);
  psram = psramFound();
  if (!initCamera(RAW_DEFAULT)) {
    delay(3000);
    ESP.restart();
  }
}

void loop() {
  pollCommands();
  int64_t now = esp_timer_get_time();
  bool due = streaming && (maxFps <= 0 || now - lastFrameUs >= 1000000LL / maxFps);
  if (due || snapPending) {
    lastFrameUs = now;
    snapPending = false;
    sendFrame();
  } else {
    delay(1);
  }
  reportStats();
}
