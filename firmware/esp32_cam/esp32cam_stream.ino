/*
  ESP32-CAM (AI Thinker) + OV3660  ->  MJPEG-over-HTTP streaming server

  Endpoints
    http://<ip>/            tiny viewer page (open it in a browser to check the image)
    http://<ip>/capture     one JPEG (with X-Timestamp header)
    http://<ip>/control     ?var=<name>&val=<int>   change camera settings on the fly
    http://<ip>:81/stream   MJPEG stream (multipart/x-mixed-replace), every frame has an
                            X-Timestamp header = capture time on the ESP32 (used for speed)

  Arduino IDE:
    Board manager : "esp32" by Espressif Systems (3.x, or 2.0.17)
    Board         : "AI Thinker ESP32-CAM"      (PSRAM must be enabled)
    Serial monitor: 115200 baud
*/

#include "esp_camera.h"
#include "img_converters.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_http_server.h"
#include "secrets.h"

// ===================== USER SETTINGS =====================
const char* HOSTNAME  = "esp32cam";          // reachable as esp32cam.local (mDNS)

// Frame size values: 5=QVGA(320x240) 8=VGA(640x480) 9=SVGA(800x600) 10=XGA(1024x768)
//                    11=HD(1280x720) 12=SXGA(1280x1024) 13=UXGA(1600x1200)
#define FRAME_SIZE    FRAMESIZE_VGA
#define JPEG_QUALITY  12          // 10 = best/biggest ... 63 = worst/smallest
#define XCLK_HZ       20000000    // try 10000000 if you see garbage/green frames
#define FLIP_VERTICAL 1           // the OV3660 on most boards comes out upside down

// The belt only occupies the middle band of the frame (see the lab photo), so we crop the sensor
// output to the middle third vertically BEFORE sending it over WiFi: about 1/3 the pixels and
// roughly 1/3 the JPEG bytes per frame, which is most of the latency on a slow WiFi link. This
// requires capturing raw pixels (PIXFORMAT_RGB565) instead of the camera's own hardware JPEG
// encoder, then cropping the raw rows (cheap: they are just contiguous memory) and JPEG-encoding
// only the cropped band in software (frame2jpg). Set to 0 to go back to sending the full,
// hardware-JPEG-encoded frame (less CPU work on the ESP32, more bytes over WiFi).
#define CROP_MIDDLE_THIRD 1
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

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %ld.%06ld\r\n\r\n";

httpd_handle_t web_httpd    = NULL;
httpd_handle_t stream_httpd = NULL;

// Crops a raw RGB565 frame to its middle third (by rows) and JPEG-encodes just that band in
// software. Raw rows are contiguous in memory, so the crop itself is just pointer arithmetic - no
// copy needed. Returns true and fills *out/*out_len (caller must free(*out)) on success; false if
// fb isn't a raw format we can crop (hardware JPEG, i.e. CROP_MIDDLE_THIRD off or no PSRAM) - the
// caller should then send fb->buf/fb->len unmodified.
static bool cropAndEncode(camera_fb_t *fb, uint8_t **out, size_t *out_len) {
  if (fb->format != PIXFORMAT_RGB565) return false;
  const size_t bypp = 2;                     // bytes per pixel, RGB565
  camera_fb_t crop = *fb;                    // shallow copy: reuse width/format/timestamp
  crop.height = fb->height / 3;
  crop.buf    = fb->buf + crop.height * fb->width * bypp;   // skip the top third
  crop.len    = crop.height * fb->width * bypp;
  sensor_t *s = esp_camera_sensor_get();     // honour the live "JPEG quality" control (/control?var=quality)
  return frame2jpg(&crop, s ? s->status.quality : JPEG_QUALITY, out, out_len);
}

// ---------------------------------------------------------------- handlers
static esp_err_t index_handler(httpd_req_t *req) {
  static const char page[] =
    "<html><body style='margin:0;background:#111'>"
    "<img id=v style='width:100%'>"
    "<script>document.getElementById('v').src='http://'+location.hostname+':81/stream';</script>"
    "</body></html>";
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, page, HTTPD_RESP_USE_STRLEN);
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  char ts[32];
  snprintf(ts, sizeof(ts), "%ld.%06ld", (long)fb->timestamp.tv_sec, (long)fb->timestamp.tv_usec);
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "X-Timestamp", ts);

  uint8_t *jpg = NULL;
  size_t jpg_len = 0;
  esp_err_t res;
  if (cropAndEncode(fb, &jpg, &jpg_len)) {
    res = httpd_resp_send(req, (const char *)jpg, jpg_len);
    free(jpg);
  } else {
    res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  }
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t stream_handler(httpd_req_t *req) {
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  char part_buf[128];
  uint32_t frames = 0;
  int64_t t_report = esp_timer_get_time();

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    uint8_t *jpg = NULL;
    size_t jpg_len = 0;
    bool cropped = cropAndEncode(fb, &jpg, &jpg_len);
    const uint8_t *send_buf = cropped ? jpg : fb->buf;
    size_t send_len = cropped ? jpg_len : fb->len;

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned)send_len,
                             (long)fb->timestamp.tv_sec, (long)fb->timestamp.tv_usec);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)send_buf, send_len);
    }
    if (cropped) free(jpg);
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;   // client disconnected

    frames++;
    int64_t now = esp_timer_get_time();
    if (now - t_report > 5000000) {
      Serial.printf("stream: %.1f fps, last frame %u KB\n",
                    frames * 1e6f / (now - t_report), (unsigned)(send_len / 1024));
      frames = 0;
      t_report = now;
    }
  }
  return res;
}

// /control?var=framesize&val=9
static esp_err_t control_handler(httpd_req_t *req) {
  char query[64], var[24], val[16];
  if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
      httpd_query_key_value(query, "var", var, sizeof(var)) != ESP_OK ||
      httpd_query_key_value(query, "val", val, sizeof(val)) != ESP_OK) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }
  int v = atoi(val);
  sensor_t *s = esp_camera_sensor_get();
  int r = -1;
  if      (!strcmp(var, "framesize"))     r = s->set_framesize(s, (framesize_t)v);
  else if (!strcmp(var, "quality"))       r = s->set_quality(s, v);          // 10..63
  else if (!strcmp(var, "brightness"))    r = s->set_brightness(s, v);       // -2..2
  else if (!strcmp(var, "contrast"))      r = s->set_contrast(s, v);         // -2..2
  else if (!strcmp(var, "saturation"))    r = s->set_saturation(s, v);       // -2..2
  else if (!strcmp(var, "vflip"))         r = s->set_vflip(s, v);            // 0/1
  else if (!strcmp(var, "hmirror"))       r = s->set_hmirror(s, v);          // 0/1
  else if (!strcmp(var, "exposure_ctrl")) r = s->set_exposure_ctrl(s, v);    // 1=auto 0=manual
  else if (!strcmp(var, "aec_value"))     r = s->set_aec_value(s, v);        // manual exposure 0..1200
  else if (!strcmp(var, "gain_ctrl"))     r = s->set_gain_ctrl(s, v);        // 1=auto 0=manual
  else if (!strcmp(var, "agc_gain"))      r = s->set_agc_gain(s, v);         // manual gain 0..30
  if (r < 0) {
    httpd_resp_send_404(req);
    return ESP_FAIL;
  }
  return httpd_resp_send(req, "OK", HTTPD_RESP_USE_STRLEN);
}

static void startServers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t index_uri   = { .uri = "/",        .method = HTTP_GET, .handler = index_handler,   .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
  httpd_uri_t control_uri = { .uri = "/control", .method = HTTP_GET, .handler = control_handler, .user_ctx = NULL };
  httpd_uri_t stream_uri  = { .uri = "/stream",  .method = HTTP_GET, .handler = stream_handler,  .user_ctx = NULL };

  if (httpd_start(&web_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(web_httpd, &index_uri);
    httpd_register_uri_handler(web_httpd, &capture_uri);
    httpd_register_uri_handler(web_httpd, &control_uri);
  }
  // The stream handler never returns while a client is connected, so it gets its own server
  config.server_port += 1;
  config.ctrl_port   += 1;
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ---------------------------------------------------------------- setup / loop
void setup() {
  Serial.begin(115200);
  Serial.println();

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
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAME_SIZE;
  config.jpeg_quality = JPEG_QUALITY;
  config.grab_mode    = CAMERA_GRAB_LATEST;      // always give the newest frame (low latency)

  if (psramFound()) {
#if CROP_MIDDLE_THIRD
    // Raw capture needed to crop rows before JPEG-encoding (see cropAndEncode()). Raw frames are
    // much bigger than JPEG (VGA RGB565 = 614 KB vs a few tens of KB), so one frame buffer is
    // enough - PSRAM headroom matters more here than double-buffering.
    config.pixel_format = PIXFORMAT_RGB565;
    config.fb_count     = 1;
#else
    config.fb_count     = 2;
#endif
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    // No PSRAM: not enough internal RAM for a raw frame buffer, so always fall back to the
    // camera's own hardware JPEG encoder at a small frame size, regardless of CROP_MIDDLE_THIRD.
    Serial.println("WARNING: no PSRAM found -> falling back to small hardware-JPEG frames (no crop)");
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size   = FRAMESIZE_QVGA;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed: 0x%x  (check ribbon cable, power and board selection)\n", err);
    delay(3000);
    ESP.restart();
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {
    Serial.println("Sensor: OV3660");
    s->set_vflip(s, FLIP_VERTICAL);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  } else {
    Serial.printf("Sensor PID 0x%x (not an OV3660)\n", s->id.PID);
  }

  WiFi.mode(WIFI_STA);
  WiFi.setHostname(HOSTNAME);
  WiFi.setSleep(false);                 // WiFi power-save adds latency and jitter
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (millis() - t0 > 20000) {
      Serial.println("\nWiFi timeout, restarting");
      ESP.restart();
    }
  }
  WiFi.setAutoReconnect(true);
  Serial.printf("\nConnected. IP: %s\n", WiFi.localIP().toString().c_str());

  if (MDNS.begin(HOSTNAME)) Serial.printf("mDNS: http://%s.local\n", HOSTNAME);

  startServers();
  Serial.printf("Viewer : http://%s/\n", WiFi.localIP().toString().c_str());
  Serial.printf("Stream : http://%s:81/stream\n", WiFi.localIP().toString().c_str());
}

void loop() {
  delay(10000);
  Serial.printf("WiFi RSSI %d dBm, free heap %u, free PSRAM %u\n",
                WiFi.RSSI(), (unsigned)ESP.getFreeHeap(), (unsigned)ESP.getFreePsram());
}
