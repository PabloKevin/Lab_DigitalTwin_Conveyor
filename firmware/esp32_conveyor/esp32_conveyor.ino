/*
  esp32_conveyor.ino - conveyor controller with MQTT (counterpart of the digital twin)

  Board:   ESP32 (Arduino-ESP32 core 3.x).  Library: "PubSubClient" by Nick O'Leary.
  Motor:   DC motor + quadrature encoder (20 pulses/rev on channel A, like the lab guide) + L298N.
  Protocol (must match config.py):
    publishes  conveyor/telemetry  {"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F"}   every 100 ms
               conveyor/status     "online" (retained)  /  "offline" (Last Will)
    subscribes conveyor/cmd/direction  F | R | S
               conveyor/cmd/speed      0..100   (% of MAX_RPM)
               conveyor/cmd/kp | ki | kd
  Safety: if the MQTT link is lost for more than FAILSAFE_MS the motor stops.

  For Arduino-ESP32 core 2.x replace ledcAttach()/ledcWrite(pin, ..) by
  ledcSetup(0, 20000, 8); ledcAttachPin(ENA, 0); ledcWrite(0, ..)
*/
#include <WiFi.h>
#include <PubSubClient.h>

// ── network ────────────────────────────────────────────────────────────
#include "secrets.h"                         // defines WIFI_SSID, WIFI_PASS, MQTT_HOST (copy secrets.h.example, never commit it)
const uint16_t MQTT_PORT = 1883;

// ── pins ───────────────────────────────────────────────────────────────
#define ENC_A 18
#define ENC_B 19
#define IN1   26
#define IN2   27
#define ENA   25

// ── conveyor ───────────────────────────────────────────────────────────
const float PPR = 20.0;                      // encoder pulses per revolution
const float MAX_RPM = 300.0;                 // RPM at 100 %
const unsigned long SAMPLE_MS = 100;
const unsigned long FAILSAFE_MS = 3000;

volatile long encCount = 0;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

char dirCmd = 'S';
float speedPct = 0;
float kp = 2.0, ki = 0.5, kd = 0.1;
float integ = 0, prevErr = 0;
int pwmOut = 0;

WiFiClient wifi;
PubSubClient mqtt(wifi);
unsigned long lastSample = 0, lastReconnect = 0, lastConnected = 0;

void IRAM_ATTR encoderISR() {
  portENTER_CRITICAL_ISR(&mux);
  if (digitalRead(ENC_B)) encCount++; else encCount--;
  portEXIT_CRITICAL_ISR(&mux);
}

void applyDirection() {
  digitalWrite(IN1, dirCmd == 'F');
  digitalWrite(IN2, dirCmd == 'R');
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  char buf[24];
  len = min(len, (unsigned int)23);
  memcpy(buf, payload, len);
  buf[len] = 0;
  const char* key = strrchr(topic, '/') + 1;

  if (!strcmp(key, "direction") && (buf[0] == 'F' || buf[0] == 'R' || buf[0] == 'S')) {
    dirCmd = buf[0];
    applyDirection();
  } else if (!strcmp(key, "speed")) speedPct = constrain(atof(buf), 0, 100);
  else if (!strcmp(key, "kp")) kp = atof(buf);
  else if (!strcmp(key, "ki")) { ki = atof(buf); integ = 0; }
  else if (!strcmp(key, "kd")) kd = atof(buf);
}

void connectMqtt() {
  if (mqtt.connect("esp32-conveyor", NULL, NULL, "conveyor/status", 1, true, "offline")) {
    mqtt.publish("conveyor/status", "online", true);
    mqtt.subscribe("conveyor/cmd/#");
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  ledcAttach(ENA, 20000, 8);
  ledcWrite(ENA, 0);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, RISING);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.println(WiFi.localIP());
  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(onMessage);
}

void loop() {
  // ---- connection handling (non blocking) + failsafe
  if (mqtt.connected()) {
    mqtt.loop();
    lastConnected = millis();
  } else {
    if (millis() - lastReconnect > 2000) { lastReconnect = millis(); connectMqtt(); }
    if (millis() - lastConnected > FAILSAFE_MS && dirCmd != 'S') { dirCmd = 'S'; applyDirection(); }
  }

  // ---- speed loop every SAMPLE_MS
  if (millis() - lastSample >= SAMPLE_MS) {
    float dt = (millis() - lastSample) / 1000.0;
    lastSample = millis();

    portENTER_CRITICAL(&mux);
    long count = encCount;
    encCount = 0;
    portEXIT_CRITICAL(&mux);
    float rpm = (count / PPR) * (60.0 / dt);               // signed

    float spMag = MAX_RPM * speedPct / 100.0;
    if (dirCmd == 'S' || spMag < 1.0) {
      integ = 0; prevErr = 0; pwmOut = 0;
    } else {
      float err = spMag - fabsf(rpm);
      integ = constrain(integ + err * dt, 0, ki > 0 ? 255.0 / ki : 0);   // anti-windup
      float out = kp * err + ki * integ + kd * (err - prevErr) / dt;
      prevErr = err;
      pwmOut = constrain((int)out, 0, 255);
    }
    ledcWrite(ENA, pwmOut);

    if (mqtt.connected()) {
      float sp = (dirCmd == 'R' ? -1 : (dirCmd == 'F' ? 1 : 0)) * spMag;
      char msg[96];
      snprintf(msg, sizeof(msg), "{\"rpm\":%.1f,\"setpoint\":%.1f,\"output\":%d,\"dir\":\"%c\"}", rpm, sp, pwmOut, dirCmd);
      mqtt.publish("conveyor/telemetry", msg);
    }
  }
}
