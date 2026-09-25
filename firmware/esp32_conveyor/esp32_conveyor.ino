/*
  esp32_conveyor.ino - conveyor controller with MQTT (counterpart of the digital twin)

  Board:     ESP32 DevKit V1 (ESP32-WROOM-32, Arduino-ESP32 core 3.x).
  Libraries: "PubSubClient" by Nick O'Leary, "PID" (PID_v1) by Brett Beauregard.
  Motor:     DC motor + KY-040-style encoder (single channel, pulses on ENC_A only) + L298N.

  The encoder measurement (pulse-interval buffer + deadband/EMA filter) and the PID (PID_v1) are
  ported as-is from the Arduino UNO sketch that was validated on the real conveyor. Only the pins,
  the PWM call (ledc instead of analogWrite) and the transport (WiFi/MQTT/JSON instead of
  Serial/CSV + single-char commands) changed for the ESP32 + digital twin.

  NOTE ON DIRECTION: like the UNO sketch, this encoder method only measures pulse timing (KY-040
  CLK only, no DT/quadrature), so it cannot sense direction by itself - the sign of the reported
  RPM is taken from the commanded direction (dirCmd), not measured. If you need to detect the motor
  spinning the wrong way (e.g. wiring fault), you need a second encoder channel (quadrature) instead.

  Protocol (must match config.py):
    publishes  conveyor/telemetry  {"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F"}   every 100 ms
               conveyor/status     "online" (retained)  /  "offline" (Last Will)
    subscribes conveyor/cmd/direction  F | R | S
               conveyor/cmd/speed      0..100   (% of MAX_RPM)
               conveyor/cmd/kp | ki | kd
  Safety: if the MQTT link is lost for more than FAILSAFE_MS the motor stops.

  Optional: HC-SR04 (TRIG/ECHO) for an independent object-distance/speed reading, ported from the
  UNO sketch and published as extra "distance_cm"/"obj_speed_cm_s" telemetry fields (the digital
  twin ignores unknown JSON fields, so this is harmless even if you don't use them in config.py).
  Not required - the twin already gets object position/speed from the camera. Set ULTRASONIC to 0
  below if the sensor isn't wired.

  For Arduino-ESP32 core 2.x replace ledcAttach()/ledcWrite(pin, ..) by
  ledcSetup(0, 20000, 8); ledcAttachPin(ENA, 0); ledcWrite(0, ..)
*/
#include <WiFi.h>
#include <PubSubClient.h>
#include <PID_v1.h>

// ── network ────────────────────────────────────────────────────────────
#include "secrets.h"                         // defines WIFI_SSID, WIFI_PASS, MQTT_HOST (copy secrets.h.example, never commit it)
const uint16_t MQTT_PORT = 1883;

// ── pins (ESP32 DevKit V1) ────────────────────────────────────────────
#define ENC_A 18                             // KY-040 CLK, interrupt input (DT is not needed by this method)
#define IN1   26                             // L298N
#define IN2   27                             // L298N
#define ENA   25                             // L298N PWM (ledc)
#define ULTRASONIC 1                         // 1 = HC-SR04 wired and read every sample, 0 = disabled
#define TRIG  32                             // HC-SR04 (only used if ULTRASONIC)
#define ECHO  33                             // HC-SR04 (only used if ULTRASONIC)

// ── conveyor ───────────────────────────────────────────────────────────
const float PPR = 20.0;                      // encoder pulses per revolution (lab guide's disk)
const float MAX_RPM = 300.0;                 // RPM at 100 % (maps the MQTT speed % command to a setpoint)
const float MAX_RPM_PHYSICAL = 250.0;        // sanity cap: an RPM estimate above this is treated as a glitch
const unsigned long SAMPLE_MS = 100;
const unsigned long FAILSAFE_MS = 3000;

// ── encoder: pulse-interval buffer (ported from the UNO sketch) ─────────
#define NUM_INTERVALS 4
volatile unsigned long intervalBuf[NUM_INTERVALS] = {0, 0, 0, 0};
volatile byte intervalIdx = 0;
volatile unsigned long lastPulseTime = 0;
float rpmFiltered = 0;

void IRAM_ATTR encoderISR() {
  unsigned long now = micros();
  unsigned long dt = now - lastPulseTime;
  if (dt > 8000) {                           // debounce: ignore glitches faster than 8 ms apart
    intervalBuf[intervalIdx] = dt;
    intervalIdx = (intervalIdx + 1) % NUM_INTERVALS;
    lastPulseTime = now;
  }
}

char dirCmd = 'S';
float speedPct = 0;
double kp = 0.4, ki = 0.5, kd = 0.0;
double pidSetpoint = 0, pidInput = 0, pidOutput = 0;
PID pid(&pidInput, &pidOutput, &pidSetpoint, kp, ki, kd, DIRECT);
int pwmOut = 0;

#if ULTRASONIC
float distanceCm = 0, lastDistanceCm = 0, objSpeedCmS = 0;
const float DIST_DETECT_THRESHOLD_CM = 55.0;
#endif

WiFiClient wifi;
PubSubClient mqtt(wifi);
unsigned long lastSample = 0, lastReconnect = 0, lastConnected = 0;

void applyDirection() {
  digitalWrite(IN1, dirCmd == 'F');
  digitalWrite(IN2, dirCmd == 'R');
}

void stopMotor() {
  pwmOut = 0;
  ledcWrite(ENA, 0);
  pid.SetMode(MANUAL);
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
    if (dirCmd == 'S') stopMotor();
    else pid.SetMode(AUTOMATIC);             // bumpless transfer handled by the PID library
  } else if (!strcmp(key, "speed")) speedPct = constrain(atof(buf), 0, 100);
  else if (!strcmp(key, "kp")) { kp = atof(buf); pid.SetTunings(kp, ki, kd); }
  else if (!strcmp(key, "ki")) { ki = atof(buf); pid.SetTunings(kp, ki, kd); }
  else if (!strcmp(key, "kd")) { kd = atof(buf); pid.SetTunings(kp, ki, kd); }
}

void connectMqtt() {
  Serial.printf("MQTT: connecting to %s:%u ... ", MQTT_HOST, MQTT_PORT);
  if (mqtt.connect("esp32-conveyor", NULL, NULL, "conveyor/status", 1, true, "offline")) {
    Serial.println("connected");
    mqtt.publish("conveyor/status", "online", true);
    mqtt.subscribe("conveyor/cmd/#");
  } else {
    // PubSubClient state(): -4 timeout, -3 lost, -2/-1 connect failed, 1..5 broker refused (bad proto/id/creds/unauthorized)
    Serial.printf("failed, state=%d\n", mqtt.state());
  }
}

#if ULTRASONIC
void readUltrasonic(float dt) {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH, 15000);
  if (duration > 0) {
    float d = duration * 0.034 / 2.0;
    if (d >= 2.0 && d <= 60.0) distanceCm = d;
  }
  if (lastDistanceCm > 0 && distanceCm < DIST_DETECT_THRESHOLD_CM) {
    float v = (lastDistanceCm - distanceCm) / dt;
    if (v < 0) v = 0;
    objSpeedCmS = 0.3 * v + 0.7 * objSpeedCmS;
  } else {
    objSpeedCmS = 0.0;
  }
  lastDistanceCm = distanceCm;
}
#endif

void setup() {
  Serial.begin(115200);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  // The L298N is a bipolar (not MOSFET) H-bridge: its switching losses/saturation drop get much
  // worse at high PWM frequency, so its effective output voltage collapses well before 20 kHz -
  // even at duty=255 the motor barely turns. The Arduino UNO's analogWrite() runs at ~490-980 Hz,
  // which is why the same wiring worked there; match that here instead of ESP32's usual 20 kHz default.
  ledcAttach(ENA, 1000, 8);
  ledcWrite(ENA, 0);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, FALLING);

#if ULTRASONIC
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
#endif

  pid.SetMode(MANUAL);
  pid.SetOutputLimits(40, 255);              // below ~40/255 the motor doesn't turn (same floor as the UNO sketch)

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());
  // Sanity check: the broker must be reachable on the SAME network as this IP above - if MQTT_HOST
  // (secrets.h) is on a different subnet, "MQTT: connecting..." below will print "failed, state=-2/-4" forever.
  Serial.printf("MQTT broker configured as: %s:%u\n", MQTT_HOST, MQTT_PORT);
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
    if (millis() - lastConnected > FAILSAFE_MS && dirCmd != 'S') {
      dirCmd = 'S'; applyDirection(); stopMotor();
    }
  }

  // ---- speed loop every SAMPLE_MS
  if (millis() - lastSample >= SAMPLE_MS) {
    float dt = (millis() - lastSample) / 1000.0;
    lastSample = millis();

    // --- RPM from the pulse-interval buffer (same method as the UNO sketch) ---
    noInterrupts();
    unsigned long sumIntervals = 0;
    int validSamples = 0;
    for (int i = 0; i < NUM_INTERVALS; i++) {
      if (intervalBuf[i] > 0) { sumIntervals += intervalBuf[i]; validSamples++; }
    }
    unsigned long sinceLastPulse = micros() - lastPulseTime;
    interrupts();

    float rpmInstant = 0.0;
    if (sinceLastPulse > 400000) {                       // motor stopped: timeout
      rpmInstant = 0.0;
      noInterrupts();
      for (int i = 0; i < NUM_INTERVALS; i++) intervalBuf[i] = 0;
      interrupts();
    } else if (validSamples > 0) {
      unsigned long avgInterval = sumIntervals / validSamples;
      float calc = (60.0e6 / PPR) / avgInterval;
      rpmInstant = (calc <= MAX_RPM_PHYSICAL) ? calc : rpmFiltered;
    }

    // deadband + EMA filter (unchanged from the UNO sketch)
    float rpmTemp = 0.2 * rpmInstant + 0.8 * rpmFiltered;
    if (fabsf(rpmTemp - rpmFiltered) >= 2.5 || rpmInstant == 0.0) rpmFiltered = rpmTemp;

    float spMag = MAX_RPM * speedPct / 100.0;
    if (dirCmd == 'S' || spMag < 1.0) {
      pwmOut = 0;
    } else {
      pidSetpoint = spMag;
      pidInput = rpmFiltered;
      pid.Compute();
      pwmOut = (int)pidOutput;
    }
    ledcWrite(ENA, pwmOut);

#if ULTRASONIC
    readUltrasonic(dt);
#endif

    if (mqtt.connected()) {
      float sp = (dirCmd == 'R' ? -1 : (dirCmd == 'F' ? 1 : 0)) * spMag;
      float rpmSigned = (dirCmd == 'R' ? -1 : 1) * rpmFiltered;   // sign from dirCmd, see NOTE ON DIRECTION above
      char msg[128];
#if ULTRASONIC
      snprintf(msg, sizeof(msg), "{\"rpm\":%.1f,\"setpoint\":%.1f,\"output\":%d,\"dir\":\"%c\",\"distance_cm\":%.1f,\"obj_speed_cm_s\":%.1f}",
               rpmSigned, sp, pwmOut, dirCmd, distanceCm, objSpeedCmS);
#else
      snprintf(msg, sizeof(msg), "{\"rpm\":%.1f,\"setpoint\":%.1f,\"output\":%d,\"dir\":\"%c\"}", rpmSigned, sp, pwmOut, dirCmd);
#endif
      mqtt.publish("conveyor/telemetry", msg);
    }
  }
}
