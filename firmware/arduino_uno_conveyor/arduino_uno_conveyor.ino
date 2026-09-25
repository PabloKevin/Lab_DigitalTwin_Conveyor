/*
  arduino_uno_conveyor.ino - conveyor controller, talks to the digital twin over USB serial
  (counterpart of serial_bridge.py; replaces the ESP32+WiFi+MQTT version - the ESP32's H-bridge
  wiring wasn't driving the motor reliably, so the conveyor controller moved back to a UNO).

  Board:     Arduino UNO.  Library: "PID" (PID_v1) by Brett Beauregard.
  Motor:     DC motor + KY-040-style encoder (single channel, pulses on ENC_A only) + XY-15AS driver.
  Sensor:    HC-SR04 (optional, see ULTRASONIC below) for an independent object-distance/speed reading.

  The encoder measurement (pulse-interval buffer + deadband/EMA filter) and the PID are the ones
  validated on the real conveyor - unchanged from the sketch that was working before the ESP32
  detour. Only the transport (JSON-over-serial instead of WiFi/MQTT) and the PID gains (now settable
  at runtime instead of hardcoded) are new.

  NOTE ON DIRECTION: this encoder method only measures pulse timing (single channel, no quadrature),
  so it cannot sense direction by itself - the sign of the reported RPM is taken from the commanded
  direction (dirCmd), not measured.

  Protocol (must match serial_bridge.py / config.py):
    Arduino -> PC   one JSON line every 100 ms:
                    {"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F","distance_cm":58.5,"obj_speed_cm_s":0.0}
    PC -> Arduino   one line per command:
                    F | R | S                       direction
                    V<0..100>                        speed setpoint, % of MAX_RPM
                    P<float> | I<float> | D<float>   Kp / Ki / Kd
  Safety: if no command line arrives for more than FAILSAFE_MS (the PC sends the current direction
  as a heartbeat even when unchanged, see serial_bridge.py), the motor stops.
*/
#include <PID_v1.h>

// ── pins ───────────────────────────────────────────────────────────────
#define ENC_A 2                               // KY-040 CLK, interrupt input (DT is not needed by this method)
#define IN1   4                               // XY-15AS
#define IN2   5                               // XY-15AS
#define ENA   6                               // XY-15AS PWM
#define ULTRASONIC 1                          // 1 = HC-SR04 wired and read every sample, 0 = disabled
#define TRIG  9                               // HC-SR04 (only used if ULTRASONIC)
#define ECHO  10                              // HC-SR04 (only used if ULTRASONIC)

// ── conveyor ───────────────────────────────────────────────────────────
const float PPR = 20.0;                       // encoder pulses per revolution (lab guide's disk)
const float MAX_RPM = 300.0;                  // RPM at 100 % (must match config.py's MAX_RPM)
const float MAX_RPM_PHYSICAL = 250.0;         // sanity cap: an RPM estimate above this is treated as a glitch
const unsigned long SAMPLE_MS = 100;
const unsigned long FAILSAFE_MS = 3000;

// ── encoder: pulse-interval buffer ───────────────────────────────────────
#define NUM_INTERVALS 4
volatile unsigned long intervalBuf[NUM_INTERVALS] = {0, 0, 0, 0};
volatile byte intervalIdx = 0;
volatile unsigned long lastPulseTime = 0;
float rpmFiltered = 0;

void encoderISR() {
  unsigned long now = micros();
  unsigned long dt = now - lastPulseTime;
  if (dt > 8000) {                            // debounce: ignore glitches faster than 8 ms apart
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

unsigned long lastSample = 0, lastRx = 0;

void applyDirection() {
  digitalWrite(IN1, dirCmd == 'F');
  digitalWrite(IN2, dirCmd == 'R');
}

void stopMotor() {
  pwmOut = 0;
  analogWrite(ENA, 0);
  pid.SetMode(MANUAL);
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
  pinMode(ENA, OUTPUT);
  analogWrite(ENA, 0);
  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, FALLING);

#if ULTRASONIC
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
#endif

  pid.SetMode(MANUAL);
  pid.SetOutputLimits(40, 255);               // below ~40/255 the motor doesn't turn

  lastRx = millis();
  lastSample = millis();
}

void loop() {
  // ---- serial commands (non-blocking) ----
  while (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == '\n' || cmd == '\r') continue;
    lastRx = millis();
    switch (cmd) {
      case 'F':
      case 'R':
        dirCmd = cmd;
        applyDirection();
        pid.SetMode(AUTOMATIC);               // bumpless transfer handled by the PID library
        break;
      case 'S':
        dirCmd = 'S';
        applyDirection();
        stopMotor();
        break;
      case 'V':
        speedPct = constrain(Serial.parseFloat(), 0, 100);
        break;
      case 'P':
        kp = Serial.parseFloat();
        pid.SetTunings(kp, ki, kd);
        break;
      case 'I':
        ki = Serial.parseFloat();
        pid.SetTunings(kp, ki, kd);
        break;
      case 'D':
        kd = Serial.parseFloat();
        pid.SetTunings(kp, ki, kd);
        break;
    }
  }

  // ---- link-loss failsafe: the PC re-sends the direction as a heartbeat even when unchanged ----
  if (millis() - lastRx > FAILSAFE_MS && dirCmd != 'S') {
    dirCmd = 'S';
    applyDirection();
    stopMotor();
  }

  // ---- speed loop every SAMPLE_MS ----
  if (millis() - lastSample >= SAMPLE_MS) {
    float dt = (millis() - lastSample) / 1000.0;
    lastSample = millis();

    // --- RPM from the pulse-interval buffer ---
    noInterrupts();
    unsigned long sumIntervals = 0;
    int validSamples = 0;
    for (int i = 0; i < NUM_INTERVALS; i++) {
      if (intervalBuf[i] > 0) { sumIntervals += intervalBuf[i]; validSamples++; }
    }
    unsigned long sinceLastPulse = micros() - lastPulseTime;
    interrupts();

    float rpmInstant = 0.0;
    if (sinceLastPulse > 400000) {                        // motor stopped: timeout
      rpmInstant = 0.0;
      noInterrupts();
      for (int i = 0; i < NUM_INTERVALS; i++) intervalBuf[i] = 0;
      interrupts();
    } else if (validSamples > 0) {
      unsigned long avgInterval = sumIntervals / validSamples;
      float calc = (60.0e6 / PPR) / avgInterval;
      rpmInstant = (calc <= MAX_RPM_PHYSICAL) ? calc : rpmFiltered;
    }

    // deadband + EMA filter
    float rpmTemp = 0.2 * rpmInstant + 0.8 * rpmFiltered;
    if (fabs(rpmTemp - rpmFiltered) >= 2.5 || rpmInstant == 0.0) rpmFiltered = rpmTemp;

    float spMag = MAX_RPM * speedPct / 100.0;
    if (dirCmd == 'S' || spMag < 1.0) {
      pwmOut = 0;
    } else {
      pidSetpoint = spMag;
      pidInput = rpmFiltered;
      pid.Compute();
      pwmOut = (int)pidOutput;
    }
    analogWrite(ENA, pwmOut);

#if ULTRASONIC
    readUltrasonic(dt);
#endif

    float sp = (dirCmd == 'R' ? -1 : (dirCmd == 'F' ? 1 : 0)) * spMag;
    float rpmSigned = (dirCmd == 'R' ? -1 : 1) * rpmFiltered;   // sign from dirCmd, see NOTE ON DIRECTION above
    Serial.print(F("{\"rpm\":"));
    Serial.print(rpmSigned, 1);
    Serial.print(F(",\"setpoint\":"));
    Serial.print(sp, 1);
    Serial.print(F(",\"output\":"));
    Serial.print(pwmOut);
    Serial.print(F(",\"dir\":\""));
    Serial.print(dirCmd);
    Serial.print(F("\""));
#if ULTRASONIC
    Serial.print(F(",\"distance_cm\":"));
    Serial.print(distanceCm, 1);
    Serial.print(F(",\"obj_speed_cm_s\":"));
    Serial.print(objSpeedCmS, 1);
#endif
    Serial.println(F("}"));
  }
}
