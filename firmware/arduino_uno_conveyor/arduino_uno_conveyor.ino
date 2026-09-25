/*
  arduino_uno_conveyor.ino - conveyor controller, talks to the digital twin over USB serial
  (counterpart of serial_bridge.py).

  Board:     Arduino UNO.  Library: "PID" (PID_v1) by Brett Beauregard.
  Motor:     DC motor + KY-040-style encoder (CLK on ENC_A) + XY-15AS driver.
  Sensor:    HC-SR04 (optional, see ULTRASONIC).

  Motor control is the validated standalone sketch (same pins, same IN1/IN2 polarity, same
  encoder filter, same PID gains/limits, same 0-100 % -> 0-200 RPM speed mapping, same default
  setpoint). Only the telemetry format (JSON instead of CSV) and a few extra commands are new.

  Serial: 115200 baud (9600 is too slow for 10 JSON lines/s). In the Arduino IDE Serial Monitor
  pick 115200 and "Newline", then type e.g.  V50  and  F  exactly like with the old sketch.

  Commands (one per line):
    F | R | S                        forward / reverse / stop
    V<0..100>                         speed setpoint, % of MAX_RPM
    P<float> | I<float> | D<float>   Kp / Ki / Kd
    H                                 keep-alive sent by the PC app every second

  Safety: the failsafe (stop if no command for FAILSAFE_MS) only arms after the first H, i.e.
  when the PC app is driving it - manual testing from the Serial Monitor behaves like before.

  Telemetry, one JSON line every 100 ms:
    {"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F","speed_pct":50,"distance_cm":58.5,"obj_speed_cm_s":0.0}
  NOTE: the encoder is single channel, so the sign of rpm comes from the commanded direction.
  distance_cm = -1 means no object in range; obj_speed_cm_s > 0 means approaching the sensor.
*/
#include <PID_v1.h>

// ── pins (same as the validated sketch) ───────────────────────────────
#define ENC_A 2                               // KY-040 CLK (interrupt)
#define ENC_B 3                               // KY-040 DT (not used by the RPM method)
#define TRIG  9                               // HC-SR04
#define ECHO  10                              // HC-SR04
#define IN1   4                               // XY-15AS
#define IN2   5                               // XY-15AS
#define ENA   6                               // XY-15AS PWM
#define ULTRASONIC 1                          // 0 if the HC-SR04 is not wired

// ── conveyor ───────────────────────────────────────────────────────────
const float PPR = 20.0;                       // 60e6 / PPR = 3,000,000 as in the old sketch
const float MAX_RPM = 300.0;                  // V100 -> 300 RPM; must match config.py
const float MAX_RPM_PHYSICAL = 350.0;         // RPM estimates above this are glitches (must be > MAX_RPM;
                                              // the 8 ms ISR lockout caps readings at 375 anyway)
const unsigned long SAMPLE_MS = 100;
const unsigned long FAILSAFE_MS = 3000;

// ── start-up kick (static friction) ─────────────────────────────────────
// From standstill the PID's minimum output (40) can't break the belt loose, so the belt sat still
// while the integral wound up, then lurched. Instead: full-ish PWM for KICK_MS, then the PID takes
// over starting from RUN_START_PWM. If the belt stalls again (no pulse for 400 ms) it re-kicks.
const int KICK_PWM = 180;
const unsigned long KICK_MS = 150;
const int RUN_START_PWM = 70;                 // tune: lowest PWM that keeps the belt moving once started
const unsigned long REKICK_MS = 1000;         // min time between kicks

// ── encoder: pulse-interval buffer ───────────────────────────────────────
#define NUM_INTERVALS 4
const unsigned long MIN_INTERVAL_US = 8000;   // lockout between accepted pulses (375 RPM max)
const unsigned long MIN_HIGH_US = 1500;       // a falling edge only counts if the line was HIGH this long
                                              // before it (rejects chatter; half-period at 300 RPM = 5 ms)
const unsigned long STOP_TIMEOUT_US = 400000; // no pulse for this long = stopped
volatile unsigned long intervalBuf[NUM_INTERVALS] = {0, 0, 0, 0};
volatile byte intervalIdx = 0;
volatile unsigned long lastPulseTime = 0;
volatile unsigned long lastEdgeTime = 0;
volatile unsigned int rawEdges = 0;           // every falling edge seen (diagnostic)
volatile unsigned int acceptedPulses = 0;     // falling edges that passed the filters (diagnostic)
float rpmFiltered = 0;

// CHANGE interrupt: rising edges only restart the HIGH timer; a falling edge is a pulse only if
// the line was stably HIGH before it. Chatter while a slot edge passes slowly (or PWM noise) makes
// short HIGH blips, which the old FALLING + 8 ms lockout counted as extra pulses -> RPM read too high.
void encoderISR() {
  unsigned long now = micros();
  bool high = PIND & _BV(PD2);                // = digitalRead(ENC_A) on the UNO, ~30x faster (keeps the ISR
                                              // short so a noisy encoder line can't starve the serial RX)
  unsigned long held = now - lastEdgeTime;
  lastEdgeTime = now;
  if (high) return;
  rawEdges++;
  if (held < MIN_HIGH_US) return;

  unsigned long dt = now - lastPulseTime;
  if (dt <= MIN_INTERVAL_US) return;
  lastPulseTime = now;
  acceptedPulses++;
  if (dt > STOP_TIMEOUT_US) return;           // first pulse after standstill: no valid interval yet
  intervalBuf[intervalIdx] = dt;
  intervalIdx = (intervalIdx + 1) % NUM_INTERVALS;
}

char dirCmd = 'S';
bool motorOn = false;
float speedPct = 50;                          // 50 % = 100 RPM, the old sketch's default setpoint
double kp = 0.4, ki = 0.5, kd = 0.0;
double setpoint = 0, input = 0, output = 0;
PID pid(&input, &output, &setpoint, kp, ki, kd, DIRECT);

#if ULTRASONIC
// The app converts distance -> belt position (config.ULTRASONIC); here we only report a clean distance.
const float US_MIN_CM = 2.0;                  // HC-SR04 blind zone
const float US_MAX_CM = 80.0;                 // beyond the far end of the belt (68 cm + 6 cm offset) = no object
const unsigned long US_TIMEOUT_US = 6000;     // echo time for ~100 cm; also bounds how long pulseIn blocks
const byte US_MAX_MISSES = 3;                 // keep the last reading through this many bad echoes
const float US_MAX_STEP_CM = 6.0;             // > max belt travel in 100 ms (~4 cm): a bigger jump = new object
float distanceCm = -1, lastDistanceCm = -1, objSpeedCmS = 0;   // -1 = no object in range
byte usMisses = 0;
#endif

unsigned long lastSample = 0, lastRx = 0;
bool hostLinked = false;                      // true once the PC app has sent a keep-alive
bool kicking = false;
unsigned long kickStart = 0;

void driveForward() {                         // same polarity as the validated sketch
  digitalWrite(IN2, HIGH);
  digitalWrite(IN1, LOW);
  dirCmd = 'F';
  motorOn = true;
  pid.SetMode(AUTOMATIC);
}

void driveReverse() {
  digitalWrite(IN2, LOW);
  digitalWrite(IN1, HIGH);
  dirCmd = 'R';
  motorOn = true;
  pid.SetMode(AUTOMATIC);
}

void stopMotor() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  dirCmd = 'S';
  motorOn = false;
  pid.SetMode(MANUAL);
  output = 0;
  analogWrite(ENA, 0);
  rpmFiltered = 0;
  kicking = false;
  kickStart = millis() - REKICK_MS;           // next start kicks immediately
}

#if ULTRASONIC
void readUltrasonic() {
  digitalWrite(TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  long duration = pulseIn(ECHO, HIGH, US_TIMEOUT_US);
  float d = duration * 0.034 / 2.0;
  if (duration > 0 && d >= US_MIN_CM && d <= US_MAX_CM) {
    distanceCm = d;
    usMisses = 0;
  } else if (++usMisses >= US_MAX_MISSES) {  // a single lost echo keeps the last value; several = gone
    usMisses = US_MAX_MISSES;
    distanceCm = -1;
  }

  // Object speed, positive = approaching the sensor (signed, so it also works in reverse).
  if (lastDistanceCm > 0 && distanceCm > 0 && fabs(lastDistanceCm - distanceCm) <= US_MAX_STEP_CM) {
    float v = (lastDistanceCm - distanceCm) / 0.1;
    objSpeedCmS = 0.3 * v + 0.7 * objSpeedCmS;
  } else {
    objSpeedCmS = 0.0;
  }
  lastDistanceCm = distanceCm;
}
#endif

// One complete line, e.g. "V50" or "P0.4". A value that doesn't parse is ignored instead of becoming
// 0 (Serial.parseFloat() returned 0 on a lost/garbled byte, and blocked the loop up to 1 s meanwhile).
void handleLine(char *line) {
  char *end;
  double val = strtod(line + 1, &end);
  bool hasVal = end != line + 1 && *end == '\0';
  switch (line[0]) {
    case 'F': driveForward(); break;
    case 'R': driveReverse(); break;
    case 'S': stopMotor(); break;
    case 'V': if (hasVal) speedPct = constrain(val, 0, 100); break;
    case 'P': if (hasVal && val >= 0) { kp = val; pid.SetTunings(kp, ki, kd); } break;
    case 'I': if (hasVal && val >= 0) { ki = val; pid.SetTunings(kp, ki, kd); } break;
    case 'D': if (hasVal && val >= 0) { kd = val; pid.SetTunings(kp, ki, kd); } break;
    case 'H': hostLinked = true; break;
  }
}

char rxBuf[16];
byte rxLen = 0;
bool rxOverflow = false;

void setup() {
  Serial.begin(115200);
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, CHANGE);

  pid.SetMode(MANUAL);
  pid.SetOutputLimits(40, 255);
  stopMotor();

  lastSample = lastRx = millis();
}

void loop() {
  // ---- commands (non-blocking: collect a line, run it on '\n') ----
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (rxLen > 0 && !rxOverflow) {
        rxBuf[rxLen] = '\0';
        lastRx = millis();
        handleLine(rxBuf);
      }
      rxLen = 0;
      rxOverflow = false;
    } else if (c != ' ') {
      if (rxLen < sizeof(rxBuf) - 1) rxBuf[rxLen++] = c;
      else rxOverflow = true;
    }
  }

  // ---- failsafe (only while the PC app is in charge) ----
  if (hostLinked && motorOn && millis() - lastRx > FAILSAFE_MS) {
    stopMotor();
    hostLinked = false;
  }

  if (millis() - lastSample < SAMPLE_MS) return;
  lastSample = millis();

#if ULTRASONIC
  readUltrasonic();
#endif

  // ---- RPM from the pulse-interval buffer ----
  noInterrupts();
  unsigned long sumIntervals = 0;
  int validSamples = 0;
  for (int i = 0; i < NUM_INTERVALS; i++) {
    if (intervalBuf[i] > 0) { sumIntervals += intervalBuf[i]; validSamples++; }
  }
  unsigned long sinceLastPulse = micros() - lastPulseTime;
  unsigned int edges = rawEdges, pulses = acceptedPulses;
  rawEdges = acceptedPulses = 0;
  interrupts();

  float rpmInstant = 0.0;
  if (sinceLastPulse > STOP_TIMEOUT_US) {
    noInterrupts();
    for (int i = 0; i < NUM_INTERVALS; i++) intervalBuf[i] = 0;
    interrupts();
  } else if (validSamples > 0) {
    float calc = (60.0e6 / PPR) / (sumIntervals / validSamples);
    rpmInstant = (calc <= MAX_RPM_PHYSICAL) ? calc : rpmFiltered;
  }

  // Plain EMA. The old "deadband" compared 0.2*(instant - filtered) against 2.5, i.e. it froze the
  // reading until the real speed moved >12.5 RPM away, then jumped - and the PID integrated meanwhile.
  rpmFiltered = 0.2 * rpmInstant + 0.8 * rpmFiltered;

  // ---- PID + motor (same structure as the validated sketch) ----
  setpoint = MAX_RPM * speedPct / 100.0;
  if (motorOn) {
    input = rpmFiltered;
    bool stalled = sinceLastPulse > STOP_TIMEOUT_US;
    if (kicking && millis() - kickStart >= KICK_MS) {   // kick over: hand over to the PID
      kicking = false;
      output = RUN_START_PWM;
      pid.SetMode(MANUAL);
      pid.SetMode(AUTOMATIC);                           // re-initialises the PID's integral from output
    }
    if (!kicking && stalled && millis() - kickStart >= REKICK_MS) {
      kicking = true;
      kickStart = millis();
    }
    if (kicking) output = KICK_PWM;
    else pid.Compute();
    analogWrite(ENA, (int)output);
  } else {
    output = 0;
    analogWrite(ENA, 0);
  }

  // ---- telemetry ----
  int sign = dirCmd == 'R' ? -1 : 1;
  Serial.print(F("{\"rpm\":"));
  Serial.print(sign * rpmFiltered, 1);
  Serial.print(F(",\"setpoint\":"));
  Serial.print(motorOn ? sign * setpoint : 0.0, 1);
  Serial.print(F(",\"output\":"));
  Serial.print((int)output);
  Serial.print(F(",\"dir\":\""));
  Serial.print(dirCmd);
  Serial.print(F("\",\"speed_pct\":"));
  Serial.print(speedPct, 0);
  Serial.print(F(",\"edges\":"));             // diagnostic: falling edges seen this sample
  Serial.print(edges);
  Serial.print(F(",\"pulses\":"));            // diagnostic: edges accepted as encoder pulses
  Serial.print(pulses);
  Serial.print(F(",\"kp\":"));                // diagnostic: gains the board is actually using
  Serial.print(kp, 2);
  Serial.print(F(",\"ki\":"));
  Serial.print(ki, 2);
  Serial.print(F(",\"kd\":"));
  Serial.print(kd, 2);
#if ULTRASONIC
  Serial.print(F(",\"distance_cm\":"));
  Serial.print(distanceCm, 1);
  Serial.print(F(",\"obj_speed_cm_s\":"));
  Serial.print(objSpeedCmS, 1);
#endif
  Serial.println(F("}"));
}
