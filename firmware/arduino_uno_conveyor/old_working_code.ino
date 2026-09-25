#include <PID_v1.h>

// Definición de pines
#define ENC_A 2 // KY-040 CLK (Interrupción)
#define ENC_B 3 // KY-040 DT
#define TRIG  9 // HC-SR04 Trigger
#define ECHO  10 // HC-SR04 Echo
#define IN1   4 // Driver Motor
#define IN2   5 // Driver Motor
#define ENA   6 // PWM Driver Motor

// Parámetro de límite físico real del motor
const float MAX_RPM_FISICAS = 250.0; 

// --- VARIABLES DEL ENCODER Y BUFFER DE INTERVALOS ---
#define NUM_INTERVALOS 4
volatile unsigned long bufferIntervalos[NUM_INTERVALOS] = {0, 0, 0, 0};
volatile byte idxIntervalo = 0;
volatile unsigned long lastPulseTime = 0;

unsigned long lastControlTime = 0;
bool motorEncendido = false;

// Variable de RPM filtrada final
float rpmFiltrada = 0;

// Variables del PID
double setpoint = 100; // RPM deseadas
double input = 0;
double output = 0;
double Kp = 0.4, Ki = 0.5, Kd = 0.0;
PID myPID(&input, &output, &setpoint, Kp, Ki, Kd, DIRECT);

// Estado reportado a la app (app.py / serial_bridge.py)
char dirCmd = 'S';   // 'F', 'R' o 'S'
int speedPct = 50;   // último V recibido (50 % = 100 RPM, el setpoint por defecto)

// Variables del Sensor Ultrasónico y Velocidad del Objeto
long duration;
float distance = 0;
float lastDistance = 0;
float velocidadObjeto = 0;
const float DIST_UMBRAL_DETECCION = 55.0;

void setup() {
  Serial.begin(115200); // config.SERIAL["baud"]; 9600 es muy lento para 10 líneas JSON/s
  pinMode(ENC_A, INPUT_PULLUP);
  pinMode(ENC_B, INPUT_PULLUP);
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);

  attachInterrupt(digitalPinToInterrupt(ENC_A), encoderISR, FALLING);

  myPID.SetMode(MANUAL);
  myPID.SetOutputLimits(40, 255);

  lastControlTime = millis();
}

void loop() {
  if (millis() - lastControlTime >= 100) {

    // --- 1. ULTRASONIDO (HC-SR04) ---
    digitalWrite(TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);

    duration = pulseIn(ECHO, HIGH, 15000);
    if (duration > 0) {
      float distMedida = duration * 0.034 / 2.0;
      if (distMedida >= 2.0 && distMedida <= 60.0) {
        distance = distMedida;
      }
    }

    // --- 2. VELOCIDAD LINEAL DEL OBJETO (cm/s) ---
    if (lastDistance > 0 && distance < DIST_UMBRAL_DETECCION) {
      float deltaDistancia = lastDistance - distance;
      float velInstantanea = deltaDistancia / 0.1;
      if (velInstantanea < 0) velInstantanea = 0;
      velocidadObjeto = (0.3 * velInstantanea) + (0.7 * velocidadObjeto);
    } else {
      velocidadObjeto = 0.0;
    }
    lastDistance = distance;

    // --- 3. CÁLCULO DE RPM CON PROMEDIO DE INTERVALOS (MULTI-PULSO) ---
    noInterrupts();
    unsigned long sumaIntervalos = 0;
    int muestrasValidas = 0;
    for (int i = 0; i < NUM_INTERVALOS; i++) {
      if (bufferIntervalos[i] > 0) {
        sumaIntervalos += bufferIntervalos[i];
        muestrasValidas++;
      }
    }
    unsigned long timeSinceLastPulse = micros() - lastPulseTime;
    interrupts();

    float rpmInstantanea = 0.0;

    if (timeSinceLastPulse > 400000) { // Timeout de motor detenido
      rpmInstantanea = 0.0;
      // Limpiar buffer al detenerse
      for (int i = 0; i < NUM_INTERVALOS; i++) bufferIntervalos[i] = 0;
    } else if (muestrasValidas > 0) {
      unsigned long avgInterval = sumaIntervalos / muestrasValidas;
      float calculo = 3000000.0 / avgInterval;

      if (calculo <= MAX_RPM_FISICAS) {
        rpmInstantanea = calculo;
      } else {
        rpmInstantanea = rpmFiltrada;
      }
    }

    // --- 4. FILTRO DE ZONA MUERTA (DEADBAND) + EMA SUAVE ---
    float rpmTemp = (0.2 * rpmInstantanea) + (0.8 * rpmFiltrada);

    // Si la variación es menor a 2.5 RPM respecto al estado estable, se ignora la fluctuación
    if (abs(rpmTemp - rpmFiltrada) >= 2.5 || rpmInstantanea == 0.0) {
      rpmFiltrada = rpmTemp;
    }

    // --- 5. CONTROL PID Y SALIDA AL MOTOR ---
    if (motorEncendido) {
      input = rpmFiltrada;
      myPID.Compute();
      analogWrite(ENA, (int)output);
    } else {
      output = 0;
      analogWrite(ENA, 0);
    }

    // --- 6. TELEMETRÍA SERIAL PARA PYTHON / DASH (una línea JSON, ver serial_bridge.py) ---
    // El encoder es de un canal: el signo de rpm/setpoint sale de la dirección comandada.
    int signo = (dirCmd == 'R') ? -1 : 1;
    Serial.print(F("{\"rpm\":"));
    Serial.print(signo * rpmFiltrada, 1);
    Serial.print(F(",\"setpoint\":"));
    Serial.print(motorEncendido ? signo * setpoint : 0.0, 1);
    Serial.print(F(",\"output\":"));
    Serial.print((int)output);
    Serial.print(F(",\"dir\":\""));
    Serial.print(dirCmd);
    Serial.print(F("\",\"speed_pct\":"));
    Serial.print(speedPct);
    Serial.print(F(",\"distance_cm\":"));
    Serial.print(distance, 1);
    Serial.print(F(",\"obj_speed_cm_s\":"));
    Serial.print(velocidadObjeto, 1);
    Serial.println(F("}"));

    lastControlTime = millis();
  }

  // --- 7. RECEPCIÓN DE COMANDOS SERIALES ---
  if (Serial.available() > 0) {
    char cmd = Serial.read();

    // Comandos de la app (uno por línea): F | R | S | V<0..100> | P<f> | I<f> | D<f> | H (keep-alive)
    if (cmd != '\n' && cmd != '\r') {
      switch (cmd) {
        case 'F':
          digitalWrite(IN2, HIGH);
          digitalWrite(IN1, LOW);
          motorEncendido = true;
          myPID.SetMode(AUTOMATIC);
          dirCmd = 'F';
          break;

        case 'R':
          digitalWrite(IN2, LOW);
          digitalWrite(IN1, HIGH);
          motorEncendido = true;
          myPID.SetMode(AUTOMATIC);
          dirCmd = 'R';
          break;

        case 'S':
          digitalWrite(IN1, LOW);
          digitalWrite(IN2, LOW);
          motorEncendido = false;
          myPID.SetMode(MANUAL);
          output = 0;
          analogWrite(ENA, 0);
          rpmFiltrada = 0;
          dirCmd = 'S';
          break;

        case 'V': {
          int vel = Serial.parseInt();
          speedPct = vel;
          setpoint = map(vel, 0, 100, 0, 200);
          break;
        }

        case 'P':
          Kp = Serial.parseFloat();
          myPID.SetTunings(Kp, Ki, Kd);
          break;

        case 'I':
          Ki = Serial.parseFloat();
          myPID.SetTunings(Kp, Ki, Kd);
          break;

        case 'D':
          Kd = Serial.parseFloat();
          myPID.SetTunings(Kp, Ki, Kd);
          break;

        case 'H':   // keep-alive de la app: no hace nada
          break;
      }
    }
  }
}

// ISR que llena el buffer circular de 4 mediciones
void encoderISR() {
  unsigned long now = micros();
  unsigned long dt = now - lastPulseTime;

  if (dt > 8000) {
    bufferIntervalos[idxIntervalo] = dt;
    idxIntervalo = (idxIntervalo + 1) % NUM_INTERVALOS;
    lastPulseTime = now;
  }
}