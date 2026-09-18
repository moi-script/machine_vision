// AeroSense feeder aim: two hobby servos (X pan, Y tilt) driven over USB serial.
//
// Protocol (115200 baud, one command per line):
//   A <x> <y>   move to angles (clamped to [ANGLE_MIN, ANGLE_MAX]) -> "OK <x> <y>"
//   C           center                                            -> "OK <cx> <cy>"
//   P           ping                                              -> "PONG"
// On boot it centers and prints READY. With no command for IDLE_MS it
// re-centers, so a crashed backend never leaves the feeder aimed off-court.
//
// Power the servos from their own 5-6 V supply. Tie its GND to the board's
// GND. Never run them from the board's 5 V pin.

#if defined(ESP32)
  #include <ESP32Servo.h>
  const int PIN_X = 18;
  const int PIN_Y = 19;
#else
  #include <Servo.h>
  const int PIN_X = 9;
  const int PIN_Y = 10;
#endif

const int ANGLE_MIN = 30;
const int ANGLE_MAX = 150;
const int CENTER = 90;
const unsigned long IDLE_MS = 60000UL;

Servo servoX;
Servo servoY;
char line[32];
byte len = 0;
unsigned long lastCmd = 0;
bool centeredForIdle = false;

int clampAngle(long v) {
  if (v < ANGLE_MIN) return ANGLE_MIN;
  if (v > ANGLE_MAX) return ANGLE_MAX;
  return (int)v;
}

void moveTo(int x, int y) {
  servoX.write(x);
  servoY.write(y);
  Serial.print(F("OK "));
  Serial.print(x);
  Serial.print(' ');
  Serial.println(y);
}

void handle(char *cmd) {
  lastCmd = millis();
  centeredForIdle = false;
  if (cmd[0] == 'A') {
    char *p = cmd + 1;
    long x = strtol(p, &p, 10);
    long y = strtol(p, &p, 10);
    moveTo(clampAngle(x), clampAngle(y));
  } else if (cmd[0] == 'C') {
    moveTo(CENTER, CENTER);
  } else if (cmd[0] == 'P') {
    Serial.println(F("PONG"));
  } else {
    Serial.println(F("ERR"));
  }
}

void setup() {
  Serial.begin(115200);
  servoX.attach(PIN_X);
  servoY.attach(PIN_Y);
  servoX.write(CENTER);
  servoY.write(CENTER);
  lastCmd = millis();
  Serial.println(F("READY"));
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = '\0';
      if (len) handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
  if (!centeredForIdle && millis() - lastCmd > IDLE_MS) {
    servoX.write(CENTER);
    servoY.write(CENTER);
    centeredForIdle = true;
  }
}
