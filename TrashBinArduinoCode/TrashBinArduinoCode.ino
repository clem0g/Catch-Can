#include <SoftwareSerial.h>

SoftwareSerial BT(11, 12); // RX, TX

// Pins
const int LeftMotorPin1 = 5;
const int LeftMotorPin2 = 4;
const int enableLeft = 3; 
const int RightMotorPin1 = 7; 
const int RightMotorPin2 = 8;
const int enableRight = 9;

const int MIN_POWER = 150; 

void setup() {
  BT.begin(9600);
  
  pinMode(LeftMotorPin1, OUTPUT);
  pinMode(LeftMotorPin2, OUTPUT);
  pinMode(enableLeft, OUTPUT);
  pinMode(RightMotorPin1, OUTPUT);
  pinMode(RightMotorPin2, OUTPUT);
  pinMode(enableRight, OUTPUT);
}

void loop() {
  if (BT.available()) {
    char cmd = BT.read();
    
    if (cmd == 'F') {       // Forward
      move(true, true);
    } else if (cmd == 'L') { // Left
      move(false, true);
    } else if (cmd == 'R') { // Right
      move(true, false);
    } else {                // Stop
      stopCar();
    }
  }
}

void move(bool leftFwd, bool rightFwd) {
  // Left Motor
  if (leftFwd) { digitalWrite(LeftMotorPin1, HIGH); digitalWrite(LeftMotorPin2, LOW); }
  else         { digitalWrite(LeftMotorPin1, LOW); digitalWrite(LeftMotorPin2, HIGH); }
  
  // Right Motor
  if (rightFwd) { digitalWrite(RightMotorPin1, HIGH); digitalWrite(RightMotorPin2, LOW); }
  else          { digitalWrite(RightMotorPin1, LOW); digitalWrite(RightMotorPin2, HIGH); }

  analogWrite(enableLeft, MIN_POWER);
  analogWrite(enableRight, MIN_POWER);
}

void stopCar() {
  analogWrite(enableLeft, 0);
  analogWrite(enableRight, 0);
}