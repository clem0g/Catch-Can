#include <SoftwareSerial.h>

SoftwareSerial BT(11, 12); // RX, TX

// --- Motor Pins ---
const int LeftMotorPin1 = 5;
const int LeftMotorPin2 = 4;
const int enableLeft = 3; 

const int RightMotorPin1 = 7; 
const int RightMotorPin2 = 8;
const int enableRight = 9;

// --- Speed Settings ---
// We use a lower power for turning to be precise
const int DRIVE_POWER = 150; 
const int TURN_POWER = 160;  // Slightly higher to overcome friction of one wheel

void setup() {
  BT.begin(9600);
  
  pinMode(LeftMotorPin1, OUTPUT);
  pinMode(LeftMotorPin2, OUTPUT);
  pinMode(enableLeft, OUTPUT);
  pinMode(RightMotorPin1, OUTPUT);
  pinMode(RightMotorPin2, OUTPUT);
  pinMode(enableRight, OUTPUT);
  
  stopCar();
}

void loop() {
  if (BT.available()) {
    char cmd = BT.read();
    
    if (cmd == 'F') {       
      moveForward();
    } else if (cmd == 'L') { 
      turnLeft();
    } else if (cmd == 'R') { 
      turnRight();
    } else {                
      stopCar();
    }
  }
}

void moveForward() {
  // Both motors forward
  digitalWrite(LeftMotorPin1, HIGH); digitalWrite(LeftMotorPin2, LOW);
  digitalWrite(RightMotorPin1, HIGH); digitalWrite(RightMotorPin2, LOW);
  analogWrite(enableLeft, DRIVE_POWER);
  analogWrite(enableRight, DRIVE_POWER);
}

void turnLeft() {
  // PIVOT LEFT: Stop Left Wheel, Drive Right Wheel
  // This is much smoother than spinning back/forth
  digitalWrite(LeftMotorPin1, LOW); digitalWrite(LeftMotorPin2, LOW); // Stop Left
  digitalWrite(RightMotorPin1, HIGH); digitalWrite(RightMotorPin2, LOW); // Fwd Right
  
  analogWrite(enableLeft, 0);       // Cut power to pivot point
  analogWrite(enableRight, TURN_POWER);
}

void turnRight() {
  // PIVOT RIGHT: Drive Left Wheel, Stop Right Wheel
  digitalWrite(LeftMotorPin1, HIGH); digitalWrite(LeftMotorPin2, LOW); // Fwd Left
  digitalWrite(RightMotorPin1, LOW); digitalWrite(RightMotorPin2, LOW); // Stop Right
  
  analogWrite(enableLeft, TURN_POWER);
  analogWrite(enableRight, 0);      // Cut power to pivot point
}

void stopCar() {
  digitalWrite(LeftMotorPin1, LOW); digitalWrite(LeftMotorPin2, LOW);
  digitalWrite(RightMotorPin1, LOW); digitalWrite(RightMotorPin2, LOW);
  analogWrite(enableLeft, 0);
  analogWrite(enableRight, 0);
}
