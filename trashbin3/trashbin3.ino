#include <SoftwareSerial.h>

SoftwareSerial BT(11, 12); // RX, TX

// --- Motor Pins ---
const int LeftMotorPin1 = 5;
const int LeftMotorPin2 = 4;
const int enableLeft = 3; 

const int RightMotorPin1 = 7; 
const int RightMotorPin2 = 8;
const int enableRight = 9;

String inputString = "";
boolean stringComplete = false;

void setup() {
  BT.begin(9600);
  Serial.begin(9600);
  
  pinMode(LeftMotorPin1, OUTPUT);
  pinMode(LeftMotorPin2, OUTPUT);
  pinMode(enableLeft, OUTPUT);
  pinMode(RightMotorPin1, OUTPUT);
  pinMode(RightMotorPin2, OUTPUT);
  pinMode(enableRight, OUTPUT);
  
  stopCar();
}

void loop() {
  // Read incoming command line by line (e.g., "M,200,200\n")
  while (BT.available()) {
    char inChar = (char)BT.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (stringComplete) {
    parseCommand(inputString);
    inputString = "";
    stringComplete = false;
  }
}

void parseCommand(String command) {
  // Expected format: "M,LeftSpeed,RightSpeed"
  // Speed range: -255 (Max Reverse) to 255 (Max Forward)
  
  if (command.startsWith("M")) {
    int firstComma = command.indexOf(',');
    int secondComma = command.lastIndexOf(',');
    
    if (firstComma > 0 && secondComma > firstComma) {
      String leftStr = command.substring(firstComma + 1, secondComma);
      String rightStr = command.substring(secondComma + 1);
      
      int leftSpeed = leftStr.toInt();
      int rightSpeed = rightStr.toInt();
      
      setMotors(leftSpeed, rightSpeed);
    }
  }
}

void setMotors(int left, int right) {
  // --- LEFT MOTOR (Inverted logic from previous version) ---
  if (left > 0) {
    // FORWARD
    digitalWrite(LeftMotorPin1, LOW); 
    digitalWrite(LeftMotorPin2, HIGH);
    analogWrite(enableLeft, left);
  } else if (left < 0) {
    // REVERSE
    digitalWrite(LeftMotorPin1, HIGH); 
    digitalWrite(LeftMotorPin2, LOW);
    analogWrite(enableLeft, abs(left));
  } else {
    // STOP
    digitalWrite(LeftMotorPin1, LOW); 
    digitalWrite(LeftMotorPin2, LOW);
    analogWrite(enableLeft, 0);
  }

  // --- RIGHT MOTOR (Inverted logic from previous version) ---
  if (right > 0) {
    // FORWARD
    digitalWrite(RightMotorPin1, LOW); 
    digitalWrite(RightMotorPin2, HIGH);
    analogWrite(enableRight, right);
  } else if (right < 0) {
    // REVERSE
    digitalWrite(RightMotorPin1, HIGH); 
    digitalWrite(RightMotorPin2, LOW);
    analogWrite(enableRight, abs(right));
  } else {
    // STOP
    digitalWrite(RightMotorPin1, LOW); 
    digitalWrite(RightMotorPin2, LOW);
    analogWrite(enableRight, 0);
  }
}

void stopCar() {
  setMotors(0, 0);
}