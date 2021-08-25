// Programmer Name  : Ms.Ang Pei Jin, Asia Pacific University Student
// Program Name     : Social Distancing Detector
// Description      : Detect physical distance between two individuals
// First written on : Monday, 24 May 2021
// Edited on        : Sunday, 30 May 2021

int buzzer = 11;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  if(Serial.available() > 0){
    if(Serial.read() == 's'){
      tone(buzzer,450);
      delay(100);
      noTone(buzzer);
      delay(100);
    }
  }
}
