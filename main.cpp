#include <Arduino.h>
#include <TM1637.h>

// --- Display Configuration ---
const int CLK = 42;
const int DIO = 41;
TM1637 tm(CLK, DIO); 

const int LED_PIN = 38; 
const float DETECTION_THRESHOLD = 50.0; 

void setup() {
  Serial.begin(115200); 
  
  // akj7 initialization
  tm.begin();
  tm.onMode();
  tm.setBrightness(7); 
  tm.clearScreen();  
  
  neopixelWrite(LED_PIN, 0, 0, 50);
  Serial.println("ESP32 Ready. Waiting for drone confidence data...");
}

void loop() {

  if (Serial.available() > 0) {
    
    String incomingData = Serial.readStringUntil('\n');
    incomingData.trim(); 
    
    if (incomingData.startsWith(">")) {
      String valueString = incomingData.substring(1);
      float confidence = valueString.toFloat();
        
      // 1. Round to the nearest whole number
      // Adding 0.5 before casting to int performs a standard round
      int roundedValue = (int)(confidence + 0.5);

      // 2. Constrain between 0 and 100
      if (roundedValue > 100) roundedValue = 100;
      if (roundedValue < 0)   roundedValue = 0;

      // 3. Clear the display first to prevent "ghost" digits 
      tm.clearScreen();

      uint8_t offset = 0;
      if (roundedValue < 10) offset = 3;      // Single digit (Position 4)
      else if (roundedValue < 100) offset = 2; // Two digits (Position 3 & 4)
      else offset = 1;                         // "100" (Position 2, 3 & 4)

      tm.display(roundedValue, false, false, offset);    

      if (confidence >= DETECTION_THRESHOLD) {
        neopixelWrite(LED_PIN, 255, 0, 0); 
      } else {
        neopixelWrite(LED_PIN, 0, 100, 0);  
      }
    }
  }
}
