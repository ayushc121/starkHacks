#include <Arduino.h>

#define NUM_BINS 16

int fftBins[NUM_BINS];

void parseFFT(String data) {
  int idx = 0;
  int last = 0;

  for (int i = 0; i <= data.length(); i++) {
    if (data[i] == ',' || i == data.length()) {
      if (idx < NUM_BINS) {
        fftBins[idx++] = data.substring(last, i).toInt();
      }
      last = i + 1;
    }
  }
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');

    // DEBUG: print raw incoming data
    Serial.print("RX: ");
    Serial.println(line);

    if (line.startsWith(">fft:")) {
      String payload = line.substring(5);
      parseFFT(payload);

      Serial.println("Parsed FFT!");
    }
  }
}