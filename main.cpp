#include <Arduino.h>
#include <math.h>
#include <TM1637.h>

// ===== CONFIG =====
#define ADC_PIN 36
#define FFT_SIZE 16
#define SAMPLE_RATE 4000  // Hz (adjust as needed)
#define SMOOTH_WINDOW 4

#define CLK = 42;
#define DIO = 41;
TM1637 tm(CLK, DIO); 


// ===== GLOBALS =====
float samples[FFT_SIZE];
float real[FFT_SIZE];
float imag[FFT_SIZE];
float smoothed[FFT_SIZE];

float confidence = 0.0;  // REQUIRED OUTPUT

// ===== SIMPLE MOVING AVERAGE =====
float smoothSample(float newSample) {
    static float buffer[SMOOTH_WINDOW] = {0};
    static int idx = 0;
    static float sum = 0;

    sum -= buffer[idx];
    buffer[idx] = newSample;
    sum += newSample;

    idx = (idx + 1) % SMOOTH_WINDOW;
    return sum / SMOOTH_WINDOW;
}

// ===== NAIVE FFT (16-point DFT) =====
void computeFFT() {
    for (int k = 0; k < FFT_SIZE; k++) {
        real[k] = 0;
        imag[k] = 0;

        for (int n = 0; n < FFT_SIZE; n++) {
            float angle = 2 * PI * k * n / FFT_SIZE;
            real[k] += samples[n] * cos(angle);
            imag[k] -= samples[n] * sin(angle);
        }
    }
}

// ===== GET MAGNITUDE =====
void getMagnitudes(float *mag) {
    for (int i = 0; i < FFT_SIZE; i++) {
        mag[i] = sqrt(real[i]*real[i] + imag[i]*imag[i]);
    }
}

// ===== TINY NEURAL NETWORK =====
// 16 inputs → 8 hidden → 1 output
float nn_weights1[FFT_SIZE][8];
float nn_weights2[8];
float nn_bias1[8];
float nn_bias2 = 0;

// Initialize with arbitrary values
void initNN() {
    for (int i = 0; i < FFT_SIZE; i++) {
        for (int j = 0; j < 8; j++) {
            nn_weights1[i][j] = (float)random(-100, 100) / 100.0;
        }
    }

    for (int j = 0; j < 8; j++) {
        nn_weights2[j] = (float)random(-100, 100) / 100.0;
        nn_bias1[j] = 0.1;
    }

    nn_bias2 = 0.1;
}

// ReLU activation
float relu(float x) {
    return x > 0 ? x : 0;
}

// Sigmoid output
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward pass
float runNN(float *input) {
    float hidden[8];

    // Hidden layer
    for (int j = 0; j < 8; j++) {
        float sum = 0;
        for (int i = 0; i < FFT_SIZE; i++) {
            sum += input[i] * nn_weights1[i][j];
        }
        sum += nn_bias1[j];
        hidden[j] = relu(sum);
    }

    // Output layer
    float output = 0;
    for (int j = 0; j < 8; j++) {
        output += hidden[j] * nn_weights2[j];
    }
    output += nn_bias2;

    return sigmoid(output);
}

// ===== SETUP =====
void setup() {
    Serial.begin(115200);
    analogReadResolution(12); // ESP32 ADC = 0–4095

    initNN();
}

// ===== LOOP =====
void loop() {

    // Sample collection
    for (int i = 0; i < FFT_SIZE; i++) {
        int raw = analogRead(ADC_PIN);

        float centered = raw - 2048; // remove DC bias
        float smooth = smoothSample(centered);

        samples[i] = smooth;

        delayMicroseconds(1000000 / SAMPLE_RATE);
    }

    // FFT
    computeFFT();

    float magnitudes[FFT_SIZE];
    getMagnitudes(magnitudes);

    // Neural Network
    confidence = runNN(magnitudes);

    int roundedValue = (int)(confidence + 0.5);

    // DISPLAY AND UI 
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
      } 
    else {
        neopixelWrite(LED_PIN, 0, 100, 0);  
      }
    
    // ===== SERIAL OUTPUT =====
    // Format: FFT values comma-separated
    for (int i = 0; i < FFT_SIZE; i++) {
        Serial.print(magnitudes[i]);
        if (i < FFT_SIZE - 1) Serial.print(",");
    }

    Serial.print(" | confidence:");
    Serial.println(confidence);
}
