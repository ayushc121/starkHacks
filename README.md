# Omen
### Acoustic Drone Detection System — StarkHacks 2026

Omen is a real-time drone detection system that uses a repurposed speaker as a microphone to identify drones by their acoustic signature. A trained neural network runs entirely on an ESP32-S3 microcontroller with no internet connection, no camera, and no moving parts.

---

## How It Works

A standard 8Ω 2W speaker is wired as a microphone and amplified through an LM358 op-amp. The ESP32-S3 samples the audio, runs a 16-point DFT on the signal, and feeds the resulting frequency bins into a small neural network (DroneNet) baked directly into the firmware as float arrays. The network outputs a confidence score between 0 and 1. If confidence exceeds the detection threshold, an LED fires and a 4-digit display shows the score.

In parallel, a Python script on a connected laptop receives the FFT data over serial and renders a live spectrogram so you can see the frequency activity in real time.

```
Speaker-mic → LM358 → ESP32-S3 ADC → 16-pt DFT → DroneNet → Confidence Score
                                                                      ↓
                                                          LED + TM1637 Display
                                                                      ↓
                                                          Serial → plot.py → Spectrogram
```

---

## Repository Structure

```
starkHacks/
├── main.cpp          # ESP32-S3 firmware — DFT, DroneNet inference, display, LED
└── plot.py           # Python live spectrogram — reads serial output from ESP32
```

### `main.cpp` — ESP32 Firmware

| Component | Detail |
|---|---|
| Microcontroller | ESP32-S3 |
| ADC pin | GPIO 36 |
| Sample rate | 4000 Hz |
| FFT size | 16 points (DFT) |
| Smoothing | 4-frame moving average |
| Display | TM1637 4-digit (CLK=42, DIO=41) |
| Indicator LED | GPIO 48 |
| Detection threshold | 0.7 confidence |

The neural network weights (`W0`, `W4`, `bn1_*` arrays) are compiled directly into the firmware — no SD card or filesystem needed. The network uses batch normalization and two linear layers to classify each FFT frame as drone or background.

### `plot.py` — Python Visualizer

Reads serial output from the ESP32 (`COM4`, 115200 baud) and renders a scrolling 10-second spectrogram using matplotlib. Expects packets in the format:

```
mag0,mag1,...mag15 | confidence:0.85
```

Target frequency bins (10 and 11) are highlighted with cyan dashed lines on the spectrogram.

---

## Hardware

| Part | Purpose |
|---|---|
| ESP32-S3 dev board | Main compute, ADC, serial |
| 8Ω 2W speaker | Repurposed as microphone |
| LM358 op-amp | Signal amplification |
| TM1637 display | Shows confidence % |
| LED | Visual detection alert |

**Speaker-as-mic note:** The speaker has a natural resonant frequency that boosts certain bands and attenuates others, particularly above 3 kHz. The DroneNet weights were trained to account for this non-flat frequency response.

---

## Setup

### ESP32 Firmware

1. Install [PlatformIO](https://platformio.org/) or Arduino IDE with ESP32-S3 board support
2. Install the `TM1637` library
3. Flash `main.cpp` to your ESP32-S3
4. Wire the speaker-mic to GPIO 36 through the LM358 amplifier circuit
5. Connect the TM1637 display to GPIO 42 (CLK) and GPIO 41 (DIO)
6. Connect indicator LED to GPIO 48

### Python Visualizer

```bash
pip install numpy pyserial matplotlib
python plot.py
```

Change `SERIAL_PORT = 'COM4'` in `plot.py` to match your system (`/dev/ttyUSB0` on Linux/Mac).

---

## The ML Pipeline

The DroneNet model was trained offline in Python using scikit-learn and PyTorch, then the weights were exported and baked into the firmware as C float arrays. Training used:

- ~80 seconds of DJI Mini 3 drone audio
- ~50 seconds of ambient room noise
- Silent segments auto-extracted from mixed clips

Feature extraction mirrors the firmware's DFT pipeline exactly so the model sees the same numbers during inference that it saw during training.

A separate `optimizer.py` script (simulated annealing, 3000 iterations) can re-tune detection thresholds for different acoustic environments without retraining the full model.

---

## Detection Threshold

The confidence threshold is set at **0.7** (`DETECTION_THRESHOLD` in `main.cpp`). Lowering it increases sensitivity but raises false positives. The value was selected by running the optimizer against labeled audio and checking the resulting confusion matrix (precision/recall tradeoff).

---

## Built At

StarkHacks 2026
