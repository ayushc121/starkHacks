import numpy as np
import sounddevice as sd
import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

import torch
import torch.nn as nn

# --- Configuration ---
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
FS = 48000
N = 2048
DEVICE_INDEX = 9
FREQ_MIN, FREQ_MAX = 300, 15000
NUM_BINS = 16
FEATURE_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
                             1.0, 1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 1.0])

HISTORY_SECONDS = 10
UPDATES_PER_SEC = 20
HISTORY_LEN = UPDATES_PER_SEC * HISTORY_SECONDS

# --- Algorithm Configuration ---
WINDOW_SEC    = 5
WINDOW_FRAMES = int(WINDOW_SEC * UPDATES_PER_SEC)
DRONE_BINS    = [10, 11]
OTHER_BINS    = [i for i in range(NUM_BINS) if i not in DRONE_BINS]
LAST_SEND_TIME = time.time()

# Time weights — smooths classifier per-frame probabilities
time_weights = np.linspace(0.5, 1.0, WINDOW_FRAMES)
time_weights = time_weights / np.sum(time_weights)

# Data buffer
fft_history = np.ones((HISTORY_LEN, NUM_BINS)) * 0.1


# ── DroneNet model definition (must match train_model.py) ─────────────────────
class DroneNet(nn.Module):
    """
    4-layer MLP with Batch Normalization and Dropout.
      Input:  33 features
      Output: 1 logit (sigmoid → drone probability)
    """
    def __init__(self, input_dim: int = 33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# --- Load classifier ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'drone_model.pth')
classifier = None
norm_mean  = None
norm_std   = None

if os.path.exists(MODEL_PATH):
    try:
        checkpoint  = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        input_dim   = checkpoint.get('input_dim', 33)
        classifier  = DroneNet(input_dim=input_dim)
        classifier.load_state_dict(checkpoint['model_state'])
        classifier.eval()   # inference mode — disables dropout
        norm_mean = checkpoint['mean'].astype(np.float32)
        norm_std  = checkpoint['std'].astype(np.float32)
        print(f"Loaded DroneNet from {MODEL_PATH}")
    except Exception as e:
        print(f"Warning: could not load model ({e}). Falling back to ratio method.")
else:
    print(f"Warning: drone_model.pth not found. Falling back to ratio method.")
    print(f"Run: python train_model.py --drone drone_data/ --background background_data/")


# --- Serial Setup ---
# try:
#     ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
#     print(f"Connected to {SERIAL_PORT}")
# except Exception as e:
#     print(f"Serial Error: {e}. Running in Plot-Only mode.")
ser = None


# --- Feature extraction (must match train_model.py exactly) ---
def frame_to_features(frame: np.ndarray) -> np.ndarray:
    """
    Converts one frame of 16 bins into the 33-feature vector
    that the classifier was trained on.
    """
    arr         = np.array(frame, dtype=np.float32)
    # Apply high-frequency focus weights
    arr         = arr * FEATURE_WEIGHTS

    diff        = np.diff(arr).tolist()
    high_energy = float(np.mean(arr[4:]))
    spread      = float(np.std(arr))
    return np.array(arr.tolist() + diff + [high_energy, spread], dtype=np.float32)


# --- Confidence calculation ---
def calculate_confidence(history_window: np.ndarray) -> float:
    """
    If a trained DroneNet is available: runs each frame through the MLP
    and returns a time-weighted average of per-frame drone probabilities (0–100%).

    Falls back to the original ratio method if no model is loaded.
    """
    if classifier is not None:
        # Build feature matrix: (WINDOW_FRAMES, 33)
        features = np.array([frame_to_features(history_window[i])
                              for i in range(WINDOW_FRAMES)])

        # Normalize using training-set statistics
        features = (features - norm_mean) / norm_std

        tensor = torch.from_numpy(features)   # shape (W, 33)

        with torch.no_grad():
            logits = classifier(tensor)           # shape (W,)
            probs  = torch.sigmoid(logits).numpy()

        final = float(np.sum(probs * time_weights) * 100)

        # Soft-cutoff: suppress uncertain predictions
        if final < 77.0:
            final = final / 2.0

        return min(100.0, max(0.0, final))

    # Fallback: original ratio method (kept as safety net)
    SNR_MIN = 1.05
    SNR_MAX = 1.30
    confidences = np.zeros(WINDOW_FRAMES)

    for i in range(WINDOW_FRAMES):
        frame      = history_window[i]
        target_mag = np.max([frame[DRONE_BINS[0]], frame[DRONE_BINS[1]]])

        if target_mag < 15.0:
            confidences[i] = 0.0
            continue

        noise_mag = np.max(frame[OTHER_BINS])
        ratio     = target_mag / (noise_mag + 0.1)

        if ratio >= SNR_MAX:
            confidences[i] = 100.0
        elif ratio <= SNR_MIN:
            confidences[i] = 0.0
        else:
            confidences[i] = ((ratio - SNR_MIN) / (SNR_MAX - SNR_MIN)) * 100.0

    return float(np.sum(confidences * time_weights))


# --- Audio callback ---
def callback(indata, frames, time_info, status):
    global fft_history, LAST_SEND_TIME

    audio    = indata[:, 0] * np.hanning(len(indata))
    fft_data = np.abs(np.fft.rfft(audio))

    idx_min     = int(FREQ_MIN * N / FS)
    idx_max     = int(FREQ_MAX * N / FS)
    focused_fft = fft_data[idx_min:idx_max]

    if len(focused_fft) < NUM_BINS:
        return

    bins     = np.array_split(focused_fft, NUM_BINS)
    features = [float(np.log1p(np.mean(b)) * 100) for b in bins]

    fft_history = np.roll(fft_history, -1, axis=0)
    fft_history[-1, :] = features

    # Transmit at 1 Hz
    current_time = time.time()
    if current_time - LAST_SEND_TIME >= 1.0:
        LAST_SEND_TIME = current_time

        recent_window = fft_history[-WINDOW_FRAMES:]
        confidence    = calculate_confidence(recent_window)

        msg = f">{confidence:.2f}\n"
        if ser:
            ser.write(msg.encode())

        print(f"Drone Confidence: {confidence:.2f}%  "
              f"{'[DRONENET]' if classifier else '[RATIO]'}")


# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(fft_history.T, aspect='auto', origin='lower',
               cmap='magma', extent=[-HISTORY_SECONDS, 0, 0, NUM_BINS],
               vmin=0, vmax=500)

ax.set_title("VantagePoint — Live Spectrogram"
             + (" (DroneNet MLP active)" if classifier else " (ratio fallback)"))
ax.set_xlabel("Seconds Ago")
ax.set_ylabel("Frequency Bin (10 & 11 are Target)")
plt.colorbar(im, label="Log Magnitude")

ax.axhline(y=10, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=12, color='cyan', linestyle='--', linewidth=1, alpha=0.5)


def update_plot(frame):
    im.set_array(fft_history.T)
    current_max = np.max(fft_history)
    if current_max > 10:
        im.set_clim(vmin=0, vmax=current_max)
    return [im]


# --- Start ---
stream = sd.InputStream(device=DEVICE_INDEX, callback=callback,
                        samplerate=FS, blocksize=N, channels=1)

with stream:
    ani = FuncAnimation(fig, update_plot, interval=1000 / UPDATES_PER_SEC,
                        blit=True, cache_frame_data=False)
    plt.show()