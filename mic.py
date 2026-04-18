import numpy as np
import sounddevice as sd
import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Configuration ---
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200
FS = 48000
N = 2048
DEVICE_INDEX = 9
FREQ_MIN, FREQ_MAX = 300, 15000
NUM_BINS = 16

HISTORY_SECONDS = 10
UPDATES_PER_SEC = 20 # Lowered slightly for better stability
HISTORY_LEN = UPDATES_PER_SEC * HISTORY_SECONDS

# Data Buffer - Initialized with a tiny bit of noise to prevent scale errors
fft_history = np.ones((HISTORY_LEN, NUM_BINS)) * 0.1

# --- Serial Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Serial Error: {e}. Running in Plot-Only mode.")
    ser = None

# --- Audio Logic ---
def callback(indata, frames, time_info, status):
    global fft_history
    if status:
        print(status)
    
    # Process Audio
    audio = indata[:, 0] * np.hanning(len(indata))
    fft_data = np.abs(np.fft.rfft(audio))
    
    # Slice and Bin
    idx_min = int(FREQ_MIN * N / FS)
    idx_max = int(FREQ_MAX * N / FS)
    focused_fft = fft_data[idx_min:idx_max]
    
    if len(focused_fft) < NUM_BINS: return
        
    bins = np.array_split(focused_fft, NUM_BINS)
    # Log scaling for better dynamic range
    features = [float(np.log1p(np.mean(b)) * 100) for b in bins]

    # Shift history and add new data row
    fft_history = np.roll(fft_history, -1, axis=0)
    fft_history[-1, :] = features

    # Send to ESP32
    if ser:
        msg = ">" + ",".join(map(str, [int(f) for f in features])) + "\n"
        ser.write(msg.encode())




# --- Plotting Logic ---
fig, ax = plt.subplots(figsize=(12, 6))
# Using 'magma' colormap. vmin/vmax helps fix the "one color" issue
im = ax.imshow(fft_history.T, aspect='auto', origin='lower', 
               cmap='magma', extent=[-HISTORY_SECONDS, 0, 0, NUM_BINS],
               vmin=0, vmax=500) # Pre-set range, will adjust below

ax.set_title("Live Audio Spectrogram (300Hz - 15kHz)")
ax.set_xlabel("Seconds Ago")
ax.set_ylabel("Frequency Bin")
plt.colorbar(im, label="Log Magnitude")

def update_plot(frame):
    # Update the image data
    im.set_array(fft_history.T)
    
    # Auto-adjust brightness (vmax) based on the loudest sound in history
    # This prevents the "one color" problem
    current_max = np.max(fft_history)
    if current_max > 10:
        im.set_clim(vmin=0, vmax=current_max)
        
    return [im]

# --- Start Threads ---
stream = sd.InputStream(device=DEVICE_INDEX, callback=callback, 
                        samplerate=FS, blocksize=N, channels=1)

with stream:
    # cache_frame_data=False removes the warning you saw
    ani = FuncAnimation(fig, update_plot, interval=1000/UPDATES_PER_SEC, 
                        blit=True, cache_frame_data=False)
    plt.show()