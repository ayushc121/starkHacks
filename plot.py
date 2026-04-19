import numpy as np
import serial
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- Configuration ---
SERIAL_PORT = 'COM4' 
BAUD_RATE = 115200
NUM_BINS = 16
HISTORY_SECONDS = 10
UPDATES_PER_SEC = 20 
HISTORY_LEN = UPDATES_PER_SEC * HISTORY_SECONDS

# Data Buffer
fft_history = np.ones((HISTORY_LEN, NUM_BINS)) * 0.1

# --- Serial Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Serial Error: {e}")
    exit()

def parse_serial_data():
    """
    Parses Arduino format: 
    mag0,mag1,...mag15 | confidence:0.85
    """
    global fft_history
    if ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').strip()
            if not line or "|" not in line:
                return

            # Split data and confidence
            data_part, conf_part = line.split("|")
            
            # Parse Bins
            features = [float(x) for x in data_part.split(",")]
            
            # Parse Confidence (for terminal print)
            conf_val = float(conf_part.split(":")[1])
            
            if len(features) == NUM_BINS:
                # Log scale for better visualization on spectrogram
                features_log = [float(np.log1p(f) * 10) for f in features]
                
                # Update History
                fft_history = np.roll(fft_history, -1, axis=0)
                fft_history[-1, :] = features_log
                
                # Optional: Print confidence to terminal
                print(f"Drone Confidence: {conf_val*100:.2f}%", end='\r')
                
        except Exception as e:
            pass # Ignore malformed packets

# --- Plotting Logic (Exactly the same) ---
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(fft_history.T, aspect='auto', origin='lower', 
               cmap='magma', extent=[-HISTORY_SECONDS, 0, 0, NUM_BINS],
               vmin=0, vmax=50) # Adjusted vmax for normalized serial data

ax.set_title("Live Spectrogram (Serial Input)")
ax.set_xlabel("Seconds Ago")
ax.set_ylabel("Frequency Bin")
plt.colorbar(im, label="Log Magnitude")

# Target bin highlights
ax.axhline(y=10, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=11, color='cyan', linestyle='--', linewidth=1, alpha=0.5)

def update_plot(frame):
    # Read serial data before updating plot
    parse_serial_data()
    
    im.set_array(fft_history.T)
    current_max = np.max(fft_history)
    if current_max > 5:
        im.set_clim(vmin=0, vmax=current_max)
    return [im]

# --- Start Animation ---
# Note: interval is lower to keep serial buffer clear
ani = FuncAnimation(fig, update_plot, interval=1000/UPDATES_PER_SEC, 
                    blit=True, cache_frame_data=False)

plt.show()

if ser:
    ser.close()
