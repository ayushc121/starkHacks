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
DEVICE_INDEX = 5
FREQ_MIN, FREQ_MAX = 300, 15000
NUM_BINS = 16

HISTORY_SECONDS = 10
UPDATES_PER_SEC = 20 
HISTORY_LEN = UPDATES_PER_SEC * HISTORY_SECONDS

# --- Algorithm Configuration ---
WINDOW_SEC = 5
WINDOW_FRAMES = int(WINDOW_SEC * UPDATES_PER_SEC)
DRONE_BINS = [10, 11]       # ZERO INDEXED
OTHER_BINS = [i for i in range(NUM_BINS) if i not in DRONE_BINS]
LAST_SEND_TIME = time.time()

# NEW: Tuning variables for sensitivity
SNR_MIN = 0.2364  # Ratio where confidence starts climbing above 0%
SNR_MAX = 0.2664  # Ratio where confidence hits 100% (Lower = more sensitive)



# Generate weights for the 5-second window (Linear ramp: recent = higher weight)
time_weights = np.linspace(0.0100, 1.0, WINDOW_FRAMES)  # Run optimizer.py to auto-tune this start value per hardware
time_weights = time_weights / np.sum(time_weights) # Normalize to sum to 1.0

# Data Buffer
fft_history = np.ones((HISTORY_LEN, NUM_BINS)) * 0.1

# --- Serial Setup ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    print(f"Connected to {SERIAL_PORT}")
except Exception as e:
    print(f"Serial Error: {e}. Running in Plot-Only mode.")
    ser = None

def calculate_confidence(history_window):
    """Calculates a 0-100% confidence score based on the 5-second window.

    Run optimizer.py to auto-tune SNR_MIN, SNR_MAX, silence gate, and weight_start
    for your specific hardware (speaker-mics have different frequency response curves):
        python optimizer.py --audio your_drone_clip.mp3
    The optimizer uses simulated annealing over 3000 iterations, auto-segments the
    clip into drone vs. silence windows, and outputs exact values to paste back here.
    """
    confidences = np.zeros(WINDOW_FRAMES)

    for i in range(WINDOW_FRAMES):
        frame = history_window[i]

        # 1. Target Magnitude (Try taking the MAX instead of the average to be more sensitive)
        target_mag = np.max([frame[DRONE_BINS[0]], frame[DRONE_BINS[1]]])

        # 2. Mean Noise Magnitude (mean avoids loud low-freq bins dominating the ratio)
        noise_mag = float(np.mean([frame[j] for j in OTHER_BINS]))

        # 3. Absolute Silence Check
        if target_mag < 9.13:
            confidences[i] = 0.0
            continue
            
        # 4. Ratio Calculation
        ratio = target_mag / (noise_mag + 0.1)
        
        # 5. Map Ratio to Percentage using the new variables
        if ratio >= SNR_MAX:       
            conf = 100.0
        elif ratio <= SNR_MIN:     
            conf = 0.0
        else:                  
            # Dynamic interpolation based on your variables
            conf = ((ratio - SNR_MIN) / (SNR_MAX - SNR_MIN)) * 100.0 
            
        confidences[i] = conf

    # 6. Apply Time Weights (Weighted Average)
    final_confidence = np.sum(confidences * time_weights)
    return final_confidence


# --- Audio Logic ---
def callback(indata, frames, time_info, status):
    global fft_history, LAST_SEND_TIME

    audio = indata[:, 0] * np.hanning(len(indata))
    fft_data = np.abs(np.fft.rfft(audio))
    
    idx_min = int(FREQ_MIN * N / FS)
    idx_max = int(FREQ_MAX * N / FS)
    focused_fft = fft_data[idx_min:idx_max]
    
    if len(focused_fft) < NUM_BINS: return
        
    bins = np.array_split(focused_fft, NUM_BINS)
    features = [float(np.log1p(np.mean(b)) * 100) for b in bins]

    # Shift history and add new data
    fft_history = np.roll(fft_history, -1, axis=0)
    fft_history[-1, :] = features

    # --- TRANSMISSION LOGIC (1 Hz) ---
    current_time = time.time()
    if current_time - LAST_SEND_TIME >= 1.0:
        LAST_SEND_TIME = current_time
        
        # Grab only the last 5 seconds of data for the algorithm
        recent_window = fft_history[-WINDOW_FRAMES:]
        confidence = calculate_confidence(recent_window)
        
        # Format explicitly to 2 decimal places with a > header
        msg = f">{confidence:.2f}\n"
        
        if ser:
            ser.write(msg.encode())
            
        # Print to terminal alongside the plot so you can verify it's working
        print(f"Drone Confidence: {confidence:.2f}%")

# --- Plotting Logic ---
fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(fft_history.T, aspect='auto', origin='lower', 
               cmap='magma', extent=[-HISTORY_SECONDS, 0, 0, NUM_BINS],
               vmin=0, vmax=500)

ax.set_title("Live Spectrogram & Background Detection")
ax.set_xlabel("Seconds Ago")
ax.set_ylabel("Frequency Bin (10 & 11 are Target)")
plt.colorbar(im, label="Log Magnitude")

# Draw horizontal lines to visually highlight the target bins on the plot
ax.axhline(y=10, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=12, color='cyan', linestyle='--', linewidth=1, alpha=0.5)

def update_plot(frame):
    im.set_array(fft_history.T)
    current_max = np.max(fft_history)
    if current_max > 10:
        im.set_clim(vmin=0, vmax=current_max)
    return [im]

# --- Start Threads ---
stream = sd.InputStream(device=DEVICE_INDEX, callback=callback, 
                        samplerate=FS, blocksize=N, channels=1)

with stream:
    ani = FuncAnimation(fig, update_plot, interval=1000/UPDATES_PER_SEC, 
                        blit=True, cache_frame_data=False)
    plt.show()
