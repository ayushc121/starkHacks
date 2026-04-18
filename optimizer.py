"""
VantagePoint Optimizer
======================
Automatically tunes SNR thresholds, silence gate, and time-weight
for your specific hardware by running simulated annealing against
a labeled drone audio clip.

Run once per machine / hardware setup. Takes ~5-10 seconds.

Usage:
    python optimizer.py --audio path/to/drone_clip.wav
    python optimizer.py --audio path/to/drone_clip.mp3

The clip must contain BOTH drone sound AND quiet/silence sections.
The script auto-segments by energy level - no manual labeling needed.

Outputs exact values to paste into mic.py.
"""

import numpy as np
import argparse
import sys
import math
import subprocess
import tempfile
from pathlib import Path

# ── Must match mic.py exactly ─────────────────────────────────────────────────

FS              = 48000
N               = 2048
FREQ_MIN        = 300
FREQ_MAX        = 15000
NUM_BINS        = 16
WINDOW_SEC      = 5
UPDATES_PER_SEC = 20
WINDOW_FRAMES   = int(WINDOW_SEC * UPDATES_PER_SEC)

DRONE_BINS = [10, 11]
OTHER_BINS = [i for i in range(NUM_BINS) if i not in DRONE_BINS]

# ── Optimizer settings ────────────────────────────────────────────────────────

SA_ITERATIONS = 3000   # fast enough for real-time use, thorough enough to converge


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(path: str) -> np.ndarray:
    """Load audio to mono float32 at FS. Handles mp3 via pydub (no ffmpeg needed)."""
    path = Path(path)
    if path.suffix.lower() in ('.mp3', '.m4a', '.ogg', '.flac', '.aac'):
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(path))
        audio = audio.set_frame_rate(FS).set_channels(1)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        data = samples / (2**15)  # normalize 16-bit to -1.0..1.0
        return data

    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
    except Exception as e:
        print(f"ERROR: Could not read audio file: {e}")
        sys.exit(1)

    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.float32:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    if sr != FS:
        from scipy.signal import resample
        data = resample(data, int(len(data) * FS / sr)).astype(np.float32)

    return data


# ── Feature extraction (exact replica of mic.py callback logic) ───────────────

def extract_bin_features(data: np.ndarray) -> np.ndarray:
    """
    Runs the same FFT + log-bin pipeline as mic.py's audio callback.
    Returns shape (num_frames, NUM_BINS).
    """
    idx_min = int(FREQ_MIN * N / FS)
    idx_max = int(FREQ_MAX * N / FS)
    frames = []

    for start in range(0, len(data) - N, N):
        chunk    = data[start:start + N]
        windowed = chunk * np.hanning(N)
        fft_data = np.abs(np.fft.rfft(windowed))
        focused  = fft_data[idx_min:idx_max]
        if len(focused) < NUM_BINS:
            continue
        bins     = np.array_split(focused, NUM_BINS)
        features = [float(np.log1p(np.mean(b)) * 100) for b in bins]
        frames.append(features)

    return np.array(frames)


def subsample_to_update_rate(frames: np.ndarray) -> np.ndarray:
    """
    mic.py stores one frame per callback at blocksize=N.
    Real rate is FS/N (~23.4 Hz). Subsample to UPDATES_PER_SEC (20 Hz)
    to match fft_history frame rate exactly.
    """
    real_fps = FS / N
    step     = max(1, round(real_fps / UPDATES_PER_SEC))
    return frames[::step]


# ── Segmentation ──────────────────────────────────────────────────────────────

def build_windows(indices, all_frames, window_size):
    """Collect WINDOW_FRAMES-length clips from runs of consecutive indices."""
    if len(indices) == 0:
        return []
    windows = []
    runs = []; run = [indices[0]]
    for i in indices[1:]:
        if i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run); run = [i]
    runs.append(run)
    for run in runs:
        if len(run) < window_size:
            continue
        for start in range(0, len(run) - window_size + 1, window_size // 2):
            w = all_frames[run[start:start + window_size]]
            if len(w) == window_size:
                windows.append(w)
    return windows


def segment_audio(frames: np.ndarray, energy_percentile: int = 35):
    """Split frames into drone-active and silent pools by peak bin energy."""
    frame_energy  = np.max(frames, axis=1)
    threshold     = np.percentile(frame_energy, energy_percentile)
    drone_idx     = np.where(frame_energy >  threshold)[0]
    silent_idx    = np.where(frame_energy <= threshold)[0]
    drone_windows  = build_windows(drone_idx,  frames, WINDOW_FRAMES)
    silent_windows = build_windows(silent_idx, frames, WINDOW_FRAMES)
    return drone_windows, silent_windows


# ── Confidence calculation ────────────────────────────────────────────────────

def calculate_confidence(window: np.ndarray, params: dict) -> float:
    """
    Mirrors mic.py's calculate_confidence() with two important fixes:
    1. Uses MEAN noise (not MAX) — MAX was always dominated by loud low-freq
       bins, making the ratio always < 1 and confidence always 0.
    2. Parameterizes silence_gate and weight_start so they can be optimized.
    """
    snr_min      = params['snr_min']
    snr_max      = params['snr_max']
    silence_gate = params['silence_gate']
    weight_start = params['weight_start']

    time_weights = np.linspace(weight_start, 1.0, WINDOW_FRAMES)
    time_weights = time_weights / np.sum(time_weights)

    confidences = np.zeros(WINDOW_FRAMES)

    for i in range(WINDOW_FRAMES):
        frame      = window[i]
        target_mag = max(frame[DRONE_BINS[0]], frame[DRONE_BINS[1]])

        if target_mag < silence_gate:
            continue

        noise_mag = float(np.mean([frame[j] for j in OTHER_BINS]))
        ratio     = target_mag / (noise_mag + 0.1)

        if ratio >= snr_max:
            confidences[i] = 100.0
        elif ratio > snr_min:
            confidences[i] = ((ratio - snr_min) / (snr_max - snr_min)) * 100.0

    return float(np.sum(confidences * time_weights))


# ── Fitness & optimizer ───────────────────────────────────────────────────────

def score_params(params, drone_windows, silent_windows):
    """Returns (fitness, avg_drone%, avg_silent%). Maximize fitness."""
    drone_scores  = [calculate_confidence(w, params) for w in drone_windows]
    silent_scores = [calculate_confidence(w, params) for w in silent_windows]
    avg_drone     = float(np.mean(drone_scores))
    avg_silent    = float(np.mean(silent_scores))
    return avg_drone - avg_silent, avg_drone, avg_silent


def clamp_params(p):
    p['snr_min']      = max(0.05, min(2.0,  p['snr_min']))
    p['snr_max']      = max(p['snr_min'] + 0.03, min(3.0, p['snr_max']))
    p['silence_gate'] = max(0.5,  min(30.0, p['silence_gate']))
    p['weight_start'] = max(0.01, min(0.99, p['weight_start']))
    return p


def run_optimizer(audio_path: str):
    print("\n" + "=" * 60)
    print("  VantagePoint — Hardware Optimizer")
    print("=" * 60)

    print(f"\n[1/4] Loading audio: {audio_path}")
    data = load_audio(audio_path)
    print(f"      Duration: {len(data)/FS:.1f}s")

    print("\n[2/4] Extracting FFT features (replicating mic.py logic)...")
    raw_frames = extract_bin_features(data)
    frames     = subsample_to_update_rate(raw_frames)
    print(f"      {len(frames)} frames at ~{UPDATES_PER_SEC} fps")

    print("\n[3/4] Segmenting drone vs. silence...")
    drone_windows, silent_windows = segment_audio(frames)
    print(f"      Drone windows : {len(drone_windows)}")
    print(f"      Silent windows: {len(silent_windows)}")

    if len(drone_windows) == 0:
        print("\n  ERROR: No drone segments found. Clip needs active drone audio.")
        sys.exit(1)
    if len(silent_windows) == 0:
        print("\n  ERROR: No silent segments found. Clip needs some quiet sections.")
        sys.exit(1)

    print(f"\n[4/4] Running simulated annealing ({SA_ITERATIONS} iterations)...")

    params = {
        'snr_min':      0.3,
        'snr_max':      0.7,
        'silence_gate': 4.0,
        'weight_start': 0.4,
    }

    cur_fit, _, _ = score_params(params, drone_windows, silent_windows)
    best_params   = params.copy()
    best_fit, best_drone, best_silent = cur_fit, 0.0, 100.0

    print(f"\n  {'Iter':>6}  {'Fitness':>8}  {'Drone%':>8}  {'Silent%':>8}")
    print(f"  {'----':>6}  {'-------':>8}  {'------':>8}  {'-------':>8}")

    for it in range(SA_ITERATIONS):
        temp = 1.0 * (0.01 / 1.0) ** (it / SA_ITERATIONS)

        new_p = params.copy()
        key   = np.random.choice(list(params.keys()))
        new_p[key] += 0.5 * temp * np.random.randn()
        new_p = clamp_params(new_p)

        nf, nd, ns = score_params(new_p, drone_windows, silent_windows)
        delta = nf - cur_fit

        if delta > 0 or np.random.rand() < math.exp(delta / (temp + 1e-9)):
            params  = new_p
            cur_fit = nf

        if nf > best_fit:
            best_fit, best_drone, best_silent = nf, nd, ns
            best_params = new_p.copy()

        if (it + 1) % 600 == 0:
            print(f"  {it+1:>6}  {best_fit:>+8.1f}  {best_drone:>7.1f}%  {best_silent:>7.1f}%")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    drone_scores  = [calculate_confidence(w, best_params) for w in drone_windows]
    silent_scores = [calculate_confidence(w, best_params) for w in silent_windows]

    tp = sum(1 for s in drone_scores  if s >= 50)
    fn = sum(1 for s in drone_scores  if s <  50)
    tn = sum(1 for s in silent_scores if s <  50)
    fp = sum(1 for s in silent_scores if s >= 50)

    total     = tp + fn + tn + fp
    precision = tp / (tp + fp + 1e-9) * 100
    recall    = tp / (tp + fn + 1e-9) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / (total + 1e-9) * 100

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n  Drone confidence  (target ~90-100%): {best_drone:.1f}%")
    print(f"  Silent confidence (target ~0%):      {best_silent:.1f}%")
    print(f"\n  Confusion matrix (threshold = 50%):")
    print(f"    TP (drone detected):  {tp:>4}")
    print(f"    FN (drone missed):    {fn:>4}")
    print(f"    TN (silence correct): {tn:>4}")
    print(f"    FP (false alarm):     {fp:>4}")
    print(f"\n    Accuracy:  {accuracy:.1f}%")
    print(f"    Precision: {precision:.1f}%")
    print(f"    Recall:    {recall:.1f}%")
    print(f"    F1 score:  {f1:.1f}%")

    print("\n" + "=" * 60)
    print("  PASTE INTO mic.py")
    print("=" * 60)
    print(f"""
# --- Optimized values (auto-generated by optimizer.py) ---

SNR_MIN = {best_params['snr_min']:.4f}
SNR_MAX = {best_params['snr_max']:.4f}

# In calculate_confidence(), make THREE changes:

# 1. Silence gate — replace the existing value:
if target_mag < {best_params['silence_gate']:.2f}:

# 2. noise_mag line — change MAX to MEAN:
#    OLD: noise_mag = np.max(frame[OTHER_BINS])
#    NEW:
noise_mag = float(np.mean([frame[j] for j in OTHER_BINS]))

# 3. time_weights line — replace weight_start value:
time_weights = np.linspace({best_params['weight_start']:.4f}, 1.0, WINDOW_FRAMES)
""")
    print("=" * 60 + "\n")

    return best_params


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VantagePoint — auto-tunes mic.py thresholds for your hardware'
    )
    parser.add_argument(
        '--audio', required=True,
        help='Drone audio clip (.wav or .mp3) with both drone sound and silence'
    )
    args = parser.parse_args()
    run_optimizer(args.audio)
