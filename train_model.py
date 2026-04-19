"""
VantagePoint — Model Trainer
=============================
Trains a Random Forest classifier on labeled audio and saves
drone_model.pkl for use by mic.py.

Usage:
    python train_model.py --drone drone.wav --background background.wav

You can pass multiple files per class:
    python train_model.py --drone d1.wav d2.wav --background bg1.wav bg2.wav

Supports .wav only. Convert .mp3 first with:
    ffmpeg -i clip.mp3 -ar 48000 -ac 1 clip.wav
"""

import numpy as np
import argparse
import sys
import pickle
import os
from pathlib import Path

# ── Must match mic.py exactly ─────────────────────────────────────────────────
FS       = 48000
N        = 2048
FREQ_MIN = 300
FREQ_MAX = 15000
NUM_BINS = 16
FEATURE_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 1.0])


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(data: np.ndarray) -> np.ndarray:
    """
    Extracts 33 features per frame:
      - 16 log-magnitude FFT bins  (same as mic.py)
      - 15 bin-to-bin gradients    (captures spectral shape)
      -  1 high-frequency energy   (bins 4-15, captures drone harmonics)
      -  1 spectral spread         (std of bins, drones are spectrally flat)
    """
    idx_min = int(FREQ_MIN * N / FS)
    idx_max = int(FREQ_MAX * N / FS)
    frames  = []

    for start in range(0, len(data) - N, N):
        chunk   = data[start:start + N] * np.hanning(N)
        fft     = np.abs(np.fft.rfft(chunk))
        focused = fft[idx_min:idx_max]
        bins    = np.array_split(focused, NUM_BINS)
        base    = [float(np.log1p(np.mean(b)) * 100) for b in bins]
        
        # Apply high-frequency focus weights
        base    = (np.array(base) * FEATURE_WEIGHTS).tolist()
        arr     = np.array(base)

        diff        = np.diff(arr).tolist()         # 15 gradient features
        high_energy = float(np.mean(arr[4:]))        # energy above low freqs
        spread      = float(np.std(arr))             # spectral spread

        frames.append(base + diff + [high_energy, spread])

    return np.array(frames)


def load_wav(path: str) -> np.ndarray:
    """Load wav to mono float32, auto-detect normalization."""
    from scipy.io import wavfile
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    max_val = np.iinfo(np.int32).max if data.max() > 32768 else np.iinfo(np.int16).max
    return data / max_val


# ── Training ──────────────────────────────────────────────────────────────────

def train(drone_paths, background_paths, output='drone_model.pkl'):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report

    print("\n" + "=" * 55)
    print("  VantagePoint — Model Trainer")
    print("=" * 55)

    # ── Load drone audio ──────────────────────────────────────────
    print("\n[1/4] Loading drone audio...")
    drone_frames = []
    for path in drone_paths:
        data   = load_wav(path)
        frames = extract_features(data)
        if len(frames) > 0:
            drone_frames.append(frames)
            # print(f"      {Path(path).name}: {len(frames)} frames")
        else:
            print(f"      SKIPPING {Path(path).name} (too short)")
    
    if not drone_frames:
        print("ERROR: No valid drone frames extracted.")
        sys.exit(1)
    drone_X = np.vstack(drone_frames)

    # ── Load background audio ──────────────────────────────────────
    print("\n[2/4] Loading background audio...")
    bg_frames = []
    for path in background_paths:
        data   = load_wav(path)
        frames = extract_features(data)
        if len(frames) > 0:
            bg_frames.append(frames)
            # print(f"      {Path(path).name}: {len(frames)} frames")
        else:
            print(f"      SKIPPING {Path(path).name} (too short)")
    
    if not bg_frames:
        print("ERROR: No valid background frames extracted.")
        sys.exit(1)
    bg_X = np.vstack(bg_frames)

    X = np.vstack([drone_X, bg_X])
    y = np.array([1] * len(drone_X) + [0] * len(bg_X))

    print(f"\n      Total: {len(drone_X)} drone frames, {len(bg_X)} background frames")
    print(f"      Feature vector: {X.shape[1]} features per frame")

    # ── Train ──────────────────────────────────────────────────────
    print("\n[3/4] Training Random Forest classifier...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # <--- CRITICAL for the data imbalance
        ))
    ])

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
    print(f"      5-fold cross-validation F1: {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X, y)
    probs = pipe.predict_proba(X)[:, 1]

    drone_mean  = probs[y == 1].mean() * 100
    drone_min   = probs[y == 1].min()  * 100
    bg_mean     = probs[y == 0].mean() * 100
    bg_max      = probs[y == 0].max()  * 100

    print(f"\n      Drone frames:      mean={drone_mean:.1f}%  min={drone_min:.1f}%")
    print(f"      Background frames: mean={bg_mean:.1f}%   max={bg_max:.1f}%")

    # ── Save ───────────────────────────────────────────────────────
    print(f"\n[4/4] Saving model to {output}...")
    with open(output, 'wb') as f:
        pickle.dump(pipe, f)

    size_kb = os.path.getsize(output) / 1024
    print(f"      Saved ({size_kb:.0f} KB)")

    print("\n" + "=" * 55)
    print("  DONE — copy drone_model.pkl next to mic.py")
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VantagePoint drone classifier')
    parser.add_argument('--drone',      nargs='+', required=True, help='Drone audio files or directories')
    parser.add_argument('--background', nargs='+', required=True, help='Background audio files or directories')
    parser.add_argument('--output',     default='drone_model.pkl', help='Output model file')
    args = parser.parse_args()

    def resolve_files(paths):
        resolved = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                found = list(path.rglob("*.wav"))
                resolved.extend([str(f) for f in found])
            elif path.exists():
                resolved.append(str(path))
            else:
                print(f"WARNING: Path not found: {p}")
        return resolved

    drone_files = resolve_files(args.drone)
    bg_files    = resolve_files(args.background)

    if not drone_files:
        print("ERROR: No drone .wav files found.")
        sys.exit(1)
    if not bg_files:
        print("ERROR: No background .wav files found.")
        sys.exit(1)

    print(f"Found {len(drone_files)} drone files and {len(bg_files)} background files.")
    train(drone_files, bg_files, args.output)
