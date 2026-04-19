"""
VantagePoint — Deep Learning Model Trainer
==========================================
Trains a 4-layer MLP (PyTorch) on labeled audio and saves
drone_model.pth for use by mic.py.

Usage:
    python train_model.py --drone drone_data/ --background background_data/

You can pass multiple files or directories per class:
    python train_model.py --drone d1.wav d2.wav --background bg1.wav bg2.wav

Supports .wav only. Convert .mp3 first with:
    ffmpeg -i clip.mp3 -ar 48000 -ac 1 clip.wav
"""

import numpy as np
import argparse
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Must match mic.py exactly ─────────────────────────────────────────────────
FS       = 48000
N        = 2048
FREQ_MIN = 300
FREQ_MAX = 15000
NUM_BINS = 16
FEATURE_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0,
                             1.0, 1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 1.0])


# ── Model Architecture ────────────────────────────────────────────────────────

class DroneNet(nn.Module):
    """
    4-layer MLP with Batch Normalization and Dropout.
      Input:  33 features  (16 freq bins + 15 gradients + 2 summary stats)
      Output: 1 logit      (sigmoid → drone probability)
    """
    def __init__(self, input_dim: int = 33):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: 33 → 64
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 64 → 32
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3: 32 → 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),

            # Output: 16 → 1  (raw logit — sigmoid applied at inference time)
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


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

    return np.array(frames, dtype=np.float32)


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

def train(drone_paths, background_paths, output='drone_model.pth'):
    print("\n" + "=" * 55)
    print("  VantagePoint — Deep Learning Trainer (PyTorch MLP)")
    print("=" * 55)

    # ── Load drone audio ──────────────────────────────────────────
    print("\n[1/5] Loading drone audio...")
    drone_frames = []
    for path in drone_paths:
        data   = load_wav(path)
        frames = extract_features(data)
        if len(frames) > 0:
            drone_frames.append(frames)
        else:
            print(f"      SKIPPING {Path(path).name} (too short)")

    if not drone_frames:
        print("ERROR: No valid drone frames extracted.")
        sys.exit(1)
    drone_X = np.vstack(drone_frames)

    # ── Load background audio ──────────────────────────────────────
    print("\n[2/5] Loading background audio...")
    bg_frames = []
    for path in background_paths:
        data   = load_wav(path)
        frames = extract_features(data)
        if len(frames) > 0:
            bg_frames.append(frames)
        else:
            print(f"      SKIPPING {Path(path).name} (too short)")

    if not bg_frames:
        print("ERROR: No valid background frames extracted.")
        sys.exit(1)
    bg_X = np.vstack(bg_frames)

    X = np.vstack([drone_X, bg_X]).astype(np.float32)
    y = np.array([1.0] * len(drone_X) + [0.0] * len(bg_X), dtype=np.float32)

    print(f"\n      Total: {len(drone_X)} drone frames, {len(bg_X)} background frames")
    print(f"      Feature vector: {X.shape[1]} features per frame")

    # ── Normalize ──────────────────────────────────────────────────
    print("\n[3/5] Normalizing features...")
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8   # avoid div-by-zero
    X_norm = (X - mean) / std

    # ── Build dataset & split 80/20 ───────────────────────────────
    dataset   = TensorDataset(torch.from_numpy(X_norm), torch.from_numpy(y))
    n_val     = max(1, int(0.2 * len(dataset)))
    n_train   = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

    # ── Class-imbalance weight ─────────────────────────────────────
    n_drone = len(drone_X)
    n_bg    = len(bg_X)
    pos_weight = torch.tensor([n_bg / (n_drone + 1e-8)], dtype=torch.float32)
    print(f"      pos_weight (imbalance correction): {pos_weight.item():.3f}")

    # ── Model, loss, optimiser ─────────────────────────────────────
    print("\n[4/5] Training DroneNet MLP...")
    model     = DroneNet(input_dim=X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', patience=5, factor=0.5
    )

    # ── Training loop with early stopping ─────────────────────────
    EPOCHS        = 150
    PATIENCE      = 15       # early stopping patience
    best_val_loss = float('inf')
    patience_ctr  = 0
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
                  + ("  <- best" if patience_ctr == 0 else ""))

        if patience_ctr >= PATIENCE:
            print(f"\n      Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs).")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # ── Training-set confidence stats ──────────────────────────────
    model.eval()
    with torch.no_grad():
        all_logits = model(torch.from_numpy(X_norm))
        all_probs  = torch.sigmoid(all_logits).numpy()

    y_arr       = y.astype(bool)
    drone_mean  = all_probs[y_arr].mean() * 100
    drone_min   = all_probs[y_arr].min()  * 100
    bg_mean     = all_probs[~y_arr].mean() * 100
    bg_max      = all_probs[~y_arr].max()  * 100

    print(f"\n      Drone frames:      mean={drone_mean:.1f}%  min={drone_min:.1f}%")
    print(f"      Background frames: mean={bg_mean:.1f}%   max={bg_max:.1f}%")

    # ── Save ───────────────────────────────────────────────────────
    print(f"\n[5/5] Saving model to {output}...")
    torch.save({
        'model_state': model.state_dict(),
        'mean':        mean,
        'std':         std,
        'input_dim':   X.shape[1],
    }, output)

    size_kb = os.path.getsize(output) / 1024
    print(f"      Saved ({size_kb:.0f} KB)")

    print("\n" + "=" * 55)
    print("  DONE — copy drone_model.pth next to mic.py")
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VantagePoint drone classifier')
    parser.add_argument('--drone',      nargs='+', required=True, help='Drone audio files or directories')
    parser.add_argument('--background', nargs='+', required=True, help='Background audio files or directories')
    parser.add_argument('--output',     default='drone_model.pth', help='Output model file (.pth)')
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
