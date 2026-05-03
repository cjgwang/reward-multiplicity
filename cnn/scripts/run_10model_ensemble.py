"""
10-Model Independent Ensemble
==============================
Trains 10 CNN reward models independently (plain MSE, seeds 0–9).
Saves every model's .pth and produces a 2×5 summary figure where each
panel shows the reward heatmap for star position (5,5) with a shared
colour scale and colorbar.

Usage:
    python scripts/run_10model_ensemble.py
    python scripts/run_10model_ensemble.py --epochs 200
"""
import sys, os, argparse
from datetime import datetime
from collections import defaultdict

CNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CNN_DIR)
RESULTS_DIR = os.path.join(CNN_DIR, "results")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=None)
args = parser.parse_args()

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from gridworld import DeterministicGridWorld
from data.datasets import generate_balanced_dataset, generate_full_enumeration_dataset
from models.cnn import DeepCNN, GridDataset
from training.loops import train_one_epoch, eval_model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ROWS, COLS   = 7, 7
CHANNELS     = 3
STAR_POS     = (5, 5)
N_POS, N_NEG = 2000, 2000
HIDDEN_CHAN   = 4
FC_HIDDEN    = 32
EPOCHS       = 150
LR           = 1e-3
BATCH_SIZE   = 64
N_MODELS     = 10
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.epochs is not None:
    EPOCHS = args.epochs

RUN_TAG = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_10models_ep{EPOCHS}"

os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: H={ROWS} W={COLS} star={STAR_POS} epochs={EPOCHS} n_models={N_MODELS}")
print(f"Run tag: {RUN_TAG}")

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def model_predict(model, imgs_t):
    model.eval()
    with torch.no_grad():
        return model(imgs_t).reshape(-1).cpu().numpy()

def r_to_heatmap_single_star(r_state, state_index_map, star_pos):
    """Return (ROWS, COLS) heatmap for agent positions when star == star_pos."""
    hm = np.full((ROWS, COLS), np.nan, dtype=np.float32)
    for idx in range(state_index_map.shape[0]):
        ax, ay = int(state_index_map[idx, 0, 0]), int(state_index_map[idx, 0, 1])
        sx, sy = int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1])
        if (sx, sy) == star_pos:
            hm[ay, ax] = float(r_state[idx])
    return hm

# ──────────────────────────────────────────────
# Build datasets + full domain
# ──────────────────────────────────────────────
print("\n── Building datasets ──")
env = DeterministicGridWorld(rows=ROWS, cols=COLS, start=(0, 0), goal=STAR_POS)
set_seed(42)
X, y = generate_balanced_dataset(star_pos=STAR_POS, n_pos=N_POS, n_neg=N_NEG, H=ROWS, W=COLS, env=env)
N = len(X)
n_val = int(0.1 * N)
train_X, val_X = X[:N - n_val], X[N - n_val:]
train_y, val_y = y[:N - n_val], y[N - n_val:]
train_loader = DataLoader(GridDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(GridDataset(val_X,   val_y),   batch_size=BATCH_SIZE, shuffle=False)

images_full, _, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
images_full_t = torch.from_numpy(images_full).to(DEVICE)
print(f"Train: {len(train_X)}  Val: {len(val_X)}  Full domain: S={env.num_states}")

criterion = nn.MSELoss()

# ──────────────────────────────────────────────
# Train all 10 models
# ──────────────────────────────────────────────
r_preds = []   # full prediction vectors, one per model

for i in range(N_MODELS):
    seed = i        # seeds 0–9 guarantee distinct initialisations
    print(f"\n── Training Model {i} (MSE only, seed={seed}) ──")
    set_seed(seed)
    model = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN,
                    H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        tl = train_one_epoch(model, train_loader, opt, criterion, DEVICE)
        vl = eval_model(model, val_loader, criterion, DEVICE)
        if ep == 1 or ep % 30 == 0 or ep == EPOCHS:
            print(f"  Epoch {ep:3d}  train_mse={tl:.5f}  val_mse={vl:.5f}")

    r_full = model_predict(model, images_full_t)
    r_preds.append(r_full)
    print(f"Model {i} r: min={r_full.min():.3f} max={r_full.max():.3f} mean={r_full.mean():.3f}")

    pth_path = os.path.join(RESULTS_DIR, f"ind10_model{i}_{RUN_TAG}.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Saved: {pth_path}")


# ──────────────────────────────────────────────
# 2×5 summary figure — shared colour scale
# row 0: heatmap at star=(5,5)  row 1: heatmap at star=(1,1)
# columns: 5 evenly-spaced models from the 10
# ──────────────────────────────────────────────
print("\n── Generating 2×5 summary figure ──")

STAR_A = (5, 5)
STAR_B = (1, 1)
selected = [0, 2, 4, 6, 8]   # 5 models, evenly spaced across the 10

hms_A = [r_to_heatmap_single_star(r_preds[i], state_index_map, STAR_A) for i in selected]
hms_B = [r_to_heatmap_single_star(r_preds[i], state_index_map, STAR_B) for i in selected]

all_vals = np.stack(hms_A + hms_B)
vmin_all = float(np.nanmin(all_vals))
vmax_all = float(np.nanmax(all_vals))
norm = Normalize(vmin=vmin_all, vmax=vmax_all)
cmap = "viridis"

fig, axes = plt.subplots(2, 5, figsize=(5 * 3.0, 2 * 3.0), constrained_layout=False)
fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.06,
                    hspace=0.4, wspace=0.3)

row_stars  = [STAR_A, STAR_B]
row_hms    = [hms_A,  hms_B]

for row, (star, hms) in enumerate(zip(row_stars, row_hms)):
    for col, (model_idx, hm) in enumerate(zip(selected, hms)):
        ax = axes[row, col]
        ax.imshow(hm, origin='upper', cmap=cmap, norm=norm, aspect='equal')
        ax.scatter([star[0]], [star[1]], s=60, marker='*',
                   c='yellow', edgecolors='black', linewidths=0.5, zorder=4)
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.tick_params(labelsize=5)
        if row == 0:
            ax.set_title(f"Model {model_idx}  (seed={model_idx})", fontsize=8, pad=3)
        if col == 0:
            ax.set_ylabel(f"star={star}", fontsize=8)

# Shared colorbar on the right
cax = fig.add_axes([0.90, 0.10, 0.025, 0.78])
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cax, label='Predicted reward')

out_path = os.path.join(RESULTS_DIR, f"ind10_summary_{RUN_TAG}.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out_path}")

print("\nDone.")
