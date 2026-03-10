"""
Train M1 to follow the star reward and visualise it.

Dataset: for every sample, the star is at a random position.
Reward = 1 when agent is at the same cell as the star, else 0.
This means M1 must learn to compare the two channels rather than
memorising a fixed location.

If M1 is trained correctly, each panel in the 7x7 grid should show
ONE bright cell — the cell where the star is for that panel — and
everything else dark.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results/m1_check", exist_ok=True)

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
from collections import defaultdict

from gridworld import DeterministicGridWorld
from data.datasets import generate_follow_star_dataset, generate_full_enumeration_dataset
from models.cnn import DeepCNN, GridDataset
from training.loops import train_one_epoch, eval_model

# ── Config ──────────────────────────────────────────────────────
ROWS, COLS   = 7, 7
CHANNELS     = 3
N_POS, N_NEG = 4000, 4000   # more data, star in all positions
HIDDEN_CHAN  = 4
FC_HIDDEN    = 32
EPOCHS       = 300
LR           = 1e-3
BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

set_seed(1)

# ── Dataset ──────────────────────────────────────────────────────
print("Building follow-star dataset...")
env_for_data = DeterministicGridWorld(rows=ROWS, cols=COLS)
X, y = generate_follow_star_dataset(env_for_data, n_pos=N_POS, n_neg=N_NEG)
N = len(X)
n_val = int(0.1 * N)
train_loader = DataLoader(GridDataset(X[:-n_val], y[:-n_val]), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(GridDataset(X[-n_val:], y[-n_val:]),  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {N - n_val}  Val: {n_val}")

# ── Train ────────────────────────────────────────────────────────
model = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN,
                H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"\nTraining M1 (follow-star) for {EPOCHS} epochs...")
for ep in range(1, EPOCHS + 1):
    tl = train_one_epoch(model, train_loader, opt, criterion, DEVICE)
    if ep == 1 or ep % 50 == 0 or ep == EPOCHS:
        vl = eval_model(model, val_loader, criterion, DEVICE)
        print(f"  Epoch {ep:3d}  train_mse={tl:.6f}  val_mse={vl:.6f}")

# ── Full-domain predictions ──────────────────────────────────────
print("\nBuilding full domain (all agent×star pairs)...")
env = DeterministicGridWorld(rows=ROWS, cols=COLS)
images_full, true_rewards_full, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
S = env.num_states
images_full_t = torch.from_numpy(images_full).to(DEVICE)

model.eval()
with torch.no_grad():
    r_pred = model(images_full_t).reshape(-1).cpu().numpy()

# Also compute true star reward for comparison
r_true = np.array([
    1.0 if (state_index_map[i,0,0] == state_index_map[i,1,0] and
            state_index_map[i,0,1] == state_index_map[i,1,1])
    else 0.0
    for i in range(S)], dtype=np.float32)

print(f"Predicted reward stats: min={r_pred.min():.3f}  max={r_pred.max():.3f}  mean={r_pred.mean():.3f}")
print(f"True reward stats:      min={r_true.min():.3f}  max={r_true.max():.3f}  mean={r_true.mean():.3f}")

# ── Build heatmaps ───────────────────────────────────────────────
def r_to_heatmaps(r_state, state_index_map, rows, cols):
    grouped = defaultdict(list)
    for idx in range(len(r_state)):
        ax, ay = int(state_index_map[idx,0,0]), int(state_index_map[idx,0,1])
        sx, sy = int(state_index_map[idx,1,0]), int(state_index_map[idx,1,1])
        grouped[(sx, sy)].append(((ax, ay), float(r_state[idx])))
    star_positions = sorted(grouped.keys(), key=lambda p: (p[1], p[0]))
    heatmaps = np.full((len(star_positions), rows, cols), np.nan, dtype=np.float32)
    for k, star in enumerate(star_positions):
        for (ax, ay), val in grouped[star]:
            heatmaps[k, ay, ax] = val
    return heatmaps, star_positions

hm_pred, sp = r_to_heatmaps(r_pred, state_index_map, ROWS, COLS)
hm_true, _  = r_to_heatmaps(r_true, state_index_map, ROWS, COLS)

# ── Plot: true vs predicted, shared colour scale ─────────────────
def plot_grid(hm, star_positions, rows, cols, title, save_path, vmin, vmax, cmap="viridis"):
    """
    7×7 grid of panels. Each panel = one possible star position.
    The yellow star marker shows where the star is for that panel.
    A well-trained follow-star model should have ONE bright cell per panel,
    exactly at the star marker.
    """
    star_to_idx = {s: i for i, s in enumerate(star_positions)}
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 1.6))
    for gy in range(rows):
        for gx in range(cols):
            ax = axes[gy, gx]
            star = (gx, gy)
            if star in star_to_idx:
                k = star_to_idx[star]
                ax.imshow(hm[k], origin='upper', cmap=cmap, norm=norm, aspect='equal')
                # mark the star position with a yellow star
                ax.scatter([gx], [gy], s=60, marker='*', c='yellow',
                           edgecolors='black', linewidths=0.4, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"★=({gx},{gy})", fontsize=5, pad=1)

    # shared colourbar
    fig.subplots_adjust(right=0.87, hspace=0.55, wspace=0.35)
    cax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='Predicted reward')
    fig.suptitle(title, fontsize=10, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# Use a shared scale across both true and predicted
vmin = min(float(np.nanmin(hm_true)), float(np.nanmin(hm_pred)))
vmax = max(float(np.nanmax(hm_true)), float(np.nanmax(hm_pred)))

plot_grid(hm_true, sp, ROWS, COLS,
          title="TRUE follow-star reward\n"
                "Each panel = one star position (yellow ★).\n"
                "Bright cell = agent is at the star → reward 1. All others = 0.",
          save_path="results/m1_check/true_star.png",
          vmin=vmin, vmax=vmax, cmap="viridis")

plot_grid(hm_pred, sp, ROWS, COLS,
          title="M1 PREDICTED reward (trained on follow-star data)\n"
                "Each panel = one star position (yellow ★).\n"
                "If trained correctly: ONE bright cell per panel, exactly at ★.",
          save_path="results/m1_check/m1_predicted.png",
          vmin=vmin, vmax=vmax, cmap="viridis")

# ── Save model ───────────────────────────────────────────────────
torch.save({"model_state_dict": model.state_dict()}, "results/m1_check/m1_star.pth")
print("Saved model: results/m1_check/m1_star.pth")
print("\nDone. Check results/m1_check/")
