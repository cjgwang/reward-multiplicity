"""
Independent Ensemble Experiment
================================
Trains two CNN reward models independently — both with plain MSE, different
random seeds. Equivalent to alpha=0 in run_l2_ensemble.py but faster: no L2
computation during training.

Usage:
    python scripts/run_independent_ensemble.py
    python scripts/run_independent_ensemble.py --epochs 200
    python scripts/run_independent_ensemble.py --model0-only
"""
import sys, os, argparse
from datetime import datetime
from collections import defaultdict

CNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CNN_DIR)
RESULTS_DIR = os.path.join(CNN_DIR, "results")

parser = argparse.ArgumentParser()
parser.add_argument("--model0-only", action="store_true", help="Train and plot model 0 then exit")
parser.add_argument("--epochs", type=int, default=None, help="Override EPOCHS for both models")
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
SEED_0       = 0
SEED_1       = 1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.epochs is not None: EPOCHS = args.epochs

RUN_TAG = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_independent_ep{EPOCHS}"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "ineffective_diversity_sweep"), exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: H={ROWS} W={COLS} star={STAR_POS} epochs={EPOCHS}")
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

def compute_true_rewards(state_index_map, mode="corner"):
    S = state_index_map.shape[0]
    r = np.zeros(S, dtype=np.float32)
    for idx in range(S):
        ax, ay = int(state_index_map[idx, 0, 0]), int(state_index_map[idx, 0, 1])
        sx, sy = int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1])
        if mode == "corner":
            r[idx] = 1.0 if (ax == STAR_POS[0] and ay == STAR_POS[1]) else 0.0
        elif mode == "star":
            r[idx] = 1.0 if (ax == sx and ay == sy) else 0.0
    return r

def l2(a, b):
    return float(np.linalg.norm(a - b))

def r_to_heatmaps(r_state, state_index_map):
    grouped = defaultdict(list)
    for idx in range(state_index_map.shape[0]):
        ax, ay = int(state_index_map[idx, 0, 0]), int(state_index_map[idx, 0, 1])
        sx, sy = int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1])
        grouped[(sx, sy)].append(((ax, ay), float(r_state[idx])))
    star_positions = sorted(grouped.keys(), key=lambda p: (p[1], p[0]))
    heatmaps = np.full((len(star_positions), ROWS, COLS), np.nan, dtype=np.float32)
    for k, star in enumerate(star_positions):
        for (ax, ay), val in grouped[star]:
            heatmaps[k, ay, ax] = val
    return heatmaps, star_positions

def draw_grid_on_axes(axes, heatmaps, star_positions, cmap, vmin, vmax):
    norm = Normalize(vmin=vmin, vmax=vmax)
    star_to_idx = {sp: i for i, sp in enumerate(star_positions)}
    for gy in range(ROWS):
        for gx in range(COLS):
            ax = axes[gy, gx]
            star = (gx, gy)
            if star in star_to_idx:
                k = star_to_idx[star]
                ax.imshow(heatmaps[k], origin='upper', cmap=cmap, norm=norm, aspect='auto')
                ax.scatter([gx], [gy], s=40, marker='*', c='yellow', edgecolors='black', linewidths=0.4, zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"★=({gx},{gy})", fontsize=4, pad=1)
    return norm

def plot_heatmap_grid(heatmaps, star_positions, title, save_path, cmap="viridis", vmin=None, vmax=None):
    if vmin is None: vmin = float(np.nanmin(heatmaps))
    if vmax is None: vmax = float(np.nanmax(heatmaps))
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.6, ROWS * 1.5))
    norm = draw_grid_on_axes(axes, heatmaps, star_positions, cmap, vmin, vmax)
    fig.subplots_adjust(right=0.88, hspace=0.6, wspace=0.4)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, label='Predicted reward')
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

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
S = env.num_states
images_full_t = torch.from_numpy(images_full).to(DEVICE)
print(f"Train: {len(train_X)}  Val: {len(val_X)}  Full domain: S={S}")

criterion = nn.MSELoss()

# ──────────────────────────────────────────────
# Train Model 0
# ──────────────────────────────────────────────
print(f"\n── Training Model 0 (MSE only, seed={SEED_0}) ──")
set_seed(SEED_0)
model0 = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN, H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
opt0 = torch.optim.Adam(model0.parameters(), lr=LR)

for ep in range(1, EPOCHS + 1):
    tl = train_one_epoch(model0, train_loader, opt0, criterion, DEVICE)
    vl = eval_model(model0, val_loader, criterion, DEVICE)
    if ep == 1 or ep % 30 == 0 or ep == EPOCHS:
        print(f"  Epoch {ep:3d}  train_mse={tl:.5f}  val_mse={vl:.5f}")

r0_full = model_predict(model0, images_full_t)
print(f"Model 0 r: min={r0_full.min():.3f} max={r0_full.max():.3f} mean={r0_full.mean():.3f}")

torch.save(model0.state_dict(), os.path.join(RESULTS_DIR, f"l2_model0_{RUN_TAG}.pth"))
print(f"Saved: results/l2_model0_{RUN_TAG}.pth")

if args.model0_only:
    print("--model0-only: done.")
    sys.exit(0)

# ──────────────────────────────────────────────
# Train Model 1
# ──────────────────────────────────────────────
print(f"\n── Training Model 1 (MSE only, seed={SEED_1}) ──")
set_seed(SEED_1)
model1 = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN, H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
opt1 = torch.optim.Adam(model1.parameters(), lr=LR)

for ep in range(1, EPOCHS + 1):
    tl = train_one_epoch(model1, train_loader, opt1, criterion, DEVICE)
    vl = eval_model(model1, val_loader, criterion, DEVICE)
    if ep == 1 or ep % 30 == 0 or ep == EPOCHS:
        print(f"  Epoch {ep:3d}  train_mse={tl:.5f}  val_mse={vl:.5f}")

r1_full = model_predict(model1, images_full_t)
print(f"Model 1 r: min={r1_full.min():.3f} max={r1_full.max():.3f} mean={r1_full.mean():.3f}")

torch.save(model1.state_dict(), os.path.join(RESULTS_DIR, f"l2_model1_{RUN_TAG}.pth"))
print(f"Saved: results/l2_model1_{RUN_TAG}.pth")

# ──────────────────────────────────────────────
# Final distances
# ──────────────────────────────────────────────
r_corner = compute_true_rewards(state_index_map, mode="corner")
r_star   = compute_true_rewards(state_index_map, mode="star")

final_l2    = l2(r0_full, r1_full)
d_true      = l2(r_corner, r_star)
d_m0_corner = l2(r0_full, r_corner)
d_m0_star   = l2(r0_full, r_star)
d_m1_corner = l2(r1_full, r_corner)
d_m1_star   = l2(r1_full, r_star)

print(f"\n── Final L2(M0, M1): {final_l2:.4f} ──")
print(f"  True corner vs True star :  {d_true:.4f}")
print(f"  M0 vs corner={d_m0_corner:.4f}  star={d_m0_star:.4f}  → {'CORNER' if d_m0_corner < d_m0_star else 'STAR'}")
print(f"  M1 vs corner={d_m1_corner:.4f}  star={d_m1_star:.4f}  → {'CORNER' if d_m1_corner < d_m1_star else 'STAR'}")

# ──────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────
print("\n── Generating plots ──")
hm_corner, sp = r_to_heatmaps(r_corner, state_index_map)
hm_star,   _  = r_to_heatmaps(r_star,   state_index_map)
hm_m0,     _  = r_to_heatmaps(r0_full,  state_index_map)
hm_m1,     _  = r_to_heatmaps(r1_full,  state_index_map)

all_vals = np.concatenate([hm_m0.reshape(-1), hm_m1.reshape(-1)])
vmin_all, vmax_all = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))

plot_heatmap_grid(hm_corner, sp,
    title="True 'Corner' reward  (r=1 iff agent at (5,5))",
    save_path=os.path.join(RESULTS_DIR, f"ind_true_corner_{RUN_TAG}.png"), cmap="Blues", vmin=0, vmax=1)
plot_heatmap_grid(hm_star, sp,
    title="True 'Follow Star' reward  (r=1 iff agent == star)",
    save_path=os.path.join(RESULTS_DIR, f"ind_true_star_{RUN_TAG}.png"), cmap="Greens", vmin=0, vmax=1)
plot_heatmap_grid(hm_m0, sp,
    title=f"Model 0 (MSE only, seed={SEED_0})\nL2 to corner={d_m0_corner:.3f}  to star={d_m0_star:.3f}",
    save_path=os.path.join(RESULTS_DIR, f"ind_model0_{RUN_TAG}.png"), vmin=vmin_all, vmax=vmax_all)
plot_heatmap_grid(hm_m1, sp,
    title=f"Model 1 (MSE only, seed={SEED_1})\nL2 to corner={d_m1_corner:.3f}  to star={d_m1_star:.3f}",
    save_path=os.path.join(RESULTS_DIR, f"ind_model1_{RUN_TAG}.png"), vmin=vmin_all, vmax=vmax_all)

# Bar chart
fig, ax = plt.subplots(figsize=(6, 3.5))
categories = ["M0 vs\nCorner", "M0 vs\nStar", "M1 vs\nCorner", "M1 vs\nStar", "M0 vs M1", "True C\nvs Star"]
values     = [d_m0_corner, d_m0_star, d_m1_corner, d_m1_star, final_l2, d_true]
colors     = ["steelblue", "steelblue", "coral", "coral", "purple", "gray"]
bars = ax.bar(categories, values, color=colors)
ax.axhline(y=d_true, color='gray', linestyle='--', linewidth=0.8, label=f"True corner/star dist ({d_true:.3f})")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{val:.3f}",
            ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, max(values) * 1.25)
ax.set_ylabel("L2 distance")
ax.set_title("L2 distances (independent ensemble)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f"ind_distances_{RUN_TAG}.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: results/ind_distances_{RUN_TAG}.png")

# Sweep image: shared viridis scale
sweep_path = os.path.join(RESULTS_DIR, "ineffective_diversity_sweep", f"{RUN_TAG}.png")
shared_norm = Normalize(vmin=vmin_all, vmax=vmax_all)

fig = plt.figure(figsize=(COLS * 3.4, ROWS * 1.5))
sf_left, sf_right = fig.subfigures(1, 2, wspace=0.08)

axes_l = sf_left.subplots(ROWS, COLS)
draw_grid_on_axes(axes_l, hm_m1, sp, "viridis", vmin_all, vmax_all)
sf_left.suptitle(
    f"Model 1  (MSE only, seed={SEED_1},  {EPOCHS} epochs)\n"
    f"L2 to corner={d_m1_corner:.3f}  to star={d_m1_star:.3f}",
    fontsize=9, fontweight='bold')
sf_left.colorbar(ScalarMappable(cmap="viridis", norm=shared_norm),
                 ax=axes_l, fraction=0.02, pad=0.02, label='Predicted reward')

axes_r = sf_right.subplots(ROWS, COLS)
draw_grid_on_axes(axes_r, hm_m0, sp, "viridis", vmin_all, vmax_all)
sf_right.suptitle(
    f"Model 0  (MSE only, seed={SEED_0},  {EPOCHS} epochs)\n"
    f"L2 to corner={d_m0_corner:.3f}  to star={d_m0_star:.3f}",
    fontsize=9, fontweight='bold')
sf_right.colorbar(ScalarMappable(cmap="viridis", norm=shared_norm),
                  ax=axes_r, fraction=0.02, pad=0.02, label='Predicted reward')

fig.suptitle(
    f"Independent ensemble   L2(M0,M1)={final_l2:.4f}   {RUN_TAG}",
    fontsize=10, y=1.01)
plt.savefig(sweep_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {sweep_path}")

print("\nDone.")
