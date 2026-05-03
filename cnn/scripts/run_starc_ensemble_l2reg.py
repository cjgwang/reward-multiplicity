"""
STARC Diversity Ensemble with L2 Regularisation
================================================
Same as run_starc_ensemble.py but both models are trained with L2
regularisation (weight_decay in Adam).

Model 0: MSE  +  L2 reg
Model 1: MSE  -  alpha * STARC_dist(M1, M0)  +  L2 reg

Usage:
    python scripts/run_starc_ensemble_l2reg.py
    python scripts/run_starc_ensemble_l2reg.py --alpha 0.5 --epochs1 200 --weight-decay 1e-4
"""
import sys, os, argparse
from datetime import datetime
from collections import defaultdict

CNN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CNN_DIR)
RESULTS_DIR = os.path.join(CNN_DIR, "results")

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",        type=float, default=None, help="Override ALPHA")
parser.add_argument("--epochs1",      type=int,   default=None, help="Override EPOCHS_1")
parser.add_argument("--weight-decay", type=float, default=None, help="Override WEIGHT_DECAY")
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
from gridworld.policies import uniform_policy
from data.datasets import generate_balanced_dataset, generate_full_enumeration_dataset
from models.cnn import DeepCNN, GridDataset
from starc.torch_ops import build_starc_precomputed, r_state_to_cnorm_flat
from training.loops import train_one_epoch, eval_model

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
ROWS, COLS     = 7, 7
CHANNELS       = 3
STAR_POS       = (5, 5)
N_POS, N_NEG   = 2000, 2000
HIDDEN_CHAN     = 4
FC_HIDDEN      = 32
EPOCHS_0       = 150
EPOCHS_1       = 150
LR             = 1e-3
BATCH_SIZE     = 64
ALPHA          = 0.1
WEIGHT_DECAY   = 1e-4
GAMMA          = 0.9
SEED_0         = 0
SEED_1         = 1
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.alpha        is not None: ALPHA        = args.alpha
if args.epochs1      is not None: EPOCHS_1     = args.epochs1
if args.weight_decay is not None: WEIGHT_DECAY = args.weight_decay

RUN_TAG = f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_alpha{ALPHA}_wd{WEIGHT_DECAY}_ep1_{EPOCHS_1}"

os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_DIR = os.path.join(RESULTS_DIR, "starc_hacking")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: H={ROWS} W={COLS} star={STAR_POS} alpha={ALPHA} weight_decay={WEIGHT_DECAY} gamma={GAMMA}")
print(f"Seeds: model0={SEED_0} model1={SEED_1}")
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

def plot_heatmap_grid(heatmaps, star_positions, title, save_path, cmap="viridis", vmin=None, vmax=None):
    if vmin is None: vmin = float(np.nanmin(heatmaps))
    if vmax is None: vmax = float(np.nanmax(heatmaps))
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 1.6, ROWS * 1.5))
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
    fig.subplots_adjust(right=0.88, hspace=0.6, wspace=0.4)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, label='Predicted reward')
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")

# ──────────────────────────────────────────────
# Build dataset + full domain
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
val_loader   = DataLoader(GridDataset(val_X, val_y),     batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {len(train_X)}  Val: {len(val_X)}")

images_full, _, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
S = env.num_states
images_full_t = torch.from_numpy(images_full).to(DEVICE)
print(f"Full domain S={S}")

# ──────────────────────────────────────────────
# Precompute STARC matrices
# ──────────────────────────────────────────────
print("\n── Precomputing STARC (building P, F) ──")
policy_np = uniform_policy(env)
starc_pre = build_starc_precomputed(env, policy_np, gamma=GAMMA, device=DEVICE)
print(f"P shape: {starc_pre['P_t'].shape}  F shape: {starc_pre['F_t'].shape}")

criterion = nn.MSELoss()

# ──────────────────────────────────────────────
# Train Model 0 — MSE + L2 reg
# ──────────────────────────────────────────────
print(f"\n── Training Model 0 (MSE + L2 reg, seed={SEED_0}) ──")
set_seed(SEED_0)
model0 = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN, H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
opt0 = torch.optim.Adam(model0.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for ep in range(1, EPOCHS_0 + 1):
    tl = train_one_epoch(model0, train_loader, opt0, criterion, DEVICE)
    vl = eval_model(model0, val_loader, criterion, DEVICE)
    if ep == 1 or ep % 30 == 0 or ep == EPOCHS_0:
        print(f"  Epoch {ep:3d}  train_mse={tl:.5f}  val_mse={vl:.5f}")

r0_full = model_predict(model0, images_full_t)
print(f"Model 0 r: min={r0_full.min():.3f} max={r0_full.max():.3f} mean={r0_full.mean():.3f}")

with torch.no_grad():
    C0_flat = r_state_to_cnorm_flat(
        torch.from_numpy(r0_full).to(DEVICE), starc_pre, GAMMA
    ).detach()

torch.save(model0.state_dict(), os.path.join(RESULTS_DIR, f"starc_l2reg_model0_{RUN_TAG}.pth"))
print(f"Saved: starc_l2reg_model0_{RUN_TAG}.pth")

# ──────────────────────────────────────────────
# Train Model 1 — MSE - alpha * STARC + L2 reg
# ──────────────────────────────────────────────
print(f"\n── Training Model 1 (MSE - {ALPHA}*STARC + L2 reg, seed={SEED_1}) ──")
set_seed(SEED_1)
model1 = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN, H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
opt1 = torch.optim.Adam(model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

for ep in range(1, EPOCHS_1 + 1):
    model1.train()
    total_mse = total_starc = 0.0
    n = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt1.zero_grad()

        mse    = criterion(model1(imgs), labels)
        r_full = model1(images_full_t).reshape(S)
        C1_flat = r_state_to_cnorm_flat(r_full, starc_pre, GAMMA)
        starc_dist = torch.norm(C1_flat - C0_flat)

        loss = mse - ALPHA * starc_dist
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=5.0)
        opt1.step()

        total_mse   += mse.item()        * imgs.size(0)
        total_starc += starc_dist.item() * imgs.size(0)
        n += imgs.size(0)

    if ep == 1 or ep % 30 == 0 or ep == EPOCHS_1:
        vl = eval_model(model1, val_loader, criterion, DEVICE)
        print(f"  Epoch {ep:3d}  mse={total_mse/n:.5f}  starc={total_starc/n:.4f}  val_mse={vl:.5f}")

r1_full = model_predict(model1, images_full_t)
torch.save(model1.state_dict(), os.path.join(RESULTS_DIR, f"starc_l2reg_model1_{RUN_TAG}.pth"))
print(f"Saved: starc_l2reg_model1_{RUN_TAG}.pth")

# ──────────────────────────────────────────────
# Final distances
# ──────────────────────────────────────────────
with torch.no_grad():
    C1_flat_final = r_state_to_cnorm_flat(
        torch.from_numpy(r1_full).to(DEVICE), starc_pre, GAMMA
    )
    final_starc = torch.norm(C0_flat - C1_flat_final).item()

r_corner = compute_true_rewards(state_index_map, mode="corner")
r_star   = compute_true_rewards(state_index_map, mode="star")

with torch.no_grad():
    C_corner    = r_state_to_cnorm_flat(torch.from_numpy(r_corner).to(DEVICE), starc_pre, GAMMA)
    C_star      = r_state_to_cnorm_flat(torch.from_numpy(r_star).to(DEVICE),   starc_pre, GAMMA)
    d_true      = torch.norm(C_corner - C_star).item()
    d_m0_corner = torch.norm(C0_flat - C_corner).item()
    d_m0_star   = torch.norm(C0_flat - C_star).item()
    d_m1_corner = torch.norm(C1_flat_final - C_corner).item()
    d_m1_star   = torch.norm(C1_flat_final - C_star).item()

print(f"\n── Final STARC(M0, M1): {final_starc:.4f}  (max=2.0) ──")
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
    save_path=os.path.join(OUT_DIR, f"starc_l2reg_true_corner_{RUN_TAG}.png"),
    cmap="Blues", vmin=0, vmax=1)
plot_heatmap_grid(hm_star, sp,
    title="True 'Follow Star' reward  (r=1 iff agent == star)",
    save_path=os.path.join(OUT_DIR, f"starc_l2reg_true_star_{RUN_TAG}.png"),
    cmap="Greens", vmin=0, vmax=1)
plot_heatmap_grid(hm_m0, sp,
    title=f"Model 0 (MSE + L2 reg wd={WEIGHT_DECAY}, seed={SEED_0})\nSTARC to corner={d_m0_corner:.3f}  to star={d_m0_star:.3f}",
    save_path=os.path.join(OUT_DIR, f"starc_l2reg_model0_{RUN_TAG}.png"),
    vmin=vmin_all, vmax=vmax_all)
plot_heatmap_grid(hm_m1, sp,
    title=f"Model 1 (MSE − {ALPHA}·STARC + L2 reg wd={WEIGHT_DECAY}, seed={SEED_1})\nSTARC to corner={d_m1_corner:.3f}  to star={d_m1_star:.3f}",
    save_path=os.path.join(OUT_DIR, f"starc_l2reg_model1_{RUN_TAG}.png"),
    vmin=vmin_all, vmax=vmax_all)

# ── Summary figure: single star row for quick comparison ──
star_y_row  = STAR_POS[1]
star_to_idx = {s: i for i, s in enumerate(sp)}
labels_rows = [
    ("True Corner",                       hm_corner, 0.0,     1.0,     "Blues"),
    ("True Follow Star",                  hm_star,   0.0,     1.0,     "Greens"),
    (f"Model 0 (MSE+L2 wd={WEIGHT_DECAY})", hm_m0,  vmin_all, vmax_all, "viridis"),
    (f"Model 1 (STARC+L2 wd={WEIGHT_DECAY})", hm_m1, vmin_all, vmax_all, "plasma"),
]

fig, axes = plt.subplots(4, COLS, figsize=(COLS * 1.8, 4 * 1.7))
for row_i, (row_label, hm, vm, vx, cm) in enumerate(labels_rows):
    norm = Normalize(vmin=vm, vmax=vx)
    for gx in range(COLS):
        ax = axes[row_i, gx]
        star = (gx, star_y_row)
        if star in star_to_idx:
            k = star_to_idx[star]
            ax.imshow(hm[k], origin='upper', cmap=cm, norm=norm, aspect='equal')
            ax.scatter([gx], [star_y_row], s=60, marker='*', c='yellow',
                       edgecolors='white', linewidths=0.5, zorder=4)
        ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
        ax.tick_params(labelsize=4, length=2)
        ax.set_title(f"★=({gx},{star_y_row})", fontsize=6, pad=1)
        if gx == 0:
            ax.set_ylabel(row_label, fontsize=7, rotation=0, labelpad=60, va='center')
        if row_i == len(labels_rows) - 1:
            ax.set_xlabel(f"x={gx}", fontsize=6)
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[row_i, :], fraction=0.012, pad=0.01, label='Predicted reward')

fig.suptitle(f"Comparison — star row y={star_y_row}  (alpha={ALPHA}, wd={WEIGHT_DECAY})\n"
             f"STARC(M0,M1)={final_starc:.3f}  True STARC(corner,star)={d_true:.3f}",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"starc_l2reg_comparison_row_{RUN_TAG}.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: starc_l2reg_comparison_row_{RUN_TAG}.png")

# ── STARC distance bar chart ──
fig, ax = plt.subplots(figsize=(6, 3.5))
categories = ["M0 vs\nCorner", "M0 vs\nStar", "M1 vs\nCorner", "M1 vs\nStar", "M0 vs M1", "True C\nvs Star"]
values     = [d_m0_corner, d_m0_star, d_m1_corner, d_m1_star, final_starc, d_true]
colors     = ["steelblue", "steelblue", "coral", "coral", "purple", "gray"]
bars = ax.bar(categories, values, color=colors)
ax.axhline(y=d_true, color='gray', linestyle='--', linewidth=0.8, label=f"True corner/star dist ({d_true:.3f})")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
            ha='center', va='bottom', fontsize=8)
ax.set_ylim(0, max(values) * 1.25)
ax.set_ylabel("STARC distance")
ax.set_title(f"STARC distances (alpha={ALPHA}, weight_decay={WEIGHT_DECAY})")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"starc_l2reg_distances_{RUN_TAG}.png"), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: starc_l2reg_distances_{RUN_TAG}.png")

print(f"\nDone. Results saved to results/starc_hacking/")
print(f"  Model 0: corner={d_m0_corner:.3f}  star={d_m0_star:.3f}  → {'CORNER' if d_m0_corner < d_m0_star else 'STAR'}")
print(f"  Model 1: corner={d_m1_corner:.3f}  star={d_m1_star:.3f}  → {'CORNER' if d_m1_corner < d_m1_star else 'STAR'}")
