"""
Loss Landscape Experiment
==========================
Question: Are "corner" and "follow-star" actually two distinct local minima of
the ambiguous fixed-star loss landscape, or is only one of them a real minimum?

Protocol:
  Phase 1 — Pre-train two models on unambiguous datasets:
    M0 trained on CORNER data     (agent at (5,5), star=random   → reward 1)
    M1 trained on FOLLOW-STAR data (agent at star, star=random   → reward 1)

  Phase 2 — Copy both into the ambiguous landscape:
    Ambiguous dataset = fixed-star (star always at (5,5)).
    Both "corner" and "follow-star" are IDENTICAL on this data.
    Fine-tune M0 and M1 on this dataset for many epochs.

  Expected outcomes:
    Result 1 (two real minima): loss stays ≈ 0 for both, heatmaps unchanged.
    Result 2 (one real minimum): one model's loss decreases & heatmaps change,
                                 indicating it was not at a true minimum.

Outputs → results/landscape/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results/landscape", exist_ok=True)

import copy
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
from gridworld.policies import uniform_policy
from data.datasets import (generate_corner_dataset_aligned,
                            generate_follow_star_dataset,
                            generate_balanced_dataset,
                            generate_full_enumeration_dataset)
from models.cnn import DeepCNN, GridDataset
from starc.torch_ops import build_starc_precomputed, r_state_to_cnorm_flat
from training.loops import train_one_epoch, eval_model

# ──────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────
ROWS, COLS   = 7, 7
CHANNELS     = 3
STAR_POS     = (5, 5)
N_POS, N_NEG = 2000, 2000
HIDDEN_CHAN  = 4
FC_HIDDEN    = 32
EPOCHS_PRE   = 200    # pre-training epochs (unambiguous datasets)
EPOCHS_FT    = 300    # fine-tuning epochs on ambiguous landscape
LR_PRE       = 1e-3
LR_FT        = 1e-4   # smaller LR for fine-tuning to track subtle drift
BATCH_SIZE   = 64
GAMMA        = 0.9
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def make_model(seed):
    set_seed(seed)
    return DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN,
                   H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)

def make_loader(X, y, shuffle=True):
    return DataLoader(GridDataset(X, y), batch_size=BATCH_SIZE, shuffle=shuffle)

def get_predictions(model, imgs_t):
    model.eval()
    with torch.no_grad():
        return model(imgs_t).reshape(-1).cpu().numpy()

# ──────────────────────────────────────────────────────────────────
# Build full domain + STARC precomputation (done once)
# ──────────────────────────────────────────────────────────────────
print("\n── Building full domain ──")
set_seed(42)
env = DeterministicGridWorld(rows=ROWS, cols=COLS, start=(0, 0), goal=STAR_POS)
images_full, true_rewards_full, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
S = env.num_states
images_full_t = torch.from_numpy(images_full).to(DEVICE)
print(f"S = {S} states")

print("── Precomputing STARC ──")
policy_np = uniform_policy(env)
starc_pre = build_starc_precomputed(env, policy_np, gamma=GAMMA, device=DEVICE)

def starc_dist(m0, m1):
    r0 = get_predictions(m0, images_full_t)
    r1 = get_predictions(m1, images_full_t)
    with torch.no_grad():
        C0 = r_state_to_cnorm_flat(torch.from_numpy(r0).to(DEVICE), starc_pre, GAMMA)
        C1 = r_state_to_cnorm_flat(torch.from_numpy(r1).to(DEVICE), starc_pre, GAMMA)
    return torch.norm(C0 - C1).item()

# ──────────────────────────────────────────────────────────────────
# Ground-truth reward vectors (for reference STARC distances)
# ──────────────────────────────────────────────────────────────────
r_true_corner = np.array([
    1.0 if (int(state_index_map[i,0,0])==STAR_POS[0] and int(state_index_map[i,0,1])==STAR_POS[1])
    else 0.0
    for i in range(S)], dtype=np.float32)

r_true_star = np.array([
    1.0 if (state_index_map[i,0,0]==state_index_map[i,1,0] and
            state_index_map[i,0,1]==state_index_map[i,1,1])
    else 0.0
    for i in range(S)], dtype=np.float32)

with torch.no_grad():
    C_true_corner = r_state_to_cnorm_flat(torch.from_numpy(r_true_corner).to(DEVICE), starc_pre, GAMMA)
    C_true_star   = r_state_to_cnorm_flat(torch.from_numpy(r_true_star).to(DEVICE),   starc_pre, GAMMA)
    d_true = torch.norm(C_true_corner - C_true_star).item()
print(f"Reference: STARC(true_corner, true_star) = {d_true:.4f}")

def starc_dist_to_true(model):
    r = get_predictions(model, images_full_t)
    with torch.no_grad():
        C = r_state_to_cnorm_flat(torch.from_numpy(r).to(DEVICE), starc_pre, GAMMA)
        dc = torch.norm(C - C_true_corner).item()
        ds = torch.norm(C - C_true_star).item()
    return dc, ds

# ──────────────────────────────────────────────────────────────────
# Heatmap helpers
# ──────────────────────────────────────────────────────────────────
def r_to_heatmaps(r_state):
    grouped = defaultdict(list)
    for idx in range(S):
        ax, ay = int(state_index_map[idx,0,0]), int(state_index_map[idx,0,1])
        sx, sy = int(state_index_map[idx,1,0]), int(state_index_map[idx,1,1])
        grouped[(sx, sy)].append(((ax, ay), float(r_state[idx])))
    star_positions = sorted(grouped.keys(), key=lambda p: (p[1], p[0]))
    heatmaps = np.full((len(star_positions), ROWS, COLS), np.nan, dtype=np.float32)
    for k, star in enumerate(star_positions):
        for (ax, ay), val in grouped[star]:
            heatmaps[k, ay, ax] = val
    return heatmaps, star_positions

# Plot a 1×7 row of heatmaps (fixed star_y row for brevity)
def plot_heatmap_row(ax_row, heatmaps, star_positions, star_y, vmin, vmax,
                     cmap="viridis", mark_corner=True):
    norm = Normalize(vmin=vmin, vmax=vmax)
    star_to_idx = {sp: i for i, sp in enumerate(star_positions)}
    for gx, ax in enumerate(ax_row):
        star = (gx, star_y)
        if star in star_to_idx:
            k = star_to_idx[star]
            ax.imshow(heatmaps[k], origin='upper', cmap=cmap, norm=norm, aspect='equal')
            ax.scatter([gx], [star_y], s=50, marker='*', c='yellow', edgecolors='white', linewidths=0.5, zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"★=({gx},{star_y})", fontsize=5, pad=1)
    return norm

# ──────────────────────────────────────────────────────────────────
# PHASE 1: Pre-train M0 (corner) and M1 (follow-star)
# ──────────────────────────────────────────────────────────────────
criterion = nn.MSELoss()

# --- M0: corner ---
print(f"\n── Phase 1a: Pre-training M0 on CORNER data ({EPOCHS_PRE} epochs) ──")
set_seed(0)
X_corner, y_corner = generate_corner_dataset_aligned(
    star_pos=STAR_POS, n_pos=N_POS, n_neg=N_NEG, H=ROWS, W=COLS, env=env)
n_val = int(0.1 * len(X_corner))
corner_train = make_loader(X_corner[:-n_val], y_corner[:-n_val])
corner_val   = make_loader(X_corner[-n_val:], y_corner[-n_val:], shuffle=False)

m0 = make_model(seed=0)
opt0 = torch.optim.Adam(m0.parameters(), lr=LR_PRE)
for ep in range(1, EPOCHS_PRE + 1):
    tl = train_one_epoch(m0, corner_train, opt0, criterion, DEVICE)
    if ep == 1 or ep % 50 == 0 or ep == EPOCHS_PRE:
        vl = eval_model(m0, corner_val, criterion, DEVICE)
        dc, ds = starc_dist_to_true(m0)
        print(f"  Ep {ep:3d}  train={tl:.5f}  val={vl:.5f}  STARC→corner={dc:.3f}  →star={ds:.3f}")

# --- M1: follow-star (load from pre-trained checkpoint) ---
print(f"\n── Phase 1b: Loading M1 from pre-trained checkpoint (results/m1_check/m1_star.pth) ──")
m1 = make_model(seed=1)
ckpt = torch.load("results/m1_check/m1_star.pth", map_location=DEVICE)
m1.load_state_dict(ckpt["model_state_dict"])
dc, ds = starc_dist_to_true(m1)
print(f"  Loaded M1  STARC→corner={dc:.3f}  →star={ds:.3f}  ({'STAR' if ds < dc else 'CORNER'})")

# ──────────────────────────────────────────────────────────────────
# PHASE 2: Evaluate both on the ambiguous landscape before fine-tuning
# ──────────────────────────────────────────────────────────────────
print("\n── Phase 2: Evaluating on ambiguous landscape (fixed star) BEFORE fine-tuning ──")
set_seed(42)
X_amb, y_amb = generate_balanced_dataset(star_pos=STAR_POS, n_pos=N_POS, n_neg=N_NEG, H=ROWS, W=COLS, env=env)
n_val = int(0.1 * len(X_amb))
amb_train = make_loader(X_amb[:-n_val], y_amb[:-n_val])
amb_val   = make_loader(X_amb[-n_val:], y_amb[-n_val:], shuffle=False)

loss_m0_before = eval_model(m0, amb_val, criterion, DEVICE)
loss_m1_before = eval_model(m1, amb_val, criterion, DEVICE)
d_before = starc_dist(m0, m1)
print(f"  M0 loss on ambiguous landscape: {loss_m0_before:.6f}")
print(f"  M1 loss on ambiguous landscape: {loss_m1_before:.6f}")
print(f"  STARC(M0, M1) before fine-tuning: {d_before:.4f}")

# Capture predictions + heatmaps BEFORE fine-tuning
r_m0_before = get_predictions(m0, images_full_t)
r_m1_before = get_predictions(m1, images_full_t)
hm_m0_before, sp = r_to_heatmaps(r_m0_before)
hm_m1_before, _  = r_to_heatmaps(r_m1_before)

# ──────────────────────────────────────────────────────────────────
# PHASE 3: Fine-tune COPIES of both models on ambiguous landscape
# ──────────────────────────────────────────────────────────────────
print(f"\n── Phase 3: Fine-tuning on ambiguous landscape ({EPOCHS_FT} epochs, lr={LR_FT}) ──")

m0_ft = copy.deepcopy(m0)
m1_ft = copy.deepcopy(m1)
opt0_ft = torch.optim.Adam(m0_ft.parameters(), lr=LR_FT)
opt1_ft = torch.optim.Adam(m1_ft.parameters(), lr=LR_FT)

losses_m0_ft = []
losses_m1_ft = []
LOG_EVERY = 10

for ep in range(1, EPOCHS_FT + 1):
    tl0 = train_one_epoch(m0_ft, amb_train, opt0_ft, criterion, DEVICE)
    tl1 = train_one_epoch(m1_ft, amb_train, opt1_ft, criterion, DEVICE)
    losses_m0_ft.append(tl0)
    losses_m1_ft.append(tl1)
    if ep == 1 or ep % 50 == 0 or ep == EPOCHS_FT:
        vl0 = eval_model(m0_ft, amb_val, criterion, DEVICE)
        vl1 = eval_model(m1_ft, amb_val, criterion, DEVICE)
        print(f"  Ep {ep:3d}  M0_ft val={vl0:.6f}  M1_ft val={vl1:.6f}")

# Capture predictions + heatmaps AFTER fine-tuning
r_m0_after = get_predictions(m0_ft, images_full_t)
r_m1_after = get_predictions(m1_ft, images_full_t)
hm_m0_after, _ = r_to_heatmaps(r_m0_after)
hm_m1_after, _ = r_to_heatmaps(r_m1_after)

loss_m0_after = eval_model(m0_ft, amb_val, criterion, DEVICE)
loss_m1_after = eval_model(m1_ft, amb_val, criterion, DEVICE)
d_after = starc_dist(m0_ft, m1_ft)

dc0_a, ds0_a = starc_dist_to_true(m0_ft)
dc1_a, ds1_a = starc_dist_to_true(m1_ft)

print(f"\n── Results ──")
print(f"  M0 val loss:  before={loss_m0_before:.6f}  after={loss_m0_after:.6f}  "
      f"delta={loss_m0_after - loss_m0_before:+.6f}")
print(f"  M1 val loss:  before={loss_m1_before:.6f}  after={loss_m1_after:.6f}  "
      f"delta={loss_m1_after - loss_m1_before:+.6f}")
print(f"  STARC(M0,M1): before={d_before:.4f}  after={d_after:.4f}  delta={d_after - d_before:+.4f}")
print(f"  M0_ft STARC→corner={dc0_a:.3f}  →star={ds0_a:.3f}  ({'CORNER' if dc0_a < ds0_a else 'STAR'})")
print(f"  M1_ft STARC→corner={dc1_a:.3f}  →star={ds1_a:.3f}  ({'CORNER' if dc1_a < ds1_a else 'STAR'})")

verdict = ("RESULT 1 (two local minima — diversity is stable)"
           if abs(loss_m0_after - loss_m0_before) < 0.01 and
              abs(loss_m1_after - loss_m1_before) < 0.01
           else "RESULT 2 (loss landscape has fewer minima than expected — models drifted)")
print(f"\n  *** {verdict} ***")

# ──────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────
print("\n── Generating plots ──")
STAR_Y_SHOW = STAR_POS[1]  # show the row of star positions at y=5

# Compute a shared vmin/vmax across all model predictions for fair comparison
all_r = np.concatenate([r_m0_before, r_m1_before, r_m0_after, r_m1_after])
vmin_m = float(np.nanmin(all_r))
vmax_m = float(np.nanmax(all_r))

# ── Figure 1: Loss curves during fine-tuning ──
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(losses_m0_ft, label="M0 (pre-trained: corner)", color="steelblue", linewidth=1.5)
ax.plot(losses_m1_ft, label="M1 (pre-trained: follow-star)", color="coral",  linewidth=1.5)
ax.axhline(y=loss_m0_before, color="steelblue", linestyle="--", linewidth=0.8,
           label=f"M0 initial loss ({loss_m0_before:.4f})")
ax.axhline(y=loss_m1_before, color="coral",     linestyle="--", linewidth=0.8,
           label=f"M1 initial loss ({loss_m1_before:.4f})")
ax.set_xlabel("Fine-tuning epoch")
ax.set_ylabel("Training MSE loss")
ax.set_title("Loss curves on ambiguous (fixed-star) landscape\n"
             f"Result 1 = flat lines ≈ 0    Result 2 = one line drops significantly")
ax.legend(fontsize=9)
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("results/landscape/loss_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/landscape/loss_curves.png")

# ── Figure 2: Heatmap comparison (before vs after, for star_y row) ──
fig, axes = plt.subplots(4, COLS, figsize=(COLS * 1.8, 4 * 1.6))
rows_spec = [
    ("M0 (corner)   BEFORE", hm_m0_before, "viridis"),
    ("M0 (corner)   AFTER",  hm_m0_after,  "viridis"),
    ("M1 (star)     BEFORE", hm_m1_before, "viridis"),
    ("M1 (star)     AFTER",  hm_m1_after,  "viridis"),
]

for row_i, (label, hm, cmap) in enumerate(rows_spec):
    norm = plot_heatmap_row(axes[row_i], hm, sp, star_y=STAR_Y_SHOW,
                            vmin=vmin_m, vmax=vmax_m, cmap=cmap)
    axes[row_i, 0].set_ylabel(label, fontsize=7, rotation=0, labelpad=70, va='center')

fig.suptitle(
    f"Before vs After fine-tuning on ambiguous landscape (star row y={STAR_Y_SHOW})\n"
    f"Yellow ★ = star position for that panel.  All panels share the same colour scale.\n"
    f"Result 1 (stable): rows look identical before/after.  Result 2 (drift): rows differ.",
    fontsize=9
)
plt.tight_layout()
plt.savefig("results/landscape/heatmaps_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/landscape/heatmaps_before_after.png")

# ── Figure 3: STARC distances summary ──
dc0_b, ds0_b = starc_dist_to_true(m0)
dc1_b, ds1_b = starc_dist_to_true(m1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels_b = ["M0→corner", "M0→star", "M1→corner", "M1→star", "M0 vs M1"]
vals_b   = [dc0_b, ds0_b, dc1_b, ds1_b, d_before]
labels_a = ["M0_ft→corner", "M0_ft→star", "M1_ft→corner", "M1_ft→star", "M0_ft vs M1_ft"]
vals_a   = [dc0_a, ds0_a, dc1_a, ds1_a, d_after]
colors   = ["steelblue","steelblue","coral","coral","purple"]

for ax, vals, title in [(axes[0], vals_b, "BEFORE fine-tuning"),
                         (axes[1], vals_a, "AFTER fine-tuning")]:
    bars = ax.bar(labels_b, vals, color=colors)
    ax.axhline(y=d_true, color='gray', linestyle='--', linewidth=0.8,
               label=f"True corner↔star ({d_true:.3f})")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, 2.3)
    ax.set_ylabel("STARC distance")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', labelrotation=30, labelsize=7)

fig.suptitle("STARC distances before and after fine-tuning on ambiguous landscape", fontsize=10)
plt.tight_layout()
plt.savefig("results/landscape/starc_summary.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: results/landscape/starc_summary.png")

# ── Figure 4: Full 7×7 grid of per-star heatmaps (all 4 variants) ──
for tag, hm, title, cmap in [
    ("m0_before", hm_m0_before, "M0 (corner) BEFORE fine-tuning", "viridis"),
    ("m0_after",  hm_m0_after,  "M0 (corner) AFTER fine-tuning",  "viridis"),
    ("m1_before", hm_m1_before, "M1 (star)   BEFORE fine-tuning", "viridis"),
    ("m1_after",  hm_m1_after,  "M1 (star)   AFTER fine-tuning",  "viridis"),
]:
    fig2, axes2 = plt.subplots(ROWS, COLS, figsize=(COLS*1.6, ROWS*1.5))
    norm2 = Normalize(vmin=vmin_m, vmax=vmax_m)
    star_to_idx = {s: i for i, s in enumerate(sp)}
    for gy in range(ROWS):
        for gx in range(COLS):
            ax2 = axes2[gy, gx]
            star = (gx, gy)
            if star in star_to_idx:
                k = star_to_idx[star]
                ax2.imshow(hm[k], origin='upper', cmap=cmap, norm=norm2, aspect='auto')
                ax2.scatter([gx], [gy], s=50, marker='*', c='yellow',
                            edgecolors='white', linewidths=0.5, zorder=4)
            ax2.set_xticks([]); ax2.set_yticks([])
            ax2.set_title(f"★=({gx},{gy})", fontsize=5, pad=1)
    fig2.suptitle(
        title + "\nYellow ★ = star for each panel.  All panels share same colour scale.",
        fontsize=9, fontweight='bold'
    )
    fig2.subplots_adjust(right=0.87, hspace=0.55, wspace=0.35)
    cax2 = fig2.add_axes([0.89, 0.15, 0.02, 0.7])
    sm2 = ScalarMappable(cmap=cmap, norm=norm2)
    sm2.set_array([])
    fig2.colorbar(sm2, cax=cax2, label='Predicted reward')
    plt.savefig(f"results/landscape/{tag}.png", dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: results/landscape/{tag}.png")

print(f"\n── All plots saved to results/landscape/ ──")
print(f"\nFinal verdict: {verdict}")
