"""
Replot all runs in results/ineffective_diversity_sweep/ from saved .pth checkpoints.

For each pair results/l2_model0_{TAG}.pth + results/l2_model1_{TAG}.pth,
regenerates the side-by-side heatmap (model0 left, model1 right) on a shared
color scale, with the final L2 distance shown in the title.

Run from cnn/:
    python scripts/replot_sweep.py
"""
import sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results/ineffective_diversity_sweep", exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

from gridworld import DeterministicGridWorld
from data.datasets import generate_full_enumeration_dataset
from models.cnn import DeepCNN

# ── Architecture constants (must match training script) ──
ROWS, COLS   = 7, 7
CHANNELS     = 3
HIDDEN_CHAN  = 4
FC_HIDDEN    = 32
STAR_POS     = (5, 5)
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Build full domain once ──
env = DeterministicGridWorld(rows=ROWS, cols=COLS, start=(0, 0), goal=STAR_POS)
print("Rendering full domain...")
images_full, _, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
images_full_t = torch.from_numpy(images_full).to(DEVICE)
S = env.num_states
print(f"Full domain S={S}")


def load_model(path):
    m = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN, H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def predict(model):
    with torch.no_grad():
        return model(images_full_t).reshape(-1).cpu().numpy()


def r_to_heatmaps(r_state):
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


def draw_grid(subfig, heatmaps, star_positions, title, cmap, vmin, vmax):
    axes = subfig.subplots(ROWS, COLS)
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
    subfig.suptitle(title, fontsize=9, fontweight='bold')
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    subfig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label='Predicted reward')


def replot(tag, alpha, epochs1, m0_path, m1_path):
    print(f"\nReplotting {tag} ...")
    m0 = load_model(m0_path)
    m1 = load_model(m1_path)

    r0 = predict(m0)
    r1 = predict(m1)
    final_l2 = float(np.linalg.norm(r0 - r1))

    hm0, sp = r_to_heatmaps(r0)
    hm1, _  = r_to_heatmaps(r1)

    all_vals = np.concatenate([hm0.reshape(-1), hm1.reshape(-1)])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    fig = plt.figure(figsize=(COLS * 3.4, ROWS * 1.5))
    sf_left, sf_right = fig.subfigures(1, 2, wspace=0.08)

    draw_grid(sf_left,  hm0, sp, f"Model 0 (MSE only)", "viridis", vmin, vmax)
    draw_grid(sf_right, hm1, sp, f"Model 1 (MSE − {alpha}·L2, {epochs1} epochs)", "viridis", vmin, vmax)

    fig.suptitle(
        f"alpha={alpha}  epochs_model1={epochs1}  L2(M0,M1)={final_l2:.4f}  |  {tag}",
        fontsize=10, y=1.01
    )

    out = f"results/ineffective_diversity_sweep/alpha{alpha}_ep1_{epochs1}_{tag}.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ── Find all model0 pths and match with model1 ──
results_dir = "results"
pattern = re.compile(r"l2_model0_(.+)\.pth$")

found = 0
for fname in sorted(os.listdir(results_dir)):
    m = pattern.match(fname)
    if not m:
        continue
    tag = m.group(1)
    m0_path = os.path.join(results_dir, fname)
    m1_path = os.path.join(results_dir, f"l2_model1_{tag}.pth")
    if not os.path.exists(m1_path):
        print(f"Skipping {tag}: no matching model1 pth")
        continue

    # Parse alpha and epochs1 from tag: {datetime}_alpha{ALPHA}_ep1_{EPOCHS_1}
    alpha_match  = re.search(r"_alpha([^_]+)_ep1_", tag)
    epochs_match = re.search(r"_ep1_(\d+)$", tag)
    alpha   = alpha_match.group(1)  if alpha_match  else "?"
    epochs1 = epochs_match.group(1) if epochs_match else "?"

    replot(tag, alpha, epochs1, m0_path, m1_path)
    found += 1

if found == 0:
    print("No model pairs found in results/. Run run_l2_ensemble.py first.")
else:
    print(f"\nDone. Replotted {found} run(s).")
