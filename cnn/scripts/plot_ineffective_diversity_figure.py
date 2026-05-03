"""
Publication-quality figure for ICML: ineffective diversity in L2 ensemble.

Layout (4 columns, 3 rows):
  Col 0 : Model 0 heatmaps
  Col 1 : Model 1 heatmaps (raw)
  Col 2 : Residual heatmaps  (r1·b+c) − r0
  Col 3 : Bar chart          raw L2 vs residual L2 for the 3 chosen stars

Columns 0-1 share a viridis colorscale. Col 2 uses a diverging RdBu_r scale.

Run from cnn/:
    python scripts/plot_ineffective_diversity_figure.py
    python scripts/plot_ineffective_diversity_figure.py --tag 20260502T160355_alpha1e-12_ep1_100
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

from gridworld import DeterministicGridWorld
from data.datasets import generate_full_enumeration_dataset
from models.cnn import DeepCNN

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="20260502T160355_alpha1e-12_ep1_100")
args = parser.parse_args()
TAG = args.tag

# ── Constants ──
CNN_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(CNN_DIR, "results")
ROWS, COLS  = 7, 7
CHANNELS    = 3
HIDDEN_CHAN = 4
FC_HIDDEN   = 32
STAR_POS    = (5, 5)
DEVICE      = torch.device("cpu")

CHOSEN_STARS = [(1, 1), (5, 4)]
STAR_LABELS  = [f"★=({x},{y})" for x, y in CHOSEN_STARS]

# ── Build domain ──
env = DeterministicGridWorld(rows=ROWS, cols=COLS, start=(0, 0), goal=STAR_POS)
print("Rendering full domain...")
images_full, _, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
images_full_t = torch.from_numpy(images_full).to(DEVICE)
S = env.num_states

star_groups = defaultdict(list)
for idx in range(S):
    star = (int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1]))
    star_groups[star].append(idx)
star_positions = sorted(star_groups.keys(), key=lambda p: (p[1], p[0]))


def load_model(path):
    m = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN,
                H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def predict(model):
    with torch.no_grad():
        return model(images_full_t).reshape(-1).cpu().numpy()


def reward_heatmap_for_star(r_state, star_xy):
    grid = np.full((ROWS, COLS), np.nan, dtype=np.float32)
    for idx in star_groups[star_xy]:
        ax = int(state_index_map[idx, 0, 0])
        ay = int(state_index_map[idx, 0, 1])
        grid[ay, ax] = r_state[idx]
    return grid


# ── Load checkpoints ──
m0_path  = os.path.join(RESULTS_DIR, f"l2_model0_{TAG}.pth")
m1_path  = os.path.join(RESULTS_DIR, f"l2_model1_{TAG}.pth")
npz_path = os.path.join(RESULTS_DIR, "ineffective_diversity_sweep", "scaling", TAG, "affine_results.npz")

print(f"Loading models from tag: {TAG}")
r0 = predict(load_model(m0_path))
r1 = predict(load_model(m1_path))

data           = np.load(npz_path)
b_all          = data["b"]
c_all          = data["c"]
saved_star_pos = [tuple(p) for p in data["star_positions"]]
star_to_npz    = {sp: i for i, sp in enumerate(saved_star_pos)}

# ── Per-star data ──
hm0_list, hm1_list, hmdiff_list = [], [], []
raw_l2_list, res_l2_list = [], []

for star_xy in CHOSEN_STARS:
    ki  = star_to_npz[star_xy]
    b   = b_all[ki]
    c   = c_all[ki]

    hm0  = reward_heatmap_for_star(r0, star_xy)
    hm1  = reward_heatmap_for_star(r1, star_xy)
    hm1t = hm1 * b + c

    hm0_list.append(hm0)
    hm1_list.append(hm1)
    hmdiff_list.append(hm1t - hm0)

    idxs    = star_groups[star_xy]
    r0_vec  = r0[idxs]
    r1_vec  = r1[idxs]
    r1t_vec = r1_vec * b + c
    raw_l2_list.append(float(np.linalg.norm(r0_vec - r1_vec)))
    res_l2_list.append(float(np.linalg.norm(r0_vec - r1t_vec)))

# ── Colorscales ──
norm      = Normalize(vmin=-1, vmax=1)
cmap      = "viridis"
diff_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
diff_cmap = "RdBu_r"

# ── Figure ──
n_rows = len(CHOSEN_STARS)
fig_w  = 3 * 2.2 + 3.4   # 3 heatmap cols + bar col
fig_h  = n_rows * 2.2 + 0.9

fig = plt.figure(figsize=(fig_w, fig_h))

outer = gridspec.GridSpec(1, 4, figure=fig,
                           width_ratios=[1, 1, 1, 1.3],
                           wspace=0.32,
                           left=0.08, right=0.97,
                           top=0.88, bottom=0.12)

# ── Cols 0-2: heatmaps ──
hm_specs = [
    ("Model 0",                          hm0_list,   cmap,      norm),
    ("Model 1 (raw)",                    hm1_list,   cmap,      norm),
    (r"Residual: $(r_1 \cdot b+c)-r_0$", hmdiff_list, diff_cmap, diff_norm),
]

for col_idx, (title, hm_list, c, n) in enumerate(hm_specs):
    inner = gridspec.GridSpecFromSubplotSpec(n_rows, 1,
                                              subplot_spec=outer[col_idx],
                                              hspace=0.12)
    for row_idx in range(n_rows):
        ax = fig.add_subplot(inner[row_idx])
        ax.imshow(hm_list[row_idx], origin="upper", cmap=c, norm=n, aspect="equal")

        sx, sy = CHOSEN_STARS[row_idx]
        ax.scatter([sx], [sy], s=90, marker="*", c="yellow",
                   edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

        if col_idx == 0:
            ax.set_ylabel(STAR_LABELS[row_idx], fontsize=9, labelpad=4)

        if row_idx == 0:
            ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

# ── Colorbars ──
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.08, 0.05, 0.44, 0.022])
fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", label="Predicted reward")

sm_diff = ScalarMappable(cmap=diff_cmap, norm=diff_norm)
sm_diff.set_array([])
cbar_diff_ax = fig.add_axes([0.56, 0.05, 0.16, 0.022])
fig.colorbar(sm_diff, cax=cbar_diff_ax, orientation="horizontal", label="Residual")

# ── Col 3: bar chart ──
ax_bar = fig.add_subplot(outer[3])
x_idx  = np.arange(n_rows)
width  = 0.35
bars_raw = ax_bar.bar(x_idx - width / 2, raw_l2_list, width,
                       label=r"Raw $L_2(r_0,\,r_1)$", color="steelblue", alpha=0.85)
bars_res = ax_bar.bar(x_idx + width / 2, res_l2_list, width,
                       label=r"Residual $L_2$ (after affine fit)", color="coral", alpha=0.85)

ax_bar.set_xticks(x_idx)
ax_bar.set_xticklabels(STAR_LABELS, fontsize=8)
ax_bar.set_ylabel("$L_2$ distance", fontsize=9)
ax_bar.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=1)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax_bar.set_axisbelow(True)

for bar in list(bars_raw) + list(bars_res):
    h = bar.get_height()
    ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                f"{h:.3f}", ha="center", va="bottom", fontsize=6.5)

# ── Save ──
out_dir  = os.path.join(RESULTS_DIR, "ineffective_diversity_sweep", "scaling", TAG)
out_path = os.path.join(out_dir, "icml_figure.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out_path}")
