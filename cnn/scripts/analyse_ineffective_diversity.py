"""
Analyse ineffective diversity in L2 ensemble results.

For each model0/model1 checkpoint pair, finds per-star affine transforms b(star), c(star)
that minimise ||r0(star, :) - (r1(star, :) * b(star) + c(star))||_2.

If residuals are small despite large raw L2(r0, r1), the diversity is "ineffective":
model1 is just a per-star rescaling/shift of model0, encoding the same policy.

Outputs saved to: results/ineffective_diversity_sweep/scaling/{TAG}/

Run from cnn/:
    python scripts/analyse_ineffective_diversity.py
    python scripts/analyse_ineffective_diversity.py --tag 20260502T151804_alpha1e-12_ep1_100
"""
import sys, os, re, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from numpy.linalg import lstsq

from gridworld import DeterministicGridWorld
from data.datasets import generate_full_enumeration_dataset
from models.cnn import DeepCNN

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default=None, help="Specific RUN_TAG to analyse (default: all)")
args = parser.parse_args()

# ── Architecture constants (must match training script) ──
ROWS, COLS  = 7, 7
CHANNELS    = 3
HIDDEN_CHAN = 4
FC_HIDDEN   = 32
STAR_POS    = (5, 5)
DEVICE      = torch.device("cpu")   # cpu fine for inference only

# ── Build full domain once ──
env = DeterministicGridWorld(rows=ROWS, cols=COLS, start=(0, 0), goal=STAR_POS)
print("Rendering full domain...")
images_full, _, state_index_map = generate_full_enumeration_dataset(env, channels=CHANNELS)
images_full_t = torch.from_numpy(images_full).to(DEVICE)
S = env.num_states

# Build star groupings once: star_positions list and indices per star
star_groups = defaultdict(list)
for idx in range(S):
    star = (int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1]))  # (x, y)
    star_groups[star].append(idx)
star_positions = sorted(star_groups.keys(), key=lambda p: (p[1], p[0]))
num_stars  = len(star_positions)
num_agents = len(star_groups[star_positions[0]])
print(f"Full domain: S={S}, num_stars={num_stars}, num_agents={num_agents}")


def load_model(path):
    m = DeepCNN(in_channels=CHANNELS, hidden_channels=HIDDEN_CHAN,
                H=ROWS, W=COLS, fc_hidden=FC_HIDDEN).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def predict(model):
    with torch.no_grad():
        return model(images_full_t).reshape(-1).cpu().numpy()


def build_star_agent_matrix(r_state):
    """Returns (num_stars, num_agents) matrix, rows ordered by star_positions."""
    mat = np.zeros((num_stars, num_agents), dtype=np.float32)
    for k, star in enumerate(star_positions):
        mat[k] = r_state[star_groups[star]]
    return mat


def fit_affine_per_star(r0_mat, r1_mat):
    """
    For each star k, solve: r0[k,:] ≈ b[k] * r1[k,:] + c[k]
    Returns b (num_stars,), c (num_stars,), residual_l2 (num_stars,), r2 (num_stars,)
    """
    b   = np.zeros(num_stars)
    c   = np.zeros(num_stars)
    res = np.zeros(num_stars)
    r2  = np.zeros(num_stars)
    for k in range(num_stars):
        y = r0_mat[k]
        x = r1_mat[k]
        A = np.column_stack([x, np.ones_like(x)])
        sol, _, _, _ = lstsq(A, y, rcond=None)
        b[k], c[k] = sol
        y_pred = x * b[k] + c[k]
        res[k] = float(np.linalg.norm(y - y_pred))
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2[k] = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return b, c, res, r2


def star_vec_to_grid(vec):
    """Maps a (num_stars,) vector to a (ROWS, COLS) grid for imshow."""
    grid = np.full((ROWS, COLS), np.nan)
    for k, (sx, sy) in enumerate(star_positions):
        grid[sy, sx] = vec[k]
    return grid


def plot_grid_heatmap(ax, grid, title, cmap="viridis", vmin=None, vmax=None, center=None):
    lo = vmin if vmin is not None else float(np.nanmin(grid))
    hi = vmax if vmax is not None else float(np.nanmax(grid))
    if center is not None and lo < center < hi:
        norm = TwoSlopeNorm(vcenter=center, vmin=lo, vmax=hi)
    else:
        norm = Normalize(vmin=lo, vmax=hi)
    im = ax.imshow(grid, origin='upper', cmap=cmap, norm=norm, aspect='equal')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("star x"); ax.set_ylabel("star y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def analyse(tag, m0_path, m1_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n── Analysing {tag} ──")

    r0 = predict(load_model(m0_path))
    r1 = predict(load_model(m1_path))

    raw_l2 = float(np.linalg.norm(r0 - r1))
    print(f"  Raw L2(r0, r1) = {raw_l2:.4f}")

    r0_mat = build_star_agent_matrix(r0)
    r1_mat = build_star_agent_matrix(r1)

    b, c, residuals, r2 = fit_affine_per_star(r0_mat, r1_mat)

    # Reconstruct transformed r1
    r1_transformed = r1_mat * b[:, None] + c[:, None]
    transformed_l2 = float(np.linalg.norm(r0_mat - r1_transformed))
    print(f"  Transformed L2(r0, r1*b+c) = {transformed_l2:.4f}")
    print(f"  b: min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}")
    print(f"  c: min={c.min():.4f}  max={c.max():.4f}  mean={c.mean():.4f}")
    print(f"  Residual L2 per star: min={residuals.min():.4f}  max={residuals.max():.4f}  mean={residuals.mean():.4f}")
    print(f"  R² per star: min={r2.min():.4f}  max={r2.max():.4f}  mean={r2.mean():.4f}")
    print(f"\n  {'star(x,y)':<12} {'b':>8} {'c':>8} {'residual':>10} {'R²':>8}")
    print(f"  {'-'*48}")
    for k, (sx, sy) in enumerate(star_positions):
        print(f"  ({sx},{sy}){'':<8} {b[k]:>8.4f} {c[k]:>8.4f} {residuals[k]:>10.4f} {r2[k]:>8.4f}")

    # ── Figure 1: b, c, residual, R² grids ──
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plot_grid_heatmap(axes[0, 0], star_vec_to_grid(b), "b(star)  [scale factor]",
                      cmap="RdBu_r", center=0)
    plot_grid_heatmap(axes[0, 1], star_vec_to_grid(c), "c(star)  [shift]",
                      cmap="RdBu_r", center=0)
    plot_grid_heatmap(axes[1, 0], star_vec_to_grid(residuals), "Residual L2 per star",
                      cmap="Reds")
    plot_grid_heatmap(axes[1, 1], star_vec_to_grid(r2), "R² per star  (1 = perfect affine fit)",
                      cmap="Greens", vmin=0, vmax=1)
    fig.suptitle(
        f"Affine fit: r0 ≈ r1·b(star) + c(star)\n"
        f"Raw L2={raw_l2:.4f}   Transformed L2={transformed_l2:.4f}   {tag}",
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "affine_fit_summary.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: affine_fit_summary.png")

    # ── Figure 2: scatter r0 vs r1 and r0 vs r1*b+c per star ──
    fig, axes = plt.subplots(ROWS, COLS, figsize=(COLS * 2.2, ROWS * 2.2))
    for k, (sx, sy) in enumerate(star_positions):
        ax = axes[sy, sx]
        y  = r0_mat[k]
        x  = r1_mat[k]
        xt = r1_transformed[k]
        all_vals = np.concatenate([y, x, xt])
        lo, hi = all_vals.min(), all_vals.max()
        ax.scatter(x,  y, s=8, alpha=0.6, color="steelblue", label="r1 (raw)")
        ax.scatter(xt, y, s=8, alpha=0.6, color="coral",     label="r1·b+c")
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.6)
        ax.set_title(f"★=({sx},{sy})\nb={b[k]:.2f} c={c[k]:.2f}\nR²={r2[k]:.3f}", fontsize=5, pad=1)
        ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=6, label='r1 raw'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='coral',     markersize=6, label='r1·b+c')]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Per-star scatter: r0 vs r1 (blue) and r0 vs r1·b+c (red)\n{tag}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_star_scatter.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: per_star_scatter.png")

    # ── Figure 3: bar chart L2 before vs after per star ──
    fig, ax = plt.subplots(figsize=(14, 4))
    x_idx = np.arange(num_stars)
    raw_l2_per_star = [float(np.linalg.norm(r0_mat[k] - r1_mat[k])) for k in range(num_stars)]
    ax.bar(x_idx - 0.2, raw_l2_per_star, width=0.4, label="Raw L2(r0, r1)", color="steelblue", alpha=0.8)
    ax.bar(x_idx + 0.2, residuals,        width=0.4, label="Residual L2 after affine fit", color="coral", alpha=0.8)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"({sx},{sy})" for sx, sy in star_positions], rotation=90, fontsize=5)
    ax.set_ylabel("L2 distance")
    ax.set_title(f"Per-star L2: raw vs after affine fit\n"
                 f"Total raw={raw_l2:.4f}   total transformed={transformed_l2:.4f}", fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_star_l2_comparison.png"), dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: per_star_l2_comparison.png")

    # ── Save numerical results ──
    np.savez(os.path.join(out_dir, "affine_results.npz"),
             b=b, c=c, residuals=residuals, r2=r2,
             raw_l2=raw_l2, transformed_l2=transformed_l2,
             star_positions=np.array(star_positions))
    print(f"  Saved: affine_results.npz")


# ── Find model pairs ──
results_dir = "results"
pattern = re.compile(r"l2_model0_(.+)\.pth$")

pairs = []
for fname in sorted(os.listdir(results_dir)):
    m = pattern.match(fname)
    if not m:
        continue
    tag = m.group(1)
    if args.tag and tag != args.tag:
        continue
    m1_path = os.path.join(results_dir, f"l2_model1_{tag}.pth")
    if not os.path.exists(m1_path):
        print(f"Skipping {tag}: no matching model1 pth")
        continue
    pairs.append((tag, os.path.join(results_dir, fname), m1_path))

if not pairs:
    print("No model pairs found. Run run_l2_ensemble.py first.")
    sys.exit(1)

for tag, m0_path, m1_path in pairs:
    out_dir = os.path.join("results", "ineffective_diversity_sweep", "scaling", tag)
    analyse(tag, m0_path, m1_path, out_dir)

print("\nDone.")
