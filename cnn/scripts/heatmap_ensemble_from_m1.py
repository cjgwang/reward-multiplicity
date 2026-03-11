"""
Generate heatmaps for ensemble_from_m1_model_0.pth and ensemble_from_m1_model_1.pth.

Usage:
    python scripts/heatmap_ensemble_from_m1.py --full_domain grid_full_domain.npz
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("results/ensemble_from_m1", exist_ok=True)

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict

from models import CNN


def r_to_heatmaps(r_state, state_index_map, rows, cols):
    grouped = defaultdict(list)
    for idx in range(len(r_state)):
        ax, ay = int(state_index_map[idx, 0, 0]), int(state_index_map[idx, 0, 1])
        sx, sy = int(state_index_map[idx, 1, 0]), int(state_index_map[idx, 1, 1])
        grouped[(sx, sy)].append(((ax, ay), float(r_state[idx])))
    star_positions = sorted(grouped.keys(), key=lambda p: (p[1], p[0]))
    heatmaps = np.full((len(star_positions), rows, cols), np.nan, dtype=np.float32)
    for k, star in enumerate(star_positions):
        for (ax, ay), val in grouped[star]:
            heatmaps[k, ay, ax] = val
    return heatmaps, star_positions


def plot_heatmap_grid(heatmaps, star_positions, rows, cols, title, save_path,
                      vmin=None, vmax=None, cmap="viridis"):
    if vmin is None: vmin = float(np.nanmin(heatmaps))
    if vmax is None: vmax = float(np.nanmax(heatmaps))
    norm = Normalize(vmin=vmin, vmax=vmax)
    star_to_idx = {sp: i for i, sp in enumerate(star_positions)}

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.5))
    for gy in range(rows):
        for gx in range(cols):
            ax = axes[gy, gx]
            star = (gx, gy)
            if star in star_to_idx:
                k = star_to_idx[star]
                ax.imshow(heatmaps[k], origin='upper', cmap=cmap, norm=norm, aspect='auto')
                ax.scatter([gx], [gy], s=60, marker='*', c='yellow',
                           edgecolors='white', linewidths=0.5, zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"★=({gx},{gy})", fontsize=5, pad=1)

    fig.subplots_adjust(right=0.88, hspace=0.6, wspace=0.4)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax, label='Predicted reward')
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_domain", type=str, default="grid_full_domain.npz")
    parser.add_argument("--model0", type=str, default="ensemble_from_m1_model_0.pth")
    parser.add_argument("--model1", type=str, default="ensemble_from_m1_model_1.pth")
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    # Load full domain
    data = np.load(args.full_domain)
    images_full = data["images_full"]
    state_index_map = data["state_index_map"]
    S = images_full.shape[0]
    images_full_t = torch.from_numpy(images_full).to(device)

    def load_model(path):
        m = CNN(in_channels=args.channels, hidden_channels=args.hidden_channels,
                H=args.rows, W=args.cols).to(device)
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        return m

    def predict(model):
        with torch.no_grad():
            return model(images_full_t).reshape(S).cpu().numpy()

    print(f"Loading {args.model0}...")
    m0 = load_model(args.model0)
    r0 = predict(m0)
    print(f"Model 0 rewards: min={r0.min():.4f}  max={r0.max():.4f}  mean={r0.mean():.4f}")

    print(f"Loading {args.model1}...")
    m1 = load_model(args.model1)
    r1 = predict(m1)
    print(f"Model 1 rewards: min={r1.min():.4f}  max={r1.max():.4f}  mean={r1.mean():.4f}")

    hm0, sp = r_to_heatmaps(r0, state_index_map, args.rows, args.cols)
    hm1, _  = r_to_heatmaps(r1, state_index_map, args.rows, args.cols)

    # Shared colour scale
    all_vals = np.concatenate([hm0.reshape(-1), hm1.reshape(-1)])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))

    print("\nGenerating heatmaps...")
    plot_heatmap_grid(hm0, sp, args.rows, args.cols,
                      title=f"ensemble_from_m1_model_0  (min={r0.min():.3f} max={r0.max():.3f})",
                      save_path="results/ensemble_from_m1/model_0.png",
                      vmin=vmin, vmax=vmax)

    plot_heatmap_grid(hm1, sp, args.rows, args.cols,
                      title=f"ensemble_from_m1_model_1  (min={r1.min():.3f} max={r1.max():.3f})",
                      save_path="results/ensemble_from_m1/model_1.png",
                      vmin=vmin, vmax=vmax)

    # Side-by-side comparison for a single star row (y=5)
    star_y = 5
    star_to_idx = {s: i for i, s in enumerate(sp)}
    fig, axes = plt.subplots(2, args.cols, figsize=(args.cols * 1.8, 2 * 1.6))
    norm = Normalize(vmin=vmin, vmax=vmax)
    for row_i, (hm, label) in enumerate([(hm0, "Model 0"), (hm1, "Model 1")]):
        for gx in range(args.cols):
            ax = axes[row_i, gx]
            star = (gx, star_y)
            if star in star_to_idx:
                k = star_to_idx[star]
                ax.imshow(hm[k], origin='upper', cmap='viridis', norm=norm, aspect='equal')
                ax.scatter([gx], [star_y], s=60, marker='*', c='yellow',
                           edgecolors='white', linewidths=0.5, zorder=4)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"★=({gx},{star_y})", fontsize=6, pad=1)
            if gx == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=50, va='center')
    fig.suptitle(f"Ensemble from M1 — star row y={star_y}", fontsize=10)
    plt.tight_layout()
    plt.savefig("results/ensemble_from_m1/comparison_row.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved: results/ensemble_from_m1/comparison_row.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
