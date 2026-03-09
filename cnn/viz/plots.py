import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import defaultdict
from typing import Sequence, Optional, Tuple


# -------------------------
# Batched model forward over full domain
# -------------------------
def model_predict_full(model: torch.nn.Module,
                       images_full: np.ndarray,
                       device: Optional[torch.device] = None,
                       batch_size: int = 256) -> np.ndarray:
    device = torch.device("cpu") if device is None else device
    model = model.to(device)
    model.eval()
    images_t = torch.from_numpy(images_full).to(device) if isinstance(images_full, np.ndarray) else images_full.to(device)
    S = images_t.shape[0]
    preds = []
    with torch.no_grad():
        for i in range(0, S, batch_size):
            out = model(images_t[i:i + batch_size])
            preds.append(out.reshape(out.shape[0]).cpu().numpy())
    return np.concatenate(preds, axis=0)


# -------------------------
# Print conv1 kernels numerically
# -------------------------
def print_conv_weights(model: torch.nn.Module) -> None:
    weights = model.conv1.weight.detach().cpu().numpy()
    print(f"Conv layer weight shape: {weights.shape}  (out_channels, in_channels, kH, kW)")
    for i in range(weights.shape[0]):
        print(f"=== Filter {i} ===")
        for ch in range(weights.shape[1]):
            print(f"Channel {ch}:")
            print(np.round(weights[i, ch], 4))
        print()


# -------------------------
# Quadrant action heatmap (triangular cells, shape (rows, cols, 4))
# -------------------------
def plot_quadrant_action_heatmap(preds_actions: np.ndarray,
                                  star_coords: Optional[Tuple[int,int]] = None,
                                  title: Optional[str] = None,
                                  cmap: str = 'viridis',
                                  vmin: Optional[float] = None,
                                  vmax: Optional[float] = None,
                                  show_grid: bool = True,
                                  cell_edge_color: str = 'k') -> None:
    rows, cols, A = preds_actions.shape
    assert A == 4, "preds_actions must have shape (rows, cols, 4)"

    if vmin is None: vmin = float(preds_actions.min())
    if vmax is None: vmax = float(preds_actions.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(cols + 1, rows + 1))
    ax.set_xlim(0, cols); ax.set_ylim(0, rows); ax.set_aspect('equal')

    for r in range(rows):
        for c in range(cols):
            cx, cy = c + 0.5, r + 0.5
            bl, br, tl, tr = (c, r), (c+1, r), (c, r+1), (c+1, r+1)
            center = (cx, cy)
            triangles = [
                [tl, tr, center],   # action 0: up
                [br, bl, center],   # action 1: down
                [bl, tl, center],   # action 2: left
                [tr, br, center],   # action 3: right
            ]
            for a in range(4):
                color = cmap_fn(norm(float(preds_actions[r, c, a])))
                ax.add_patch(Polygon(triangles[a], closed=True,
                                     facecolor=color, edgecolor=cell_edge_color, linewidth=0.3))

    if show_grid:
        for x in range(cols + 1):
            ax.plot([x, x], [0, rows], color='black', linewidth=0.6, alpha=0.6)
        for y in range(rows + 1):
            ax.plot([0, cols], [y, y], color='black', linewidth=0.6, alpha=0.6)

    if star_coords is not None:
        sx, sy = star_coords
        ax.scatter([sx + 0.5], [sy + 0.5], s=200, marker='*',
                   edgecolor='white', facecolor='yellow', linewidth=1.0, zorder=5)

    sm = ScalarMappable(norm=norm, cmap=cmap_fn)
    sm.set_array(preds_actions)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04).set_label('Value')

    ax.set_xticks(np.arange(0.5, cols, 1)); ax.set_xticklabels(np.arange(0, cols))
    ax.set_yticks(np.arange(0.5, rows, 1)); ax.set_yticklabels(np.arange(0, rows))
    ax.set_xlabel('X coordinate (col)', fontsize=12)
    ax.set_ylabel('Y coordinate (row)', fontsize=12)
    if title: ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout(); plt.show()


# -------------------------
# Per-star agent heatmaps
# -------------------------
def build_agent_heatmaps_per_star(r_pred_full: np.ndarray,
                                   state_index_map: np.ndarray,
                                   env) -> Tuple[np.ndarray, list]:
    """
    r_pred_full: (S,)
    state_index_map: (S,2,2) — [idx,0]=agent[x,y], [idx,1]=star[x,y]
    Returns: heatmaps (K, rows, cols), star_positions list of (x,y)
    """
    S = state_index_map.shape[0]
    rows, cols = env.rows, env.cols
    grouped = defaultdict(list)
    for idx in range(S):
        agent = tuple(state_index_map[idx, 0].tolist())
        star = tuple(state_index_map[idx, 1].tolist())
        grouped[star].append((agent, float(r_pred_full[idx])))

    star_positions = sorted(grouped.keys(), key=lambda p: (p[1], p[0]))
    heatmaps = np.full((len(star_positions), rows, cols), np.nan, dtype=np.float32)
    for k, star in enumerate(star_positions):
        for (ax, ay), val in grouped[star]:
            heatmaps[k, ay, ax] = val
    return heatmaps, star_positions


def plot_star_heatmap_grid(heatmaps: np.ndarray,
                            star_positions: Sequence[Tuple[int,int]],
                            env,
                            title_prefix: str = "Model",
                            cmap: str = "viridis",
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            annotate: bool = False,
                            figsize_per_panel: Tuple[float,float] = (2.2, 2.0),
                            save_path: Optional[str] = None,
                            star_marker_kwargs: Optional[dict] = None):
    K, rows, cols = heatmaps.shape
    ncols = int(math.ceil(math.sqrt(K)))
    nrows = int(math.ceil(K / ncols))

    if vmin is None: vmin = float(np.nanmin(heatmaps))
    if vmax is None: vmax = float(np.nanmax(heatmaps))
    norm = Normalize(vmin=vmin, vmax=vmax)

    if star_marker_kwargs is None:
        star_marker_kwargs = {"s": 120, "marker": "*", "edgecolor": "white",
                               "facecolor": "yellow", "linewidth": 1.0, "zorder": 4}

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]))
    axes = np.atleast_2d(axes).flatten()

    for i in range(nrows * ncols):
        ax = axes[i]
        ax.set_xticks(np.arange(cols)); ax.set_xticklabels(np.arange(cols))
        ax.set_yticks(np.arange(rows)); ax.set_yticklabels(np.arange(rows))
        ax.set_aspect('equal'); ax.invert_yaxis()
        if i < K:
            ax.imshow(heatmaps[i], origin='upper', cmap=cmap, norm=norm)
            sx, sy = star_positions[i]
            ax.set_title(f"star=({sx},{sy})", fontsize=8)
            ax.scatter([sx], [sy], **star_marker_kwargs)
            if annotate:
                for y in range(rows):
                    for x in range(cols):
                        val = heatmaps[i, y, x]
                        if not np.isnan(val):
                            ax.text(x, y, f"{val:.2f}", ha='center', va='center', fontsize=6, color='white')
        else:
            ax.axis('off')

    fig.subplots_adjust(right=0.88, hspace=0.6, wspace=0.4)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(heatmaps)
    fig.colorbar(sm, cax=cax)
    fig.suptitle(title_prefix, fontsize=12)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    return fig


# -------------------------
# Compute preds_actions (rows, cols, 4) from model for a fixed star
# -------------------------
def compute_preds_actions_from_model(model: torch.nn.Module,
                                      env,
                                      star_pos,
                                      device: torch.device) -> np.ndarray:
    """Returns preds_actions (rows, cols, 4) using model's r(next_state)."""
    from gridworld.renderer import render_state_as_image
    rows, cols = env.rows, env.cols
    preds_actions = np.zeros((rows, cols, 4), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for r in range(rows):
            for c in range(cols):
                agent_pos = np.array([c, r], dtype=int)
                state = (agent_pos, star_pos)
                for a in range(env.num_actions):
                    next_state = env.next_state(state, a)
                    img = render_state_as_image(env, next_state)
                    img_t = torch.from_numpy(img).unsqueeze(0).float().to(device)
                    preds_actions[r, c, a] = model(img_t).item()
    return preds_actions
