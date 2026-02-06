import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ----------------------------------------------------------------------------------------------------------
# Rendering

# Renders a gridworld with ★ for the star and ● for the start position
def render_cartesian_gridworld(env) -> None:
    fig, ax = plt.subplots(figsize=(env.cols + 1, env.rows + 1))

    # Set the limits to match a 5x5 grid
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)

    # Draw the grid lines
    # We use a loop to ensure lines are at integer boundaries
    for i in range(env.rows + 1):
        ax.axhline(i, color='black', lw=1.5, zorder=2)
    for i in range(env.cols + 1):
        ax.axvline(i, color='black', lw=1.5, zorder=2)

    # Place the Star and the Start at corner
    # To center them in the cell, we add 0.5 to the integer coordinates.
    start = env.start_coord + 0.5
    star = env.goal_coord + 0.5

    ax.text(start[0], start[1], '⬤',
            ha='center', va='center',
            fontsize=25, color="#FF0000",
            zorder=3)
    ax.text(star[0], star[1], '★',
            ha='center', va='center',
            fontsize=45, color='#FFD700',
            zorder=3)

    # Configure Number Lines (Axes)
    # Ticks at 0.5, 1.5, 2.5, 3.5, 4.5 correspond to labels 0, 1, 2, 3, 4
    ax.set_xticks(np.arange(0.5, env.cols, 1))
    ax.set_xticklabels(np.arange(env.cols))
    ax.set_yticks(np.arange(0.5, env.rows, 1))
    ax.set_yticklabels(np.arange(env.rows))

    # Axis Labels
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.set_title("Gridworld", fontsize=14, fontweight='bold')

    # Ensure the plot is square and clean
    ax.set_aspect('equal')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

"""
    Draw each cell partitioned into 4 triangles (up/down/left/right) colored by preds_actions.
    preds_actions: (cols, rows, 4)  actions order: 0=up,1=down,2=left,3=right
"""
def plot_quadrant_action_heatmap(preds_actions, star_coords=None, title=None,
                                 cmap='viridis', vmin=None, vmax=None,
                                 show_grid=True, cell_edge_color='k') -> None:

    cols, rows, A = preds_actions.shape
    assert A == 4, "preds_actions must be shape (4, rows, cols)"

    if vmin is None:
        vmin = float(preds_actions.min())
    if vmax is None:
        vmax = float(preds_actions.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=(cols + 1, rows + 1))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    for r in range(rows):
        for c in range(cols):
            cx = c + 0.5
            cy = r + 0.5
            bl = (c, r)
            br = (c + 1, r)
            tl = (c, r + 1)
            tr = (c + 1, r + 1)
            center = (cx, cy)

            tri_up    = [tl, tr, center]    # action 0
            tri_down  = [br, bl, center]    # action 1
            tri_left  = [bl, tl, center]    # action 2
            tri_right = [tr, br, center]    # action 3
            triangles = [tri_up, tri_down, tri_left, tri_right]

            for a in range(4):
                val = preds_actions[c, r, a]
                color = cmap(norm(val))
                poly = Polygon(triangles[a], closed=True,
                               facecolor=color, edgecolor=cell_edge_color, linewidth=0.3)
                ax.add_patch(poly)

    # grid lines
    if show_grid:
        for x in range(cols + 1):
            ax.plot([x, x], [0, rows], color='black', linewidth=0.6, alpha=0.6)
        for y in range(rows + 1):
            ax.plot([0, cols], [y, y], color='black', linewidth=0.6, alpha=0.6)

    # star markers
    if star_coords is not None:
        sc, sr = star_coords
        ax.scatter([sc + 0.5], [sr + 0.5], s=200, marker='*',
                    edgecolor='white', facecolor='yellow', linewidth=1.0, zorder=5)

    # shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(preds_actions)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalised Reward Value')

    ax.set_xticks(np.arange(0.5, cols, 1))
    ax.set_yticks(np.arange(0.5, rows, 1))
    ax.set_xticklabels(np.arange(0, cols))
    ax.set_yticklabels(np.arange(0, rows))

    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    if(title is not None):
        ax.set_title(title, fontsize=14, fontweight='bold')

    plt.show()

# Visualizes the reward in env using standard settings for fixed star position
def visualize_reward_quadrant(env, reward: Reward, star: Position = None, title: str = "Reward Map") -> None:
    if(star is None):
        star = env.goal_coord
    cols, rows, A = env.cols, env.rows, env.num_actions
    preds_actions = np.zeros((cols, rows, A), dtype=np.float32)

    for i in range(env.num_positions):
        for a in range(A):
            coords = env.idx_to_coord(i)
            s1 = (coords, star)
            s2 = env.next_state(s1, a)
            preds_actions[coords[0], coords[1], a] = reward((s1, a, s2))

    plot_quadrant_action_heatmap(preds_actions, star, title, vmin=-1, vmax=1)