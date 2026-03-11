"""Generate per-star agent heatmaps for ensemble_model_0 and ensemble_model_1."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from gridworld import DeterministicGridWorld
from models import CNN
from viz.plots import model_predict_full, build_agent_heatmaps_per_star, plot_star_heatmap_grid

device = torch.device("cpu")
env = DeterministicGridWorld(rows=7, cols=7)

data = np.load("grid_full_domain.npz")
images_full = data["images_full"]
state_index_map = data["state_index_map"]

os.makedirs("results", exist_ok=True)

for i in range(2):
    ckpt = torch.load(f"ensemble_model_{i}.pth", map_location=device, weights_only=True)
    model = CNN(in_channels=3, hidden_channels=1, H=7, W=7).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    r_pred = model_predict_full(model, images_full, device=device)
    heatmaps, star_positions = build_agent_heatmaps_per_star(r_pred, state_index_map, env)

    save_path = f"results/ensemble_member_{i}_heatmaps.png"
    import matplotlib
    matplotlib.use("Agg")
    plot_star_heatmap_grid(heatmaps, star_positions, env,
                           title_prefix=f"Ensemble Member {i}",
                           save_path=save_path)
    print(f"Saved {save_path}")
