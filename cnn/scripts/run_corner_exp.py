"""
Corner experiment: train CNN on the aligned-corner dataset, then visualize.

The "corner" dataset has reward 1.0 when agent is at star_pos regardless of
where the star actually is — so the model must learn agent position, not star position.

After training, this script:
  1. Runs the trained model over the full domain (all (agent,star) pairs)
  2. Plots per-star agent heatmaps
  3. Plots a triangular action heatmap for an example star
  4. Prints numeric conv kernels

Usage:
    python scripts/run_corner_exp.py
    python scripts/run_corner_exp.py --epochs 200 --alpha 0.0 --hidden_channels 1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ExperimentConfig
from gridworld import DeterministicGridWorld
from data import generate_corner_dataset_aligned, generate_full_enumeration_dataset
from models import CNN, GridDataset
from training import train_one_epoch, eval_model
from viz import (model_predict_full, build_agent_heatmaps_per_star,
                 plot_star_heatmap_grid, compute_preds_actions_from_model,
                 plot_quadrant_action_heatmap, print_conv_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--star_pos", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--n_pos", type=int, default=2000)
    parser.add_argument("--n_neg", type=int, default=2000)
    parser.add_argument("--hidden_channels", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_model", type=str, default="corner_model.pth")
    parser.add_argument("--example_star", type=int, nargs=2, default=[5, 5], metavar=("X", "Y"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    star_pos = tuple(args.star_pos)

    # --- Dataset ---
    X, y = generate_corner_dataset_aligned(
        star_pos=star_pos, n_pos=args.n_pos, n_neg=args.n_neg,
        H=args.rows, W=args.cols,
    )
    N = len(X)
    n_val = int(0.1 * N)
    train_X, val_X = X[:N - n_val], X[N - n_val:]
    train_y, val_y = y[:N - n_val], y[N - n_val:]

    train_loader = DataLoader(GridDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(GridDataset(val_X,   val_y),   batch_size=args.batch_size, shuffle=False)

    # --- Model ---
    model = CNN(in_channels=args.channels, hidden_channels=args.hidden_channels,
                H=args.rows, W=args.cols).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"Training corner experiment: {len(train_X)} train, {len(val_X)} val")
    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss   = eval_model(model, val_loader, criterion, device)
        if ep == 1 or ep % 20 == 0 or ep == args.epochs:
            print(f"Epoch {ep:03d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

    torch.save({"model_state_dict": model.state_dict()}, args.save_model)
    print(f"Saved model to {args.save_model}")

    # --- Full-domain evaluation and visualization ---
    env = DeterministicGridWorld(rows=args.rows, cols=args.cols, start=(0, 0), goal=star_pos)

    print("Rendering full-domain images...")
    images_full, true_rewards_full, state_index_map = generate_full_enumeration_dataset(env, channels=args.channels)
    S = images_full.shape[0]

    print("Forwarding full domain through model...")
    r_state = model_predict_full(model, images_full, device=device)
    print(f"r_state: min={r_state.min():.4f}  max={r_state.max():.4f}  mean={r_state.mean():.4f}")

    # Per-star heatmaps
    heatmaps, star_positions = build_agent_heatmaps_per_star(r_state, state_index_map, env)
    plot_star_heatmap_grid(heatmaps, star_positions, env, title_prefix="Corner experiment — per-star heatmaps")

    # Triangular action heatmap for example star
    example_star = np.array(args.example_star, dtype=int)
    print(f"Computing triangular action heatmap for star={tuple(args.example_star)}...")
    preds_actions = compute_preds_actions_from_model(model, env, example_star, device)
    plot_quadrant_action_heatmap(preds_actions, star_coords=tuple(args.example_star),
                                  title=f"Action heatmap star={tuple(args.example_star)}")

    # Conv kernels
    print("\nConv kernels:")
    print_conv_weights(model)


if __name__ == "__main__":
    main()
