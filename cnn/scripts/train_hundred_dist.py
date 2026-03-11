#!/usr/bin/env python3
"""
scripts/train_and_save_many.py

Train T independent CNNs on a fixed-star balanced dataset and save:
 - one PNG per trial containing per-star heatmaps (grid of panels),
 - checkpoints for each trained model,
 - a compressed NPZ summary with r_states and L2 distances.

Usage:
    python scripts/train_and_save_many.py --trials 100 --epochs 25 --out fixedstar_results.npz
"""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import time
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from gridworld.env import DeterministicGridWorld

# project imports (match your repo layout)
from config import ExperimentConfig
from data import generate_balanced_dataset, generate_full_enumeration_dataset
from models import DeepCNN, GridDataset
from training import train_one_epoch, eval_model

# visualization helpers (ensure they exist in your repo as earlier)
# - model_predict_full(images_full)
# - build_agent_heatmaps_per_star(r_pred_full, state_index_map, env)
# - plot_star_heatmap_grid(heatmaps, star_positions, env, title_prefix=..., vmin=..., vmax=..., save_path=...)
from viz import model_predict_full, build_agent_heatmaps_per_star, plot_star_heatmap_grid

import matplotlib
matplotlib.use("Agg")  # headless backend for saving figures

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--rows", type=int, default=7)
    p.add_argument("--cols", type=int, default=7)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--star_pos", type=int, nargs=2, default=[5, 5])
    p.add_argument("--n_pos", type=int, default=2000)
    p.add_argument("--n_neg", type=int, default=2000)
    p.add_argument("--hidden_channels", type=int, default=8)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--start_seed", type=int, default=1000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out", type=str, default="fixedstar_results.npz")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--plots_dir", type=str, default="plots")
    args = p.parse_args()

    cfg = ExperimentConfig(
        rows=args.rows, cols=args.cols, channels=args.channels,
        star_pos=tuple(args.star_pos),
        n_pos=args.n_pos, n_neg=args.n_neg,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.start_seed, device=args.device, frozen_ckpt_path=None
    )

    # device selection
    device = cfg.get_device() if hasattr(cfg, "get_device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Device:", device)

    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.plots_dir)

    # create env instance to enumerate full domain
    env = DeterministicGridWorld(rows=args.rows, cols=args.cols, start=(0,0), goal=(args.cols-1, args.rows-1))

    print("Generating full-domain enumeration dataset...")
    images_full, true_rewards_full, state_index_map = generate_full_enumeration_dataset(env, channels=args.channels)
    images_full = images_full.astype(np.float32)                      # (S, C, H, W)
    true_rewards_full = np.array(true_rewards_full).reshape(-1)       # (S,)
    S = images_full.shape[0]
    print(f"Full domain S={S}, images_full shape={images_full.shape}")

    T = args.trials
    r_states = np.zeros((T, S), dtype=np.float32)
    l2s = np.zeros((T,), dtype=np.float32)
    ckpt_paths = []

    for t in range(T):
        seed = args.start_seed + t
        print(f"\n=== Trial {t+1}/{T} seed={seed} ===")
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        X, y = generate_balanced_dataset(star_pos=[(1,1),(2,2),(3,3),(4,4),(5,5)], n_pos=args.n_pos, n_neg=args.n_neg, H=args.rows, W=args.cols)
        N = len(X)
        n_val = max(1, int(0.1 * N))
        train_X, val_X = X[:N - n_val], X[N - n_val:]
        train_y, val_y = y[:N - n_val], y[N - n_val:]
        train_loader = DataLoader(GridDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(GridDataset(val_X,   val_y),   batch_size=args.batch_size, shuffle=False)

        model = DeepCNN(in_channels=args.channels, hidden_channels=args.hidden_channels, H=args.rows, W=args.cols).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.MSELoss()

        # Train
        t0 = time.time()
        for ep in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, opt, criterion, device)
            val_loss = eval_model(model, val_loader, criterion, device)
            if ep == 1 or ep % 10 == 0 or ep == args.epochs:
                print(f"  Epoch {ep:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")
        dt = time.time() - t0
        print(f"  Finished training (took {dt:.1f}s)")

        # save checkpoint
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        ckpt_name = os.path.join(args.checkpoint_dir, f"frozen_seed{seed}_{ts}.pth")
        torch.save({"model_state_dict": model.state_dict(), "seed": seed, "cfg": vars(cfg)}, ckpt_name)
        ckpt_paths.append(ckpt_name)
        print("Saved checkpoint:", ckpt_name)

        # forward full-domain to get r_state
        model.eval()
        r_state = model_predict_full(model, images_full, device=device, batch_size=256).reshape(-1)
        r_states[t] = r_state.astype(np.float32)

        # L2 to true reward
        l2 = float(np.linalg.norm(r_state - true_rewards_full))
        l2s[t] = l2
        print(f"  Trial {t}: L2 to true reward = {l2:.6f}")

        # convert r_state -> per-star heatmaps and save a PNG for this trial
        heatmaps, star_positions = build_agent_heatmaps_per_star(r_state, state_index_map, env)
        # name the PNG with seed and trial index
        png_name = os.path.join(args.plots_dir, f"trial_{t+1:03d}_seed{seed}_{ts}.png")
        try:
            # use your plotting helper to draw and save the multi-panel grid
            plot_star_heatmap_grid(heatmaps, star_positions, env,
                                   title_prefix=f"Trial {t+1} seed={seed}",
                                   cmap="viridis", vmin=None, vmax=None,
                                   annotate=False, figsize_per_panel=(2.4,2.0),
                                   save_path=png_name,
                                   star_marker_kwargs=None)
            print("Saved heatmap PNG:", png_name)
        except Exception as e:
            print("Failed to save heatmap for trial", t+1, ":", e)

    # save results summary (r_states, l2s, ckpt_paths, true_rewards, state_index_map, cfg)
    out_name = args.out
    base, ext = os.path.splitext(out_name)
    ts_all = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_full = f"{base}_{T}trials_{ts_all}{ext}"
    np.savez_compressed(out_full,
                        r_states=r_states,
                        l2s=l2s,
                        ckpt_paths=np.array(ckpt_paths, dtype=object),
                        true_rewards_full=true_rewards_full,
                        state_index_map=state_index_map,
                        cfg=vars(cfg))
    print("Saved results to", out_full)

if __name__ == "__main__":
    main()