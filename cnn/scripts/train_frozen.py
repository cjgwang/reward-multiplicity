"""
Train a frozen (reference) reward model on the fixed-star dataset and save it.

Usage:
    python scripts/train_frozen.py
    python scripts/train_frozen.py --epochs 60 --hidden_channels 8 --out frozen_model.pth
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from data import generate_balanced_dataset
from models import CNN, GridDataset
from training import train_one_epoch, eval_model
import torch.nn as nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--star_pos", type=int, nargs=2, default=[5, 5], metavar=("ROW", "COL"))
    parser.add_argument("--n_pos", type=int, default=2000)
    parser.add_argument("--n_neg", type=int, default=2000)
    parser.add_argument("--hidden_channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="frozen_model.pth")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        rows=args.rows, cols=args.cols, channels=args.channels,
        star_pos=tuple(args.star_pos),
        n_pos=args.n_pos, n_neg=args.n_neg,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        seed=args.seed, device=args.device, frozen_ckpt_path=args.out,
    )
    device = cfg.get_device()
    print(f"Device: {device}")

    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    X, y = generate_balanced_dataset(
        star_pos=cfg.star_pos, n_pos=cfg.n_pos, n_neg=cfg.n_neg,
        H=cfg.rows, W=cfg.cols,
    )
    N = len(X)
    n_val = max(1, int(0.1 * N))
    train_X, val_X = X[:N - n_val], X[N - n_val:]
    train_y, val_y = y[:N - n_val], y[N - n_val:]

    train_loader = DataLoader(GridDataset(train_X, train_y), batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(GridDataset(val_X,   val_y),   batch_size=cfg.batch_size, shuffle=False)

    model = CNN(in_channels=cfg.channels, hidden_channels=cfg.hidden_channels,
                H=cfg.rows, W=cfg.cols).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.MSELoss()

    print(f"Training on {len(train_X)} samples, validating on {len(val_X)}")
    for ep in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        val_loss   = eval_model(model, val_loader, criterion, device)
        if ep == 1 or ep % 10 == 0 or ep == cfg.epochs:
            print(f"Epoch {ep:03d}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

    torch.save({"model_state_dict": model.state_dict(), "config": vars(cfg)}, cfg.frozen_ckpt_path)
    print(f"Saved to {cfg.frozen_ckpt_path}")


if __name__ == "__main__":
    main()
