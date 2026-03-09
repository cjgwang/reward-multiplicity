"""
Train an ensemble of reward models where later models are trained with
MSE + STARC loss against the frozen first model.

Usage:
    python scripts/train_ensemble.py --full_domain grid_full_domain.npz
    python scripts/train_ensemble.py --alpha 0.00001 --ensemble_size 3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig
from gridworld import DeterministicGridWorld
from gridworld.policies import uniform_policy
from data import generate_balanced_dataset
from models import CNN, GridDataset
from starc.torch_ops import build_starc_precomputed, r_state_to_cnorm_flat
from training import train_model_mse, train_against_frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--star_pos", type=int, nargs=2, default=[5, 5])
    parser.add_argument("--n_pos", type=int, default=2000)
    parser.add_argument("--n_neg", type=int, default=2000)
    parser.add_argument("--hidden_channels", type=int, default=1)
    parser.add_argument("--full_domain", type=str, default="grid_full_domain.npz",
                        help="Path to .npz built by build_full_domain.py")
    parser.add_argument("--alpha", type=float, default=0.00001)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--ensemble_size", type=int, default=2)
    parser.add_argument("--epochs_first", type=int, default=60)
    parser.add_argument("--epochs_others", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # --- Load full-domain artifacts ---
    data = np.load(args.full_domain)
    images_full = data["images_full"]              # (S, C, H, W)
    true_rewards_full = data["true_rewards_full"].reshape(-1)
    S = images_full.shape[0]
    images_full_t = torch.from_numpy(images_full).to(device)

    # --- Build training dataset ---
    X, y = generate_balanced_dataset(
        star_pos=tuple(args.star_pos), n_pos=args.n_pos, n_neg=args.n_neg,
        H=args.rows, W=args.cols,
    )
    train_loader = DataLoader(GridDataset(X, y), batch_size=args.batch_size, shuffle=True)

    # --- Precompute STARC tensors ---
    env = DeterministicGridWorld(rows=args.rows, cols=args.cols)
    policy_np = uniform_policy(env)
    starc_pre = build_starc_precomputed(env, policy_np, gamma=args.gamma, device=device)

    # --- Train first model (frozen reference) ---
    print("\nTraining first model (will be frozen)...")
    first_model = CNN(in_channels=args.channels, hidden_channels=args.hidden_channels,
                      H=args.rows, W=args.cols).to(device)
    first_model = train_model_mse(first_model, train_loader,
                                   epochs=args.epochs_first, lr=args.lr,
                                   device=device, log_prefix="First model")

    first_model.eval()
    with torch.no_grad():
        r_frozen = first_model(images_full_t).reshape(S)
    C_frozen_flat = r_state_to_cnorm_flat(r_frozen, starc_pre, args.gamma).detach()
    print(f"Frozen model C shape: {C_frozen_flat.shape}")
    torch.save({"model_state_dict": first_model.state_dict()}, "ensemble_model_0.pth")

    # --- Train remaining ensemble members against frozen ---
    ensemble_models = [first_model]
    for m_idx in range(1, args.ensemble_size):
        print(f"\nTraining ensemble member {m_idx} (alpha={args.alpha})...")
        model = CNN(in_channels=args.channels, hidden_channels=args.hidden_channels,
                    H=args.rows, W=args.cols).to(device)
        model = train_against_frozen(
            model, train_loader,
            frozen_C_flat=C_frozen_flat,
            images_full_t=images_full_t,
            starc_precomputed=starc_pre,
            epochs=args.epochs_others, lr=args.lr,
            alpha=args.alpha, gamma=args.gamma,
            device=device, S=S,
            log_prefix=f"Member {m_idx}",
        )
        ensemble_models.append(model)
        torch.save({"model_state_dict": model.state_dict()}, f"ensemble_model_{m_idx}.pth")

    print(f"\nDone. Saved {args.ensemble_size} model checkpoints: ensemble_model_0.pth ... ensemble_model_{args.ensemble_size-1}.pth")


if __name__ == "__main__":
    main()
