"""
Build and save the full-domain dataset (all (agent, star) pairs) to a .npz file.

Usage:
    python scripts/build_full_domain.py
    python scripts/build_full_domain.py --rows 7 --cols 7 --out grid_full_domain.npz
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
from gridworld import DeterministicGridWorld
from data import generate_full_enumeration_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--out", type=str, default="grid_full_domain.npz")
    args = parser.parse_args()

    env = DeterministicGridWorld(rows=args.rows, cols=args.cols, start=(0, 0), goal=(args.cols-1, args.rows-1))
    images_full, true_rewards_full, state_index_map = generate_full_enumeration_dataset(env, channels=args.channels)

    chan_mean = images_full.mean(axis=(0, 2, 3))
    chan_std  = images_full.std(axis=(0, 2, 3)) + 1e-8

    np.savez_compressed(
        args.out,
        images_full=images_full,
        true_rewards_full=true_rewards_full,
        state_index_map=state_index_map,
        chan_mean=chan_mean,
        chan_std=chan_std,
    )

    print(f"Saved to {args.out}")
    print(f"  images_full:       {images_full.shape}")
    print(f"  true_rewards_full: {true_rewards_full.shape}")
    print(f"  state_index_map:   {state_index_map.shape}")
    print(f"  chan_mean: {chan_mean}")
    print(f"  chan_std:  {chan_std}")


if __name__ == "__main__":
    main()
