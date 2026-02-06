import numpy as np
import torch
from gridworld import DeterministicGridWorld, star_reward, uniform_policy, sample_trajectory
from train import build_dataset_from_trajectories, SmallRewardNet, train_ensemble, ensemble_MSE
from starc import ensemble_STARc_loss
from render import render_cartesian_gridworld

def main():
    rng = np.random.RandomState(234235)
    get_seed = lambda : (rng.randint(0, 2**32))
    torch.manual_seed(get_seed())

    env = DeterministicGridWorld()
    reward_fn = star_reward(env)
    policy = uniform_policy(env)
    
    trajectories = []
    for i in range(5):
        env.goal_coord = np.array((i, i))
        trajectories += [sample_trajectory(env, 1000, policy, seed=get_seed()) for _ in range(5)]
    
    dataset = build_dataset_from_trajectories(trajectories, reward_fn)
    print(f"Dataset built with {len(dataset[0])} samples.")

    # Initialize Ensemble
    nets = [SmallRewardNet() for _ in range(3)]
    
    # Combined Loss example
    mse_l = ensemble_MSE()
    starc_l = ensemble_STARc_loss(env, policy)
    
    def combined_loss(y, outputs):
        l1, t1 = mse_l(y, outputs)
        l2, t2 = starc_l(y, outputs)
        # Example weight: 1.0 * MSE + 0.1 * STARc
        return l1 + 0.1 * l2, {**t1, **t2}

    print("Starting Training...")
    history = train_ensemble(nets, dataset, combined_loss, epochs=20)
    
    print("Training complete. Rendering environment...")
    render_cartesian_gridworld(env)

if __name__ == "__main__":
    main()