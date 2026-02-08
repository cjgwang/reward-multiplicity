import jax
import jax.numpy as jnp
from new.gridworld import DeterministicGridWorld, star_reward, uniform_policy, sample_trajectory
from new.train import build_dataset_from_trajectories, SmallRewardNet, train_ensemble, ensemble_MSE
from new.starc import ensemble_STARc_loss
from new.render import render_cartesian_gridworld


def main():
    # PRNG key
    key = jax.random.PRNGKey(234235)

    # Helper for small seeds (int32)
    def get_seed():
        nonlocal key
        key, sub = jax.random.split(key)
        # int32 otherwise overflow
        return int(jax.random.randint(sub, (), 0, 2**31 - 1))

    env = DeterministicGridWorld()
    reward_fn = star_reward(env)
    policy = uniform_policy(env)
    print("Initialised seed. Now building trajectories.")
    trajectories = []
    for i in range(5):
        env.goal_coord = jnp.array((i, i))
        for _ in range(5):
            trajectories.append(
                sample_trajectory(
                    env,
                    length=1000,
                    policy=policy,
                    key=jax.random.PRNGKey(get_seed()),
                )
            )

    dataset = build_dataset_from_trajectories(trajectories, reward_fn)
    print(f"Dataset built with {dataset[0].shape[0]} samples.")

    # Initialize ensemble of 3 nets
    nets = [SmallRewardNet() for _ in range(3)]

    # calculate losses
    mse_l = ensemble_MSE()
    starc_l = ensemble_STARc_loss(env, policy)

    def combined_loss(y, outputs):
        l1, t1 = mse_l(y, outputs)
        l2, t2 = starc_l(y, outputs)
        return l1 + 0.1 * l2, {**t1, **t2}

    print("Starting Training...")
    history = train_ensemble(nets, dataset, combined_loss, epochs=20)

    print("Training complete. Rendering environment...")
    render_cartesian_gridworld(env)


if __name__ == "__main__":
    main()
