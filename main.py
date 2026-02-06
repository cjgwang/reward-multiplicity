rng = np.random.RandomState(234235)
seed = lambda : (rng.randint(0, 2**32))
torch.manual_seed(seed())

env = DeterministicGridWorld()
reward = star_reward(env)
policy = uniform_policy(env)
trajectories = []
for i in range(5):
    env.goal_coord = np.array((i, i))
    trajectories += [sample_trajectory(env, 1000, policy, seed=seed()) for _ in range(5)]
dataset = build_dataset_from_trajectories(trajectories, reward)

nets = [SmallRewardNet(hidden=[32,32,32,32]).to(default_device) for _ in range(200)]
outs = [reward_from_net(net) for net in nets]

mse = ensemble_MSE()
starc = ensemble_STARc_loss(env, policy)
f = lambda x,y : x - 1e-3 * y
loss = combine_ensemble_losses(f, mse, starc, "mse", "starc")

history = train_ensemble(nets, dataset, loss, seed=seed(),
                            epochs=100, batch_size=2048, lr=1e-3, reg=1e-4)
plot_ensemble_history(history)

i = 0
for out in outs:
    i += 1
    print(i, starc_distance(env, policy, reward, out))

print(starc_distance(env, policy, reward, outs[21]))


for s in range(env.num_positions):
    star = env.idx_to_coord(s)
    visualize_reward_quadrant(env, outs[6], star)