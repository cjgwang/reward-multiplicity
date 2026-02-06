import torch
import numpy as np

# @title STARc (setup) {"display-mode":"form"}

Tensor = torch.Tensor
Array = np.ndarray

# Builds a transition matrix given a deterministic gridworld environment
def build_transition_matrix(env): # (N, A, N)
    N, A = env.num_states, env.num_actions

    # Create empty matrix for P
    P = np.zeros((N, A, N))

    for state_idx in range(N):
        # Loop over all possible states
        for a in range(A):
            # Loop over all possible actions
            next_state = env.next_state_from_idx(state_idx, a)
            P[state_idx, a, next_state] = 1
    return P

# Builds successor representation given the transition matrix and a gamma value
def build_successor_representation(P, policy: Policy, gamma : float = 0.9): # (N, N)
    N, _ = policy.shape
    P = np.einsum("sax,sa -> sx", build_transition_matrix(env), policy)
    F = np.linalg.inv(np.eye(N) - gamma * P)
    return F

# Builds the reward matrix in the deterministic case
def compute_reward_matrix(env, reward: Reward): # (N, A)
    N, A = env.num_states, env.num_actions
    R = np.zeros((N, A))
    for state_idx in range(N):
        state = env.idx_to_state(state_idx)
        for a in range(A):
            next_state = env.next_state(state, a)
            R[state_idx, a] = reward((state, a, next_state))
    return R

# Builds canonicalised reward matrix assuming fixed star
def canonicalise_reward(F: Tensor, P: Tensor, R: Tensor, policy: Tensor, gamma : float = 0.9):
    N, A = R.shape
    V = F @ (R * policy).sum(dim=1)
    C = R - V.unsqueeze(1).expand(N, A) + gamma * P @ V
    return C

def canonicalise_reward(F: Array, P: Array, R: Array, policy: Policy, gamma : float = 0.9):
    N, A = R.shape
    V = F @ (R * policy).sum(axis=1)
    C = R - V.reshape(N, 1) + gamma * P @ V
    return C

# Normalisation step as in STARc paper (L2)
def s_norm(Rc: Array):
    norm = np.linalg.norm(Rc)
    return Rc if norm == 0 else Rc / norm
def net_s_norm(Rc: Tensor):
    norm = torch.norm(Rc)
    return Rc if norm == 0 else Rc / norm

# Calculates STARc distance using L2 norm
def starc_distance(env, policy: Policy, r1: Reward, r2: Reward, gamma : float = 0.9):
    P = build_transition_matrix(env)
    F = build_successor_representation(P, policy, gamma)
    R1 = compute_reward_matrix(env, r1)
    R2 = compute_reward_matrix(env, r2)
    C1 = canonicalise_reward(F, P, R1, policy, gamma)
    C2 = canonicalise_reward(F, P, R2, policy, gamma)
    return np.linalg.norm(s_norm(C1) - s_norm(C2))

# Loss function for ensemble that returns average starc distance
def ensemble_STARc_loss(env, policy: Policy, gamma: float = 0.9, frozen=[], device=default_device):
    # Precalculate fixed matrices P, F, and X (for quick evaluations from net to reward matrix)
    N, A = env.num_states, env.num_actions
    P = build_transition_matrix(env)
    F = build_successor_representation(P, policy, gamma)
    X = []
    for s in range(N):
        for a in range(A):
            ns = env.next_state_from_idx(s, a)
            t = (env.idx_to_state(s), a, env.idx_to_state(ns))
            X.append(transition_to_vector(t))

    # Transfer matrices to device as pytorch tensors
    policy_t = torch.from_numpy(policy).float().to(device)
    P_t = torch.from_numpy(P).float().to(device)
    F_t = torch.from_numpy(F).float().to(device)
    X_t = torch.tensor(np.array(X)).float().to(device)

    # Calculate canonicalisations of frozen rewards in advance
    frozen_rewards = [net(X_t).detach().reshape(N, A) for net in frozen]
    frozen_canonicalised = [net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma)) for R in frozen_rewards]
    m = len(frozen_rewards)

    def loss(y: Output, outputs: list[Tuple[SmallRewardNet, Output]]) -> Tuple[float, dict]:
        total_dist = torch.tensor(0.0, device=device)
        max_dist = torch.tensor(0.0, device=device)
        rewards = [net(X_t).reshape(N, A) for net, _ in outputs]
        canonicalised = [net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma)) for R in rewards]
        n = len(rewards)
        for i in range(n):
            for j in range(i+1, n + m):
                if(j < n):
                    dist = torch.norm(canonicalised[i] - canonicalised[j]) # Active rewards
                else:
                    dist = torch.norm(canonicalised[i] - frozen_canonicalised[j - n]) # Frozen rewards
                total_dist += dist
                max_dist = torch.max(dist, max_dist)

        avg_dist = total_dist / (n*m + n*(n-1)/2)
        trackers = {}
        trackers["avg_dist"] = avg_dist.detach().item()
        trackers["max_dist"] = max_dist.detach().item()

        return avg_dist, trackers
    return loss

# Merges two loss functions for ensembles using f(l1, l2) and renames any trackers to {t_i}_... for i=1,2
def combine_ensemble_losses(f: Callable[[float, float], float], l1: LossFunction, l2: LossFunction, n1="loss1", n2="loss2"):
    def loss(y: Output, outputs: list[Tuple[SmallRewardNet, Output]]):
        loss1, t1 = l1(y, outputs)
        loss2, t2 = l2(y, outputs)
        loss = f(loss1, loss2)

        trackers = {}
        for name, val in t1.items():
            trackers[n1 + "_" + name] = val
        for name, val in t2.items():
            trackers[n2 + "_" + name] = val

        return loss, trackers
    return loss