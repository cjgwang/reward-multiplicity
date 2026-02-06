import jax
import jax.numpy as jnp
from typing import Tuple, Callable, List
from main.gridworld import DeterministicGridWorld, Policy, Reward
from main.train import SmallRewardNet, transition_to_vector, default_device

Tensor = jnp.ndarray
Array = jnp.ndarray
Output = jnp.ndarray


def build_transition_matrix(env: DeterministicGridWorld):
    N, A = env.num_states, env.num_actions
    P = jnp.zeros((N, A, N))
    for state_idx in range(N):
        for a in range(A):
            next_state = env.next_state_from_idx(state_idx, a)
            P = P.at[state_idx, a, next_state].set(1.0)
    return P


def build_successor_representation(P, policy: Policy, gamma: float = 0.9):
    N, _ = policy.shape
    P_pol = jnp.einsum("sax,sa->sx", P, policy)
    F = jnp.linalg.inv(jnp.eye(N) - gamma * P_pol)
    return F


def compute_reward_matrix(env: DeterministicGridWorld, reward: Reward):
    N, A = env.num_states, env.num_actions
    R = jnp.zeros((N, A))
    for state_idx in range(N):
        state = env.idx_to_state(state_idx)
        for a in range(A):
            next_state = env.next_state(state, a)
            R = R.at[state_idx, a].set(
                reward((state, a, next_state))
            )
    return R


def canonicalise_reward(F, P, R, policy, gamma: float = 0.9):
    N, A = R.shape
    V = F @ jnp.sum(R * policy, axis=1)
    C = R - V[:, None] + gamma * (P @ V)
    return C


def s_norm(Rc: Array):
    norm = jnp.linalg.norm(Rc)
    return Rc if norm == 0 else Rc / norm


def net_s_norm(Rc: Tensor):
    norm = jnp.linalg.norm(Rc)
    return Rc if norm == 0 else Rc / norm


def ensemble_STARc_loss(
    env: DeterministicGridWorld,
    policy: Policy,
    gamma: float = 0.9,
    frozen=[],
    device=default_device,   # kept for signature compatibility
):
    N, A = env.num_states, env.num_actions
    P = build_transition_matrix(env)
    F = build_successor_representation(P, policy, gamma)

    X = []
    for s in range(N):
        for a in range(A):
            ns = env.next_state_from_idx(s, a)
            t = (env.idx_to_state(s), a, env.idx_to_state(ns))
            X.append(transition_to_vector(t))

    policy_t = jnp.asarray(policy, dtype=jnp.float32)
    P_t = jnp.asarray(P, dtype=jnp.float32)
    F_t = jnp.asarray(F, dtype=jnp.float32)
    X_t = jnp.asarray(X, dtype=jnp.float32)

    frozen_rewards = [net(X_t).reshape(N, A) for net in frozen]
    frozen_canonicalised = [
        net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma))
        for R in frozen_rewards
    ]
    m = len(frozen_rewards)

    def loss(y: Output, outputs: List[Tuple[SmallRewardNet, Output]]) -> Tuple[Tensor, dict]:
        total_dist = jnp.array(0.0)
        max_dist = jnp.array(0.0)

        rewards = [net(X_t).reshape(N, A) for net, _ in outputs]
        canonicalised = [
            net_s_norm(canonicalise_reward(F_t, P_t, R, policy_t, gamma))
            for R in rewards
        ]
        n = len(rewards)

        for i in range(n):
            for j in range(i + 1, n + m):
                if j < n:
                    dist = jnp.linalg.norm(canonicalised[i] - canonicalised[j])
                else:
                    dist = jnp.linalg.norm(
                        canonicalised[i] - frozen_canonicalised[j - n]
                    )
                total_dist = total_dist + dist
                max_dist = jnp.maximum(dist, max_dist)

        avg_dist = total_dist / (n * m + n * (n - 1) / 2 + 1e-8)
        trackers = {
            "avg_dist": float(avg_dist),
            "max_dist": float(max_dist),
        }
        return avg_dist, trackers

    return loss
