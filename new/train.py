import jax
import jax.numpy as jnp
import optax
from typing import Tuple, Callable, Optional, List, Dict, Any
from new.gridworld import Transition, Reward, Trajectory

Vector = jnp.ndarray
Output = jnp.ndarray
Dataset = Tuple[jnp.ndarray, jnp.ndarray]

default_device = None

# Helpers
# ---------------------------
def transition_to_vector(t: Transition) -> Vector:
    s1, a, s2 = t
    s1_arr = jnp.asarray(s1).reshape(-1)
    s2_arr = jnp.asarray(s2).reshape(-1)
    a_arr = jnp.asarray([a])
    vector = jnp.concatenate((s1_arr, a_arr, s2_arr))
    return vector

def build_dataset_from_trajectories(trajectories: List[Trajectory], reward: Reward) -> Dataset:
    Xs = []
    Ys = []
    for traj in trajectories:
        n = len(traj)
        for i in range(n - 1):
            t = (traj[i][0], traj[i][1], traj[i + 1][0])
            Xs.append(transition_to_vector(t))
            Ys.append(jnp.asarray(reward(t), dtype=jnp.float32))
    if len(Xs) == 0:
        return jnp.zeros((0, 9), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)
    X = jnp.vstack(Xs).astype(jnp.float32)
    y = jnp.array(Ys, dtype=jnp.float32)
    return X, y


# SmallRewardNet
# ---------------------------
class SmallRewardNet:
    """Small MLP implemented with pure JAX arrays (no Flax dependency).

    - params: list of (W, b) tuples for each Linear layer
    - activation: ReLU between layers
    - __call__(x) applies the network using current params
    """

    def __init__(self, input_dim: int = 9, hidden: List[int] = [32], seed: Optional[int] = None):
        if seed is None:
            seed = 0
        key = jax.random.PRNGKey(seed)
        keys = jax.random.split(key, num=len(hidden) + 1)

        self.layer_sizes = [input_dim] + list(hidden) + [1]
        # params will be a list of dicts: {"W":..., "b":...}
        params = []
        for i in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[i]
            out_dim = self.layer_sizes[i + 1]
            k = keys[i]
            glorot_lim = jnp.sqrt(6.0 / (in_dim + out_dim))
            W = jax.random.uniform(k, (in_dim, out_dim), minval=-glorot_lim, maxval=glorot_lim, dtype=jnp.float32)
            b = jnp.zeros((out_dim,), dtype=jnp.float32)
            params.append({"W": W, "b": b})
        self.params: List[Dict[str, jnp.ndarray]] = params

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the network using internal params.

        x: shape (batch, input_dim) or (input_dim,)
        returns: shape (batch,) or scalar
        """
        # Ensure 2D
        x_arr = jnp.asarray(x, dtype=jnp.float32)
        was_1d = (x_arr.ndim == 1)
        if was_1d:
            x_arr = x_arr[None, :]

        out = x_arr
        for i, layer in enumerate(self.params):
            W = layer["W"]
            b = layer["b"]
            out = jnp.dot(out, W) + b
            # Apply ReLU for all but last layer
            if i < len(self.params) - 1:
                out = jax.nn.relu(out)
        # squeeze last dim
        out = out.squeeze(-1)
        if was_1d:
            return out[0]
        return out

    def get_params_pytree(self) -> List[Dict[str, jnp.ndarray]]:
        return self.params

    def set_params_pytree(self, params_pytree: List[Dict[str, jnp.ndarray]]):
        self.params = params_pytree

# Training utilities (JAX + optax)
# ---------------------------
LossFunction = Callable[[Output, list[Tuple[SmallRewardNet, Output]]], Tuple[jnp.ndarray, Optional[dict]]]

def train_reward_net(net: SmallRewardNet, dataset: Dataset, epochs: int = 50, batch_size: int = 64,
                     lr: float = 1e-3, reg: float = 1e-5, seed: Optional[int] = None) -> None:
    """Train a single net using optax Adam on the provided dataset (pure JAX)."""
    X, y = dataset
    if seed is None:
        seed = 0
    key = jax.random.PRNGKey(seed)

    params = net.get_params_pytree()
    # Use optax optimizer on the pytree
    optimizer = optax.adam(learning_rate=lr, eps=1e-8)
    opt_state = optimizer.init(params)

    num = X.shape[0]
    steps_per_epoch = max(1, (num + batch_size - 1) // batch_size)

    @jax.jit
    def loss_and_grads(p, xb, yb):
        # temporarily set net params to p for forward
        def forward(p_local, xb_local):
            out = xb_local
            for i, layer in enumerate(p_local):
                W = layer["W"]
                b = layer["b"]
                out = jnp.dot(out, W) + b
                if i < len(p_local) - 1:
                    out = jax.nn.relu(out)
            return out.squeeze(-1)

        preds = forward(p, xb)
        loss = jnp.mean((preds - yb) ** 2)
        l2 = 0.0
        for layer in p:
            l2 = l2 + jnp.sum(layer["W"] ** 2)
        loss = loss + reg * l2
        grads = jax.grad(lambda pp: jnp.mean((forward(pp, xb) - yb) ** 2) + reg * sum([jnp.sum(l["W"] ** 2) for l in pp]))(p)
        return loss, grads

    for ep in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num)
        tot = 0.0
        n_batches = 0
        for i in range(0, num, batch_size):
            batch_idx = perm[i:i + batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]
            loss, grads = loss_and_grads(params, xb, yb)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            tot += float(loss)
            n_batches += 1
        net.set_params_pytree(params)
        if n_batches > 0 and ep % max(1, epochs // 5) == 0:
            print(f"[ep {ep}/{epochs}] loss={tot / n_batches:.8f}")

def train_ensemble(nets: List[SmallRewardNet], dataset: Dataset,
                   loss_fn: LossFunction, epochs: int = 50, batch_size: int = 64,
                   lr: float = 1e-3, reg: float = 1e-5, seed: Optional[int] = None) -> dict:
    """Train an ensemble of SmallRewardNet instances using optax Adam.

    - nets: list of SmallRewardNet (each with internal params)
    - loss_fn: callable that takes (y_batch, outputs_list) and returns (loss_scalar, trackers)
      where outputs_list is [(net_obj, predictions_jnp), ...]
    """
    X, y = dataset
    if seed is None:
        seed = 0
    key = jax.random.PRNGKey(seed)

    num = X.shape[0]
    if num == 0:
        return {}

    # Build params pytree: a list of params dicts (one per net)
    params_list = [net.get_params_pytree() for net in nets]
    optimizer = optax.adam(learning_rate=lr, eps=1e-8)
    opt_state = optimizer.init(params_list)

    history: Dict[str, List[float]] = {}

    # loss + grads function
    def compute_loss_and_grads(params_pytree, xb, yb):
        outputs = []
        for i, p in enumerate(params_pytree):
            # forward using p
            out = xb
            for li, layer in enumerate(p):
                W = layer["W"]
                b = layer["b"]
                out = jnp.dot(out, W) + b
                if li < len(p) - 1:
                    out = jax.nn.relu(out)
            preds = out.squeeze(-1)
            outputs.append((nets[i], preds))

        loss_value, trackers = loss_fn(yb, outputs)
        return loss_value, trackers

    def loss_for_grad(pytree, xb, yb):
        loss_val, _ = compute_loss_and_grads(pytree, xb, yb)
        # add L2 regularisation
        l2 = 0.0
        for p in pytree:
            for layer in p:
                l2 = l2 + jnp.sum(layer["W"] ** 2)
        loss_with_reg = loss_val + reg * l2
        return loss_with_reg

    grad_fn = jax.grad(loss_for_grad)

    # Training loop
    for ep in range(1, epochs + 1):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, num)
        tot = 0.0
        n_batches = 0

        for start in range(0, num, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb = X[batch_idx]
            yb = y[batch_idx]

            # compute grads
            grads = grad_fn(params_list, xb, yb)

            updates, opt_state = optimizer.update(grads, opt_state, params_list)
            params_list = optax.apply_updates(params_list, updates)

            # compute loss and trackers for logging (use updated params for accurate trackers)
            loss_val, trackers = compute_loss_and_grads(params_list, xb, yb)
            tot += float(loss_val)
            n_batches += 1

            # store trackers
            if trackers is not None:
                for name, val in trackers.items():
                    if name in history:
                        history[name].append(float(val))
                    else:
                        history[name] = [float(val)]

        # write back updated params into net objects
        for i, net in enumerate(nets):
            net.set_params_pytree(params_list[i])

        if n_batches > 0 and ep % max(1, epochs // 5) == 0:
            print(f"[ep {ep}/{epochs}] loss={tot / n_batches:.8f} trackers:")
            for name, vals in history.items():
                print(f"{name} : {vals[-1] :.8f}")

    return history

# Ensemble MSE loss (JAX)
# ---------------------------
def ensemble_MSE():
    def loss(y: Output, outputs: List[Tuple[SmallRewardNet, Output]]) -> Tuple[jnp.ndarray, dict]:
        total_loss = 0.0
        max_loss = 0.0
        for net, out in outputs:
            l = jnp.mean((out - y) ** 2)
            total_loss = total_loss + l
            max_loss = jnp.maximum(max_loss, l)
        return total_loss, {"total_loss": total_loss, "max_loss": max_loss}
    return loss

# reward_from_net for JAX nets
# ---------------------------
def reward_from_net(net: SmallRewardNet):
    """Return a reward function that calls the given SmallRewardNet (using its internal params)."""
    def r(t: Transition) -> float:
        vec = transition_to_vector(t)
        # net returns scalar when given 1D input
        out = net(vec)
        return float(out)
    return r
