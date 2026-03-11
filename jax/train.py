import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Callable, Optional, List
from gridword import Transition, Reward, Trajectory

Vector = np.ndarray
Output = np.ndarray
Dataset = Tuple[np.ndarray, np.ndarray]

default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transition_to_vector(t: Transition) -> Vector:
    s1, a, s2 = t
    # Fixed the concatenation logic to work with numpy correctly
    vector = np.concatenate((np.concatenate(s1), [a], np.concatenate(s2)))
    return vector

def build_dataset_from_trajectories(trajectories: List[Trajectory], reward: Reward) -> Dataset:
    Xs, Ys = [], []
    for traj in trajectories:
        n = len(traj)
        for i in range(n-1):
            t = (traj[i][0], traj[i][1], traj[i+1][0])
            Xs.append(transition_to_vector(t))
            Ys.append(reward(t))
    if len(Xs) == 0:
        return np.zeros((0,9), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.vstack(Xs).astype(np.float32)
    y = np.array(Ys, dtype=np.float32)
    return X, y

class SmallRewardNet(nn.Module):
    def __init__(self, input_dim=9, hidden=[32]):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_reward_net(net: SmallRewardNet, dataset: Dataset, device=default_device, epochs=50, batch_size=64, lr=1e-3, reg=1e-5, seed=None) -> None:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    X, y = dataset
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    dataset_obj = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=reg)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        tot = 0.0; n_batches = 0
        for xb, yb in loader:
            pred = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); n_batches += 1
        if n_batches > 0 and ep % max(1, epochs//5) == 0:
            print(f"[ep {ep}/{epochs}] loss={tot / n_batches:.8f}")

LossFunction = Callable[[Output, list[Tuple[SmallRewardNet, Output]]], Tuple[float, Optional[dict]]]

def train_ensemble(nets: List[SmallRewardNet], dataset: Dataset,
                     loss_fn: LossFunction, device=default_device,
                     epochs=50, batch_size=64, lr=1e-3, reg=1e-5, seed=None) -> dict:
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    X, y = dataset
    X_t = torch.from_numpy(X).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)
    dataset_obj = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)
    params = []
    for net in nets:
        params += list(net.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=reg)
    history = {}

    for ep in range(1, epochs+1):
        tot = 0.0; n_batches = 0
        for xb, yb in loader:
            outputs = []
            for net in nets:
                outputs.append((net, net(xb)))
            loss, trackers = loss_fn(yb, outputs)

            if(trackers is not None):
                for name, val in trackers.items():
                    if(name in history):
                        history[name].append(val)
                    else:
                        history[name] = [val]

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            n_batches += 1
        if n_batches > 0 and ep % max(1, epochs//5) == 0:
            print(f"[ep {ep}/{epochs}] loss={tot / n_batches:.8f} trackers:")
            for name, val in history.items():
                print(f"{name} : {val[-1] :.8f}")
    return history

def ensemble_MSE():
    loss_fn = nn.MSELoss()
    def loss(y: Output, outputs: List[Tuple[SmallRewardNet, Output]]) -> Tuple[float, dict]:
        total_loss = 0.0
        max_loss = 0.0
        for net, out in outputs:
            l = loss_fn(out, y)
            total_loss += l
            max_loss = max(max_loss, l)
        return (total_loss, {"total_loss":total_loss.item(), "max_loss":max_loss.item()})
    return loss

def reward_from_net(net, device=default_device):
    def r(t: Transition) -> float:
        net.eval()
        vec = transition_to_vector(t)
        x = torch.from_numpy(vec).unsqueeze(0).float().to(device)
        out = net(x)
        return float(out.detach().cpu().squeeze())
    return r