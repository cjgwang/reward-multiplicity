import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from starc.torch_ops import r_state_to_cnorm_flat


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    opt: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return total_loss / total


def eval_model(model: nn.Module, loader: DataLoader,
               criterion: nn.Module,
               device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            total += imgs.size(0)
    return total_loss / total


def train_model_mse(model: nn.Module, loader: DataLoader,
                    epochs: int, lr: float,
                    device: torch.device,
                    log_prefix: str = "Model") -> nn.Module:
    """Train with MSE regression loss."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        t0 = time.time()
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            running_loss += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        t1 = time.time()
        print(f"[{log_prefix}] Epoch {ep}/{epochs}  MSE={running_loss/n:.6f}  time={t1-t0:.1f}s")
    return model


def train_against_frozen(model: nn.Module,
                          loader: DataLoader,
                          frozen_C_flat: torch.Tensor,
                          images_full_t: torch.Tensor,
                          starc_precomputed: dict,
                          epochs: int,
                          lr: float,
                          alpha: float,
                          gamma: float,
                          device: torch.device,
                          S: int,
                          log_prefix: str = "Model") -> nn.Module:
    """
    Train a model with MSE + STARC loss against a frozen reference.

    Combined loss: MSE - alpha * STARC_distance(model, frozen)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = running_mse = running_starc = 0.0
        n = 0
        t0 = time.time()
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            opt.zero_grad()

            out = model(imgs)
            mse = criterion(out, labels)

            # Full-domain forward (gradient flows through this)
            r_pred_full = model(images_full_t).reshape(S)
            C_pred_flat = r_state_to_cnorm_flat(r_pred_full, starc_precomputed, gamma)
            starc_dist = torch.norm(C_pred_flat - frozen_C_flat.to(device))

            loss = mse - alpha * starc_dist
            loss.backward()
            opt.step()

            running_loss  += float(loss.item())   * imgs.size(0)
            running_mse   += float(mse.item())    * imgs.size(0)
            running_starc += float(starc_dist.item()) * imgs.size(0)
            n += imgs.size(0)

        t1 = time.time()
        print(f"[{log_prefix}] Epoch {ep}/{epochs}  "
              f"loss={running_loss/n:.6f}  mse={running_mse/n:.6f}  "
              f"starc={running_starc/n:.6f}  time={t1-t0:.1f}s")

    return model
