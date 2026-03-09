from dataclasses import dataclass, field
import torch


@dataclass
class ExperimentConfig:
    # Grid world
    rows: int = 7
    cols: int = 7
    channels: int = 3
    star_pos: tuple = (5, 5)       # default fixed star / goal position

    # Dataset
    n_pos: int = 2000
    n_neg: int = 2000

    # Model
    hidden_channels: int = 1

    # Training
    batch_size: int = 64
    epochs: int = 200
    lr: float = 5e-3
    seed: int = 42

    # STARC ensemble
    alpha: float = 0.0             # weight for -alpha * STARC (0 = disabled)
    gamma: float = 0.9
    ensemble_size: int = 2
    epochs_first: int = 60
    epochs_others: int = 60

    # I/O
    full_domain_path: str = "grid_full_domain.npz"
    frozen_ckpt_path: str = "frozen_model.pth"
    final_model_path: str = "tinycnn_reg.pth"

    # Device (resolved at runtime)
    device: str = "auto"           # "auto", "cpu", "cuda"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
