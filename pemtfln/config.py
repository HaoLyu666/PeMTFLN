from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ARGS: dict[str, Any] = {
    "encoder_size": 64,
    "n_head": 4,
    "in_length": 21,
    "out_length": 20,
    "para_length": 4,
    "f_length": 7,
    "veh_num": 6,
    "out_dim": 2,
    "num_task": 2,
    "num_mc": 0,
    "dropout": 0.05,
    "gamma": 0.7,
    "batch_size": 32,
    "epoch": 20,
    "transformer_layer": 2,
    "learning_rate": 5e-4,
    "num_worker": 0,
    "time_step": 0.1,
    "train_flag": False,
}


def seed_everything(seed: int = 72) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_default_args(device: str | torch.device | None = None, **overrides: Any) -> dict[str, Any]:
    args = deepcopy(DEFAULT_ARGS)
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args["device"] = torch.device(device)
    args.update(overrides)
    return args
