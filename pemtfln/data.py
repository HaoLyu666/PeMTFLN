from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class HighSimDataset(Dataset):
    """Dataset wrapper for preprocessed HIGH-SIM platoon arrays.

    Expected npz keys are ``train_data``, ``val_data`` and ``test_data``.
    Feature indices follow the processed data used in the paper code:

    0 platoon id, 1 gap, 2 speed, 3 speed difference, 4 acceleration,
    5 PET, 6 SSDD, 7 vehicle length, 8 preceding vehicle length,
    9 preceding vehicle speed.
    """

    def __init__(
        self,
        data_path: str | Path,
        split: str,
        in_length: int,
        out_length: int,
        max_samples: int | None = None,
    ) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        data = np.load(data_path)
        if split not in data.files:
            raise KeyError(f"Split {split!r} not found in {data_path}. Available keys: {data.files}")

        values = data[split]
        if max_samples is not None:
            values = values[:max_samples]

        self.hist = values[:, :, :in_length, [1, 2, 3, 4, 6, 7, 8]]
        self.fut = values[:, :, in_length : in_length + out_length, [1, 2, 6]]
        self.nextv = values[:, :1, in_length - 1 : in_length + out_length, 9:10]

    def __len__(self) -> int:
        return len(self.hist)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.hist[idx], dtype=torch.float32),
            torch.tensor(self.fut[idx], dtype=torch.float32),
            torch.tensor(self.nextv[idx], dtype=torch.float32),
        )
