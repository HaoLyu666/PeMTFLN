from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import get_default_args, seed_everything
from .data import HighSimDataset
from .model import Encoder, Predictor


def load_encoder(checkpoint_path: str | Path, args: dict) -> Encoder:
    model = Encoder(args).to(args["device"])
    state_dict = torch.load(checkpoint_path, map_location=args["device"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def masked_rmse_sum(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = torch.pow(pred - target, 2) * mask
    values = torch.pow(torch.sum(error, dim=0), 0.5)
    counts = torch.pow(torch.sum(mask, dim=0), 0.5)
    return values, counts


def masked_mape_sum(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    error = torch.abs(pred - target) / (torch.abs(target) + 1e-8) * 100
    error = torch.where(torch.isnan(error), torch.zeros_like(error), error)
    values = torch.sum(error * mask, dim=0)
    counts = torch.sum(mask, dim=0)
    return values, counts


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    split: str = "test_data",
    batch_size: int | None = None,
    device: str | torch.device | None = None,
    max_samples: int | None = None,
    seed: int = 72,
) -> tuple[pd.DataFrame, dict[str, float]]:
    seed_everything(seed)
    args = get_default_args(device=device)
    if batch_size is not None:
        args["batch_size"] = batch_size

    dataset = HighSimDataset(data_path, split, args["in_length"], args["out_length"], max_samples=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_worker"],
        pin_memory=args["device"].type == "cuda",
        drop_last=False,
    )

    encoder = load_encoder(checkpoint_path, args)
    predictor = Predictor(args)
    device_obj = args["device"]

    rmse_values = torch.zeros(args["veh_num"], args["out_length"], args["out_dim"], device=device_obj)
    rmse_counts = torch.zeros_like(rmse_values) + 1e-8
    mape_values = torch.zeros_like(rmse_values)
    mape_counts = torch.zeros_like(rmse_values) + 1e-8

    with torch.no_grad():
        for hist, fut, nextv in dataloader:
            hist = hist.to(device_obj)
            fut = fut.to(device_obj)
            nextv = nextv.to(device_obj)
            params, _, _ = encoder(hist)
            initial_state = hist[:, :, -1, 0:2]
            equilibrium_history = hist[:, :, :, 0:2]
            predictions = predictor.forward(params, nextv, initial_state, equilibrium_history)
            mask = torch.ones_like(fut[:, :, :, :2], device=device_obj)

            rmse_batch, rmse_count = masked_rmse_sum(predictions[:, :, :, :2], fut[:, :, :, :2], mask)
            mape_batch, mape_count = masked_mape_sum(predictions[:, :, :, :2], fut[:, :, :, :2], mask)
            rmse_values += rmse_batch
            rmse_counts += rmse_count
            mape_values += mape_batch
            mape_counts += mape_count

    rmse = rmse_values / rmse_counts
    mape = mape_values / mape_counts
    by_step = pd.DataFrame(
        {
            "step": np.arange(1, args["out_length"] + 1),
            "rmse_gap": torch.mean(rmse[:, :, 0], dim=0).cpu().numpy(),
            "rmse_speed": torch.mean(rmse[:, :, 1], dim=0).cpu().numpy(),
            "mape_gap": torch.mean(mape[:, :, 0], dim=0).cpu().numpy(),
            "mape_speed": torch.mean(mape[:, :, 1], dim=0).cpu().numpy(),
        }
    )
    summary = {
        "rmse_gap": float(by_step["rmse_gap"].mean()),
        "rmse_speed": float(by_step["rmse_speed"].mean()),
        "mape_gap": float(by_step["mape_gap"].mean()),
        "mape_speed": float(by_step["mape_speed"].mean()),
        "samples": float(len(dataset)),
    }
    return by_step, summary
