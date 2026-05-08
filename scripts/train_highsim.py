from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pemtfln.config import get_default_args, seed_everything
from pemtfln.data import HighSimDataset
from pemtfln.losses import DynamicWeightAverage, parameter_distribution_loss
from pemtfln.model import Encoder, Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PeMTFLN on preprocessed HIGH-SIM data.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "sample_highsim.npz")
    parser.add_argument("--epochs", type=int, default=2, help="Use 20 for the paper-scale setting.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda:0. Defaults to CUDA when available.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "training")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    seed_everything()
    args = get_default_args(device=cli.device, epoch=cli.epochs, batch_size=cli.batch_size, train_flag=True)
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = cli.output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    dataset = HighSimDataset(cli.data, "train_data", args["in_length"], args["out_length"], max_samples=cli.max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_worker"],
        pin_memory=args["device"].type == "cuda",
        drop_last=True,
    )

    encoder = Encoder(args).to(args["device"])
    predictor = Predictor(args)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args["gamma"])
    dwa = DynamicWeightAverage(task_num=args["num_task"], device=args["device"])
    train_loss_buffer = torch.zeros(args["num_task"] + 1, args["epoch"])
    history: list[list[float]] = []

    for epoch in range(args["epoch"]):
        encoder.train()
        total_gap = 0.0
        total_speed = 0.0
        total_loss = 0.0
        for hist, fut, nextv in dataloader:
            hist = hist.to(args["device"])
            fut = fut.to(args["device"])
            nextv = nextv.to(args["device"])

            params, mu, log_var = encoder(hist)
            predictions = predictor.forward(params, nextv, hist[:, :, -1, 0:2], hist[:, :, :, 0:2])

            loss_gap = torch.nn.functional.mse_loss(predictions[:, :, :, 0], fut[:, :, :, 0])
            loss_speed = torch.nn.functional.mse_loss(predictions[:, :, :, 1], fut[:, :, :, 1])
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            task_losses = [loss_gap, loss_speed]
            weights = dwa.weights(task_losses, epoch, train_loss_buffer[: args["num_task"], :])
            weighted_loss = sum(loss * weight for loss, weight in zip(task_losses, weights))
            param_loss = parameter_distribution_loss(params)
            param_weight = torch.clamp(0.1 * weighted_loss.detach() / (param_loss.detach() + 1e-8), min=0.05, max=0.2)
            loss = weighted_loss + 0.0025 * kl_divergence + param_weight * param_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10)
            optimizer.step()

            total_gap += loss_gap.item()
            total_speed += loss_speed.item()
            total_loss += loss.item()

        batches = max(1, len(dataloader))
        avg_gap = total_gap / batches
        avg_speed = total_speed / batches
        avg_loss = total_loss / batches
        train_loss_buffer[:, epoch] = torch.tensor([avg_gap, avg_speed, avg_loss])
        history.append([epoch + 1, avg_gap, avg_speed, avg_loss, optimizer.param_groups[0]["lr"]])
        torch.save(encoder.state_dict(), checkpoint_dir / f"epoch{epoch + 1}_e.tar")
        scheduler.step()
        print(
            f"epoch={epoch + 1} "
            f"gap_loss={avg_gap:.6f} speed_loss={avg_speed:.6f} total_loss={avg_loss:.6f}"
        )

    with (cli.output_dir / "train_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "gap_loss", "speed_loss", "total_loss", "learning_rate"])
        writer.writerows(history)


if __name__ == "__main__":
    main()
