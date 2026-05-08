from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pemtfln.evaluation import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PeMTFLN checkpoint on HIGH-SIM data.")
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "sample_highsim.npz")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "checkpoints" / "pemtfln_highsim_epoch20.tar")
    parser.add_argument("--split", default="test_data", choices=["train_data", "val_data", "test_data"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Example: cpu, cuda:0. Defaults to CUDA when available.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "evaluation_sample.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    by_step, summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )
    by_step.to_csv(args.output, index=False)
    print(f"Saved per-step metrics to {args.output}")
    print(
        "Summary: "
        f"RMSE gap={summary['rmse_gap']:.4f}, "
        f"RMSE speed={summary['rmse_speed']:.4f}, "
        f"MAPE gap={summary['mape_gap']:.4f}, "
        f"MAPE speed={summary['mape_speed']:.4f}, "
        f"samples={int(summary['samples'])}"
    )


if __name__ == "__main__":
    main()
