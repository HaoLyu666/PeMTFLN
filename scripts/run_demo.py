from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pemtfln.evaluation import evaluate_checkpoint


def main() -> None:
    output_path = REPO_ROOT / "results" / "demo_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_step, summary = evaluate_checkpoint(
        checkpoint_path=REPO_ROOT / "checkpoints" / "pemtfln_highsim_epoch20.tar",
        data_path=REPO_ROOT / "data" / "sample_highsim.npz",
        split="test_data",
        batch_size=16,
        device="cpu",
        max_samples=32,
    )
    by_step.to_csv(output_path, index=False)
    print(f"Demo complete. Metrics saved to {output_path}")
    print(
        "Sample summary: "
        f"RMSE gap={summary['rmse_gap']:.4f}, "
        f"RMSE speed={summary['rmse_speed']:.4f}"
    )


if __name__ == "__main__":
    main()
