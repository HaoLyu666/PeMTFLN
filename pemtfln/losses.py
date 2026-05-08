from __future__ import annotations

import torch
import torch.nn.functional as F


class DynamicWeightAverage:
    """Dynamic Weight Average for multi-task trajectory losses."""

    def __init__(self, task_num: int, device: torch.device, temperature: float = 2.0) -> None:
        self.task_num = task_num
        self.device = device
        self.temperature = temperature

    def weights(self, losses: list[torch.Tensor], epoch: int, train_loss_buffer: torch.Tensor) -> list[float]:
        if epoch > 1:
            ratios = train_loss_buffer[:, epoch - 1] / train_loss_buffer[:, epoch - 2]
            batch_weight = self.task_num * F.softmax(ratios.to(self.device) / self.temperature, dim=-1)
        else:
            batch_weight = torch.ones(len(losses), device=self.device)
        return batch_weight.detach().tolist()


def parameter_distribution_loss(params: torch.Tensor) -> torch.Tensor:
    p_min, p_max = 0.02, 5.0
    penalties = [torch.relu(p_min - params[..., i]) + torch.relu(params[..., i] - p_max) for i in range(3)]
    return sum(torch.mean(penalty) for penalty in penalties)
