from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
    vocab: list[str],
    scheduler: Any | None = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "step": step,
            "best_val_loss": best_val_loss,
            "config": config,
            "vocab": vocab,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
