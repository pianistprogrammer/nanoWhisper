from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_env() -> None:
    import os
    # Search in current directory and its parent directories for a .env file
    curr = Path.cwd()
    for parent in [curr] + list(curr.parents):
        env_path = parent / ".env"
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'\"")
                        if key and key not in os.environ:
                            os.environ[key] = val
                break
            except Exception:
                pass


def pick_device(requested: str | None = None) -> torch.device:
    load_env()
    import os

    # Priority: Command line arg -> Environment variable -> Auto-detection
    device_str = requested or os.environ.get("DEVICE")
    if device_str:
        device = torch.device(device_str.lower().strip())
        if device.type == "mps":
            _assert_mps_usable()
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        _assert_mps_usable()
        return torch.device("mps")
    return torch.device("cpu")


def _assert_mps_usable() -> None:
    if not torch.backends.mps.is_built():
        raise RuntimeError("This PyTorch build does not include MPS support.")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but PyTorch does not report an available Apple GPU. "
            "Check your PyTorch/macOS compatibility before training with --device mps."
        )
    try:
        torch.empty(1, device="mps")
    except RuntimeError as exc:
        raise RuntimeError(f"MPS was requested, but PyTorch could not create an MPS tensor: {exc}") from exc
