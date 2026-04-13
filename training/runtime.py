from __future__ import annotations

import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(accelerator: str, devices: int | str) -> torch.device:
    if devices not in (1, "1", "auto"):
        raise ValueError(
            "Pure PyTorch training currently supports a single device only. Use --devices 1."
        )

    accelerator = str(accelerator).lower()
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return torch.device("cuda")
    if accelerator == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if accelerator == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    raise ValueError(f"Unsupported accelerator: {accelerator}")


def resolve_amp(
    device: torch.device,
    precision: str,
) -> tuple[bool, torch.dtype | None, bool]:
    precision = precision.lower()

    if device.type == "cuda":
        if precision in {"bf16", "bf16-true", "bf16-mixed"}:
            return True, torch.bfloat16, False
        if precision in {"16", "16-true", "16-mixed", "fp16", "fp16-mixed"}:
            return True, torch.float16, True

    return False, None, False


def autocast_context(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
):
    if not amp_enabled or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, tuple):
        return tuple(move_to_device(value, device) for value in obj)
    if isinstance(obj, list):
        return [move_to_device(value, device) for value in obj]
    return obj


def to_scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_runtime(total_steps: int, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        estimated_stepping_batches=total_steps,
        callback_metrics={},
        sanity_checking=False,
        is_global_zero=True,
        default_root_dir=str(run_dir),
    )


def drain_model_metrics(model) -> dict[str, Any]:
    if hasattr(model, "consume_logged_metrics"):
        return model.consume_logged_metrics()
    return {}


def format_metrics(metrics: dict[str, Any], names: list[str]) -> str:
    parts = []
    for name in names:
        value = to_scalar(metrics.get(name))
        if value is not None:
            parts.append(f"{name}={value:.4f}")
    return " | ".join(parts)

