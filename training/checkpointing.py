from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.runtime import to_scalar


def resolve_run_dir(log_dir: Path, experiment_name: str, resume_from_checkpoint: Path | None) -> Path:
    if resume_from_checkpoint is not None:
        checkpoint_dir = resume_from_checkpoint.parent
        if checkpoint_dir.name == "checkpoints":
            return checkpoint_dir.parent
        return checkpoint_dir
    return log_dir / experiment_name


def sanitize_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace("._orig_mod", ""): value
        for key, value in state_dict.items()
    }


def cpu_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in sanitize_state_dict(state_dict).items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu()
        else:
            result[key] = value
    return result


def save_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_ap_all: float,
) -> None:
    callback_metrics = {
        key: to_scalar(value)
        for key, value in getattr(model.trainer, "callback_metrics", {}).items()
    }
    callback_metrics = {
        key: value for key, value in callback_metrics.items() if value is not None
    }

    checkpoint = {
        "model_state_dict": cpu_state_dict(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": int(model.global_step),
        "best_val_ap_all": best_val_ap_all,
        "callback_metrics": callback_metrics,
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_training_state(
    checkpoint_path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> tuple[int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        model_state_dict = checkpoint.get("state_dict", checkpoint)

    incompatible = model.load_state_dict(
        sanitize_state_dict(model_state_dict),
        strict=False,
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Resume checkpoint is incompatible.\n"
            f"Missing keys: {incompatible.missing_keys}\n"
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None and checkpoint.get("optimizer_states"):
        optimizer_state = checkpoint["optimizer_states"][0]
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler_state is None and checkpoint.get("lr_schedulers"):
        scheduler_state = checkpoint["lr_schedulers"][0]
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    callback_metrics = {
        key: value
        for key, value in checkpoint.get("callback_metrics", {}).items()
        if to_scalar(value) is not None
    }
    model.trainer.callback_metrics.update(callback_metrics)
    model.global_step = int(checkpoint.get("global_step", 0))

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_ap_all = float(
        checkpoint.get(
            "best_val_ap_all",
            callback_metrics.get("metrics/val_ap_all", float("-inf")),
        )
    )
    return start_epoch, best_val_ap_all

