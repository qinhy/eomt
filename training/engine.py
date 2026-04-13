from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from training.runtime import (
    autocast_context,
    drain_model_metrics,
    format_metrics,
    move_to_device,
)


def train_one_epoch(
    model,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger,
    epoch: int,
    log_every_n_steps: int,
) -> None:
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            loss = model.training_step(batch, batch_idx)

        if not torch.isfinite(loss):
            print(
                f"[WARN] Skipping batch {batch_idx + 1}/{len(train_loader)} "
                f"at epoch {epoch + 1}: non-finite loss={loss.detach().item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        stepped = False

        try:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=0.01,
                    error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm):
                    print(
                        f"[WARN] Skipping optimizer step at batch {batch_idx + 1}: "
                        f"non-finite grad_norm={grad_norm.detach().item()}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()
                stepped = True
            else:
                loss.backward()
                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=0.01,
                    error_if_nonfinite=True,
                )
                if not torch.isfinite(grad_norm):
                    print(
                        f"[WARN] Skipping optimizer step at batch {batch_idx + 1}: "
                        f"non-finite grad_norm={grad_norm.detach().item()}"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()
                stepped = True
        except RuntimeError as exc:
            print(
                f"[WARN] RuntimeError during backward/step at batch {batch_idx + 1}: {exc}"
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.update()
            continue

        if stepped:
            scheduler.step()
            model.global_step = int(model.global_step) + 1

        model.on_train_batch_end(loss.detach(), batch, batch_idx)

        step_metrics = drain_model_metrics(model)
        step_metrics["lr/min"] = min(group["lr"] for group in optimizer.param_groups)
        step_metrics["lr/max"] = max(group["lr"] for group in optimizer.param_groups)
        logger.log_metrics("train", epoch + 1, int(model.global_step), step_metrics)

        should_print = (
            batch_idx == 0
            or (batch_idx + 1) % log_every_n_steps == 0
            or batch_idx + 1 == len(train_loader)
        )
        if should_print:
            message = format_metrics(
                step_metrics,
                [
                    "loss_total",
                    "mask",
                    "dice",
                    "cls",
                    "bbox",
                    "giou",
                    "lr/max",
                ],
            )
            prefix = (
                f"Epoch {epoch + 1}/{getattr(model, 'max_epochs', '?')} "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Step {int(model.global_step)}"
            )
            print(f"{prefix} | {message}" if message else prefix)


def validate(
    model,
    val_loader,
    device: torch.device,
    logger,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    model.trainer.callback_metrics.clear()

    with torch.inference_mode():
        for batch_idx, batch in enumerate(val_loader):
            batch = move_to_device(batch, device)
            model.validation_step(batch, batch_idx)

    model.on_validation_epoch_end()
    metrics = drain_model_metrics(model)
    logger.log_metrics("val", epoch + 1, int(model.global_step), metrics)
    model.on_validation_end()
    return metrics

