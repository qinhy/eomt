import argparse
import csv
import json
import os
import random
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("EOMT_DISABLE_LIGHTNING", "1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from datasets.coco_instance import COCOInstance
from models.eomt_dinov3 import EoMT
from training.mask_classification_instance import MaskClassificationInstance



def _default_encoder_repo() -> Path:
    return (REPO_ROOT / "../dinov3").resolve()


def _default_encoder_weights() -> Path:
    return (
        REPO_ROOT / "../BitNetCNN/data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ).resolve()


def _parse_devices(value: str):
    return int(value) if value.isdigit() else value


def _default_precision(accelerator: str) -> str:
    if accelerator == "cpu":
        return "32-true"
    if accelerator == "auto" and not torch.cuda.is_available():
        return "32-true"
    return "bf16-true"


def _validate_paths(args: argparse.Namespace) -> None:
    if not args.data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {args.data_path}"
        )
    if not args.encoder_repo.exists():
        raise FileNotFoundError(
            f"DINOv3 repo path does not exist: {args.encoder_repo}"
        )
    if not args.encoder_weights.exists():
        raise FileNotFoundError(
            f"DINOv3 checkpoint does not exist: {args.encoder_weights}"
        )
    if args.ckpt_path is not None and not args.ckpt_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint does not exist: {args.ckpt_path}"
        )
    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.exists()
    ):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone pure PyTorch COCO instance training script for EoMT DINOv3."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.1, 2.0))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=2)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument(
        "--experiment-name",
        default="coco_instance_eomt_large_640_dinov3_pure",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder-repo", type=Path, default=_default_encoder_repo())
    parser.add_argument(
        "--encoder-model",
        default="dinov3_vits16",
    )
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=_default_encoder_weights(),
    )
    parser.add_argument("--num-q", type=int, default=200)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--llrd", type=float, default=0.8)
    parser.add_argument("--lr-mult", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--warmup-steps", type=int, nargs=2, default=(2000, 3000))
    parser.add_argument("--num-points", type=int, default=12544)
    parser.add_argument("--oversample-ratio", type=float, default=3.0)
    parser.add_argument("--importance-sample-ratio", type=float, default=0.75)
    parser.add_argument("--no-object-coefficient", type=float, default=0.1)
    parser.add_argument("--mask-coefficient", type=float, default=5.0)
    parser.add_argument("--dice-coefficient", type=float, default=5.0)
    parser.add_argument("--class-coefficient", type=float, default=2.0)
    parser.add_argument("--bbox-l1-coefficient", type=float, default=5.0)
    parser.add_argument("--bbox-giou-coefficient", type=float, default=2.0)
    parser.add_argument("--mask-thresh", type=float, default=0.8)
    parser.add_argument("--overlap-thresh", type=float, default=0.8)
    parser.add_argument("--eval-top-k-instances", type=int, default=100)
    parser.add_argument(
        "--attn-mask-annealing-start-steps",
        type=int,
        nargs="+",
        default=(14782, 29564, 44346, 59128),
    )
    parser.add_argument(
        "--attn-mask-annealing-end-steps",
        type=int,
        nargs="+",
        default=(29564, 44346, 59128, 73910),
    )
    parser.add_argument(
        "--delta-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--llrd-l2-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--color-jitter-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--check-empty-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--masked-attn-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--attn-mask-annealing-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--load-ckpt-class-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    if len(args.attn_mask_annealing_start_steps) != args.num_blocks:
        raise ValueError(
            "--attn-mask-annealing-start-steps must match --num-blocks"
        )
    if len(args.attn_mask_annealing_end_steps) != args.num_blocks:
        raise ValueError(
            "--attn-mask-annealing-end-steps must match --num-blocks"
        )

    args.devices = _parse_devices(args.devices)
    args.img_size = tuple(args.img_size)
    args.scale_range = tuple(args.scale_range)
    args.warmup_steps = tuple(args.warmup_steps)
    args.data_path = args.data_path.resolve()
    args.log_dir = args.log_dir.resolve()
    args.encoder_repo = args.encoder_repo.resolve()
    args.encoder_weights = args.encoder_weights.resolve()
    if args.ckpt_path is not None:
        args.ckpt_path = args.ckpt_path.resolve()
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve()
    if args.precision is None:
        args.precision = _default_precision(args.accelerator)

    _validate_paths(args)
    return args


def build_datamodule(args: argparse.Namespace) -> COCOInstance:
    return COCOInstance(
        path=str(args.data_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_classes=args.num_classes,
        color_jitter_enabled=args.color_jitter_enabled,
        scale_range=args.scale_range,
        check_empty_targets=args.check_empty_targets,
    )


def build_model(args: argparse.Namespace) -> MaskClassificationInstance:
    network = EoMT(
        encoder_weights=str(args.encoder_weights),
        num_classes=args.num_classes,
        num_q=args.num_q,
        num_blocks=args.num_blocks,
        masked_attn_enabled=args.masked_attn_enabled,
        bbox_head_enabled=True,
        encoder_repo=str(args.encoder_repo),
        encoder_model=args.encoder_model,
        fsrcnnx2=True,
        precision=args.precision,
    )
    return MaskClassificationInstance(
        network=network,
        img_size=args.img_size,
        num_classes=args.num_classes,
        attn_mask_annealing_enabled=args.attn_mask_annealing_enabled,
        attn_mask_annealing_start_steps=list(args.attn_mask_annealing_start_steps),
        attn_mask_annealing_end_steps=list(args.attn_mask_annealing_end_steps),
        lr=args.lr,
        llrd=args.llrd,
        llrd_l2_enabled=args.llrd_l2_enabled,
        lr_mult=args.lr_mult,
        weight_decay=args.weight_decay,
        num_points=args.num_points,
        oversample_ratio=args.oversample_ratio,
        importance_sample_ratio=args.importance_sample_ratio,
        poly_power=args.poly_power,
        warmup_steps=list(args.warmup_steps),
        no_object_coefficient=args.no_object_coefficient,
        mask_coefficient=args.mask_coefficient,
        dice_coefficient=args.dice_coefficient,
        class_coefficient=args.class_coefficient,
        bbox_l1_coefficient=args.bbox_l1_coefficient,
        bbox_giou_coefficient=args.bbox_giou_coefficient,
        mask_thresh=args.mask_thresh,
        overlap_thresh=args.overlap_thresh,
        eval_top_k_instances=args.eval_top_k_instances,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path is not None else None,
        delta_weights=args.delta_weights,
        load_ckpt_class_head=args.load_ckpt_class_head,
    )


def configure_runtime() -> None:
    torch.set_float32_matmul_precision("medium")
    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
    warnings.filterwarnings(
        "ignore",
        message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
    )
    warnings.filterwarnings(
        "ignore", message=r"^Grad strides do not match bucket view strides.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*functools.partial will be a method descriptor in future Python versions*",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.devices not in (1, "1", "auto"):
        raise ValueError(
            "Pure PyTorch training currently supports a single device only. Use --devices 1."
        )

    accelerator = str(args.accelerator).lower()
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

    raise ValueError(f"Unsupported accelerator: {args.accelerator}")


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


class MetricCSVLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.csv"
        self._file = self.metrics_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=("timestamp", "epoch", "step", "phase", "name", "value"),
        )
        if self.metrics_path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    def save_hparams(self, args: argparse.Namespace) -> None:
        with (self.log_dir / "hparams.json").open("w", encoding="utf-8") as file:
            json.dump(vars(args), file, indent=2, default=str)

    def log_metrics(
        self,
        phase: str,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        timestamp = time.time()
        for name, value in sorted(metrics.items()):
            scalar = to_scalar(value)
            if scalar is None:
                continue
            self._writer.writerow(
                {
                    "timestamp": f"{timestamp:.6f}",
                    "epoch": epoch,
                    "step": step,
                    "phase": phase,
                    "name": name,
                    "value": f"{scalar:.10f}",
                }
            )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.resume_from_checkpoint is not None:
        checkpoint_dir = args.resume_from_checkpoint.parent
        if checkpoint_dir.name == "checkpoints":
            return checkpoint_dir.parent
        return checkpoint_dir

    return args.log_dir / args.experiment_name


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


def drain_model_metrics(model: MaskClassificationInstance) -> dict[str, Any]:
    if hasattr(model, "consume_logged_metrics"):
        return model.consume_logged_metrics()
    return {}


def build_runtime(total_steps: int, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        estimated_stepping_batches=total_steps,
        callback_metrics={},
        sanity_checking=False,
        is_global_zero=True,
        default_root_dir=str(run_dir),
    )


def save_checkpoint(
    path: Path,
    model: MaskClassificationInstance,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
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
    model: MaskClassificationInstance,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
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


def format_metrics(metrics: dict[str, Any], names: list[str]) -> str:
    items = []
    for name in names:
        value = to_scalar(metrics.get(name))
        if value is not None:
            items.append(f"{name}={value:.4f}")
    return " | ".join(items)


def train_one_epoch(
    model: MaskClassificationInstance,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    logger: MetricCSVLogger,
    epoch: int,
    log_every_n_steps: int,
) -> None:
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            loss = model.training_step(batch, batch_idx)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=0.01)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=0.01)
            optimizer.step()

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
    model: MaskClassificationInstance,
    val_loader,
    device: torch.device,
    logger: MetricCSVLogger,
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


def main() -> None:
    args = parse_args()
    configure_runtime()
    seed_everything(args.seed)

    device = resolve_device(args)
    amp_enabled, amp_dtype, use_scaler = resolve_amp(device, args.precision)
    scaler = torch.amp.GradScaler(device=device,enabled=use_scaler)

    run_dir = resolve_run_dir(args)
    logger = MetricCSVLogger(run_dir)
    logger.save_hparams(args)

    datamodule = build_datamodule(args)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = build_model(args)
    model.max_epochs = args.max_epochs
    model.to(device)
    model.logger = SimpleNamespace(log_dir=str(run_dir))
    model.trainer = build_runtime(
        total_steps=len(train_loader) * args.max_epochs,
        run_dir=run_dir,
    )

    optim_config = model.configure_optimizers()
    optimizer = optim_config["optimizer"]
    scheduler = optim_config["lr_scheduler"]["scheduler"]

    start_epoch = 0
    best_val_ap_all = float("-inf")
    if args.resume_from_checkpoint is not None:
        start_epoch, best_val_ap_all = load_training_state(
            args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    if args.compile:
        model.network = torch.compile(model.network)

    print(
        f"Training on {device} with precision={args.precision}, "
        f"log_dir={run_dir}"
    )

    try:
        for epoch in range(start_epoch, args.max_epochs):
            model.current_epoch = epoch
            epoch_start = time.time()

            train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                logger=logger,
                epoch=epoch,
                log_every_n_steps=args.log_every_n_steps,
            )

            should_validate = (
                (epoch + 1) % args.check_val_every_n_epoch == 0
                or epoch + 1 == args.max_epochs
            )
            if should_validate:
                val_metrics = validate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    logger=logger,
                    epoch=epoch,
                )
                current_val_ap_all = to_scalar(val_metrics.get("metrics/val_ap_all"))
                if (
                    current_val_ap_all is not None
                    and current_val_ap_all > best_val_ap_all
                ):
                    best_val_ap_all = current_val_ap_all
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        best_val_ap_all,
                    )

            save_checkpoint(
                run_dir / "checkpoints" / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_ap_all,
            )

            logger.log_metrics(
                "epoch",
                epoch + 1,
                int(model.global_step),
                {"epoch/duration_sec": time.time() - epoch_start},
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
