from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, box_iou, generalized_box_iou_loss, masks_to_boxes
from torchvision import tv_tensors


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.coco_instance import COCOInstance
from training.checkpointing import cpu_state_dict, resolve_run_dir, sanitize_state_dict
from training.csv_logger import MetricCSVLogger
from training.runtime import autocast_context, resolve_amp, resolve_device, seed_everything
from training.scheduler import TwoStageWarmupPolySchedule
from scripts.utils import box_cxcywh_to_xyxy, save_mask_with_box

class MaskResidualBoxHead(nn.Module):
    def __init__(self, hidden=64, delta_scale=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.delta_scale = delta_scale
        self._coord_cache = {}

    def get_coords(self, H, W, device, dtype, N):
        key = (H, W, device.type, device.index, str(dtype))
        if key not in self._coord_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]
            self._coord_cache[key] = coord
        return self._coord_cache[key].expand(N, -1, -1, -1)

    def forward(self, mask_logits, boxes_cxcywh=None):
        B, Q, H, W = mask_logits.shape
        N = B * Q

        if boxes_cxcywh is None:
            with torch.no_grad():
                masks = (mask_logits.reshape(N, H, W) > 0.0)

                # pixel-space xyxy
                base_boxes = masks_to_boxes(masks)

                # handle empty masks explicitly
                empty = masks.flatten(1).sum(dim=1) == 0
                if empty.any():
                    base_boxes[empty] = 0.0

                # normalize to [0, 1]
                base_boxes = base_boxes.to(mask_logits.device, mask_logits.dtype)                
                base_boxes[:, [0, 2]] /= max(W - 1, 1)
                base_boxes[:, [1, 3]] /= max(H - 1, 1)
                
                boxes_cxcywh = box_convert(
                    base_boxes,
                    in_fmt="xyxy",
                    out_fmt="cxcywh",
                ).clamp(0, 1)
                
        else:
            boxes_cxcywh = boxes_cxcywh.reshape(N, 4).to(mask_logits.device, mask_logits.dtype)

        x = mask_logits.sigmoid().reshape(N, 1, H, W)
        coord = self.get_coords(H, W, x.device, x.dtype, N)
        x = torch.cat([x, coord], dim=1)

        # small bounded residual in normalized box space
        delta = self.delta_scale * torch.tanh(self.net(x))

        out_boxes_cxcywh = (boxes_cxcywh + delta).clamp(0.0, 1.0)
        return out_boxes_cxcywh.reshape(B, Q, 4)
    

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
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_path}")
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.exists():
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MaskResidualBoxHead directly from COCO binary masks."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument(
        "--experiment-name",
        default="mask_residual_box_head_coco",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--poly-power", type=float, default=0.9)
    parser.add_argument("--l1-coefficient", type=float, default=5.0)
    parser.add_argument("--giou-coefficient", type=float, default=2.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--check-empty-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "dataset_cache")
    parser.add_argument(
        "--force-rebuild-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--verbose-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    validate_paths: bool = True,
) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    args.devices = _parse_devices(args.devices)
    args.img_size = tuple(args.img_size)
    args.data_path = args.data_path.resolve()
    args.log_dir = args.log_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve()
    if args.precision is None:
        args.precision = _default_precision(args.accelerator)

    if validate_paths:
        _validate_paths(args)
    return args


def build_data_module(args: argparse.Namespace) -> COCOInstance:
    return COCOInstance(
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_classes=80,
        check_empty_targets=args.check_empty_targets,
        cache_dir=args.cache_dir,
        force_rebuild_cache=args.force_rebuild_cache,
        verbose_cache=args.verbose_cache,
        save_cache=args.save_cache,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
    )


def build_instance_batch(
    targets: list[dict] | tuple[dict, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mask_batches = []
    box_cxcywh_batches = []

    for target in targets:
        masks = target["masks"]
        boxes:tv_tensors.BoundingBoxes = target["boxes"]
        height, width = boxes.canvas_size
        # fmt = boxes.format
        # clamping_mode = boxes.clamping_mode


        if masks.ndim == 2:
            masks = masks.unsqueeze(0)

        if masks.numel() == 0 or boxes.numel() == 0:
            continue

        masks = masks.to(device=device, dtype=torch.bool)
        boxes = boxes.to(device=device, dtype=torch.float32)

        keep = masks.flatten(1).any(dim=1)
        if not keep.any():
            continue

        masks = masks[keep]
        boxes = boxes[keep]

        scale = torch.tensor(
            [width, height, width, height],
            device=device,
            dtype=boxes.dtype,
        )

        boxes_cxcywh = box_convert(
            boxes / scale,
            in_fmt="xyxy",
            out_fmt="cxcywh",
        ).clamp(0, 1)        

        mask_batches.append(masks.float())
        box_cxcywh_batches.append(boxes_cxcywh)

    if not mask_batches:
        return None, None

    mask_batch = torch.cat(mask_batches, dim=0).unsqueeze(1)
    box_cxcywh_batch = torch.cat(box_cxcywh_batches, dim=0)

    return mask_batch, box_cxcywh_batch

def compute_batch_metrics(
    model: MaskResidualBoxHead,
    mask_batch: torch.Tensor,
    target_boxes_cxcywh: torch.Tensor,
    *,
    l1_coefficient: float,
    giou_coefficient: float,
) -> dict[str, torch.Tensor]:
    pred_logits = model.forward(mask_batch).squeeze(1)
    pred_boxes_cxcywh = pred_logits#.sigmoid()

    loss_l1 = F.l1_loss(pred_boxes_cxcywh, target_boxes_cxcywh, reduction="mean")

    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
    target_xyxy = box_cxcywh_to_xyxy(target_boxes_cxcywh)
    loss_giou = generalized_box_iou_loss(
        pred_xyxy.float(),
        target_xyxy.float(),
        reduction="mean",
    )

    iou = box_iou(pred_xyxy.float(), target_xyxy.float()).diag().mean()
    loss_total = l1_coefficient * loss_l1 + giou_coefficient * loss_giou

    return {
        "loss_total": loss_total,
        "loss_l1": loss_l1,
        "loss_giou": loss_giou,
        "iou": iou,
        "instances": torch.tensor(
            float(target_boxes_cxcywh.shape[0]),
            device=target_boxes_cxcywh.device,
        ),
    }


def save_training_state(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    *,
    epoch: int,
    global_step: int,
    best_val_iou: float,
) -> None:
    checkpoint = {
        "model_state_dict": cpu_state_dict(model.state_dict()),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_iou": float(best_val_iou),
    }
    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_weights(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(model.state_dict()), path)


def load_training_state(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
) -> tuple[int, int, float]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        sanitize_state_dict(checkpoint["model_state_dict"]),
        strict=True,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    global_step = int(checkpoint.get("global_step", 0))
    best_val_iou = float(checkpoint.get("best_val_iou", float("-inf")))
    return start_epoch, global_step, best_val_iou


def _aggregate_epoch_metrics(
    running: dict[str, float],
    total_instances: int,
) -> dict[str, float]:
    if total_instances == 0:
        return {
            "loss_total": float("nan"),
            "loss_l1": float("nan"),
            "loss_giou": float("nan"),
            "iou": float("nan"),
            "instances": 0.0,
        }

    denom = float(total_instances)
    return {
        "loss_total": running["loss_total"] / denom,
        "loss_l1": running["loss_l1"] / denom,
        "loss_giou": running["loss_giou"] / denom,
        "iou": running["iou"] / denom,
        "instances": denom,
    }


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    epoch: int,
    max_epochs: int,
    logger: MetricCSVLogger,
    global_step: int,
    log_every_n_steps: int,
    l1_coefficient: float,
    giou_coefficient: float,
    clip_grad_norm: float,
) -> tuple[int, dict[str, float]]:
    model.train()

    running = {
        "loss_total": 0.0,
        "loss_l1": 0.0,
        "loss_giou": 0.0,
        "iou": 0.0,
    }
    total_instances = 0

    for batch_idx, batch in enumerate(train_loader):
        _, targets = batch
        mask_batch, box_cxcywh_batch = build_instance_batch(targets, device=device)

        if mask_batch is None or box_cxcywh_batch is None:
            continue

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled, amp_dtype):
            metrics = compute_batch_metrics(
                model,
                mask_batch,
                box_cxcywh_batch,
                l1_coefficient=l1_coefficient,
                giou_coefficient=giou_coefficient,
            )
            loss = metrics["loss_total"]

        if not torch.isfinite(loss):
            print(
                f"[WARN] Skipping batch {batch_idx + 1}/{len(train_loader)} "
                f"at epoch {epoch + 1}: non-finite loss={loss.detach().item()}"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        try:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad_norm,
                    error_if_nonfinite=True,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
        except RuntimeError as exc:
            print(
                f"[WARN] RuntimeError during backward/step at batch {batch_idx + 1}: {exc}"
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.update()
            continue

        scheduler.step()
        global_step += 1

        batch_instances = int(box_cxcywh_batch.shape[0])
        total_instances += batch_instances
        for key in ("loss_total", "loss_l1", "loss_giou", "iou"):
            running[key] += float(metrics[key].detach().item()) * batch_instances

        step_metrics = {
            "loss_total": float(metrics["loss_total"].detach().item()),
            "loss_l1": float(metrics["loss_l1"].detach().item()),
            "loss_giou": float(metrics["loss_giou"].detach().item()),
            "iou": float(metrics["iou"].detach().item()),
            "instances": float(batch_instances),
            "grad_norm": float(grad_norm.detach().item()),
            "lr": float(max(group["lr"] for group in optimizer.param_groups)),
        }
        logger.log_metrics("train", epoch + 1, global_step, step_metrics)

        should_print = (
            batch_idx == 0
            or (batch_idx + 1) % log_every_n_steps == 0
            or batch_idx + 1 == len(train_loader)
        )
        if should_print:
            print(
                f"Epoch {epoch + 1}/{max_epochs} "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"Step {global_step} | "
                f"loss_total={step_metrics['loss_total']:.4f} | "
                f"loss_l1={step_metrics['loss_l1']:.4f} | "
                f"loss_giou={step_metrics['loss_giou']:.4f} | "
                f"iou={step_metrics['iou']:.4f} | "
                f"instances={batch_instances} | "
                f"lr={step_metrics['lr']:.6f}"
            )

    epoch_metrics = _aggregate_epoch_metrics(running, total_instances)
    logger.log_metrics("train_epoch", epoch + 1, global_step, epoch_metrics)
    return global_step, epoch_metrics


def validate(
    model: torch.nn.Module,
    val_loader,
    *,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    epoch: int,
    global_step: int,
    logger: MetricCSVLogger,
    l1_coefficient: float,
    giou_coefficient: float,
) -> dict[str, float]:
    model.eval()

    running = {
        "loss_total": 0.0,
        "loss_l1": 0.0,
        "loss_giou": 0.0,
        "iou": 0.0,
    }
    total_instances = 0

    with torch.inference_mode():
        for _, targets in val_loader:
            mask_batch, target_boxes = build_instance_batch(targets, device=device)
            if mask_batch is None or target_boxes is None:
                continue

            with autocast_context(device, amp_enabled, amp_dtype):
                metrics = compute_batch_metrics(
                    model,
                    mask_batch,
                    target_boxes,
                    l1_coefficient=l1_coefficient,
                    giou_coefficient=giou_coefficient,
                )

            batch_instances = int(target_boxes.shape[0])
            total_instances += batch_instances
            for key in ("loss_total", "loss_l1", "loss_giou", "iou"):
                running[key] += float(metrics[key].detach().item()) * batch_instances

    epoch_metrics = _aggregate_epoch_metrics(running, total_instances)
    logger.log_metrics("val", epoch + 1, global_step, epoch_metrics)
    return epoch_metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")

    device = resolve_device(args.accelerator, args.devices)
    amp_enabled, amp_dtype, use_scaler = resolve_amp(device, args.precision)
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    run_dir = resolve_run_dir(
        args.log_dir,
        args.experiment_name,
        args.resume_from_checkpoint,
    )
    logger = MetricCSVLogger(run_dir)
    logger.save_hparams(args)

    datamodule = build_data_module(args)
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = MaskResidualBoxHead(hidden=args.hidden).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.max_epochs)
    scheduler = TwoStageWarmupPolySchedule(
        optimizer,
        num_backbone_params=0,
        warmup_steps=(args.warmup_steps, 0),
        total_steps=total_steps,
        poly_power=args.poly_power,
    )

    start_epoch = 0
    global_step = 0
    best_val_iou = float("-inf")

    if args.resume_from_checkpoint is not None:
        start_epoch, global_step, best_val_iou = load_training_state(
            args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
        )

    if args.compile:
        model = torch.compile(model)

    param_count = sum(int(param.numel()) for param in model.parameters())
    print(
        f"Training MaskResidualBoxHead on {device} with precision={args.precision}, "
        f"params={param_count:,}, log_dir={run_dir}"
    )

    for epoch in range(start_epoch, args.max_epochs):
        global_step, train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch,
            max_epochs=args.max_epochs,
            logger=logger,
            global_step=global_step,
            log_every_n_steps=args.log_every_n_steps,
            l1_coefficient=args.l1_coefficient,
            giou_coefficient=args.giou_coefficient,
            clip_grad_norm=args.clip_grad_norm,
        )

        val_metrics = validate(
            model,
            val_loader,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            epoch=epoch,
            global_step=global_step,
            logger=logger,
            l1_coefficient=args.l1_coefficient,
            giou_coefficient=args.giou_coefficient,
        )

        print(
            f"Epoch {epoch + 1}/{args.max_epochs} summary | "
            f"train_loss={train_metrics['loss_total']:.4f} | "
            f"train_iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss_total']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        current_val_iou = val_metrics["iou"]
        if current_val_iou == current_val_iou and current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            save_training_state(
                run_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                global_step=global_step,
                best_val_iou=best_val_iou,
            )
            save_weights(
                run_dir / "weights" / "best_mask_residual_box_head.pt",
                model,
            )

        save_training_state(
            run_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch=epoch,
            global_step=global_step,
            best_val_iou=best_val_iou,
        )
        save_weights(run_dir / "weights" / "last_mask_residual_box_head.pt", model)

    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


#   uv run python scripts/train_mask_residual_box_head.py \
#     --data-path /path/to/coco \
#     --accelerator cuda \
#     --devices 1 \
#     --batch-size 16 \
#     --num-workers 4 \
#     --img-size 640 640 \
#     --max-epochs 12 \
#     --experiment-name mask_residual_box_head_coco