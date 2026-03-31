import argparse
import sys
import warnings
from pathlib import Path
from types import MethodType

import torch
from lightning.fabric.utilities.seed import seed_everything
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.coco_instance import COCOInstance
from models.eomt_dinov3 import EoMT
from training.mask_classification_instance import MaskClassificationInstance


def _should_check_val_fx(
    self: _TrainingEpochLoop, data_fetcher: _DataFetcher
) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            is_val_check_batch = (
                self.global_step % self.trainer.val_check_batch == 0
                and not self._should_accumulate()
            )

    return is_val_check_batch


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
    return "16-mixed"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone COCO instance training script for EoMT DINOv3."
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--img-size", type=int, nargs=2, default=(640, 640))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.1, 2.0))
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=2)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="1")
    parser.add_argument("--precision", default=None)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs")
    parser.add_argument(
        "--experiment-name",
        default="coco_instance_eomt_large_640_dinov3_py",
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


def build_trainer(args: argparse.Namespace) -> Trainer:
    logger = CSVLogger(
        save_dir=str(args.log_dir),
        name=args.experiment_name,
    )
    return Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        precision=args.precision,
        logger=logger,
        enable_model_summary=False,
        callbacks=[
            ModelSummary(max_depth=3),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        gradient_clip_val=0.01,
        gradient_clip_algorithm="norm",
    )


def configure_runtime() -> None:
    torch.set_float32_matmul_precision("medium")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.suppress_errors = True
    warnings.filterwarnings(
        "ignore",
        message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
    )
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


def main() -> None:
    args = parse_args()
    configure_runtime()
    seed_everything(args.seed, workers=True)

    datamodule = build_datamodule(args)
    model = build_model(args)
    trainer = build_trainer(args)

    trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
        _should_check_val_fx, trainer.fit_loop.epoch_loop
    )

    model.train()
    trainable_model = torch.compile(model) if args.compile else model
    trainer.fit(
        trainable_model,
        datamodule=datamodule,
        ckpt_path=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
    )


if __name__ == "__main__":
    main()
