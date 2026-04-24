from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    
from scripts.utils import summ
from training.checkpointing import (
    load_training_state,
    resolve_run_dir,
    save_checkpoint,
)
from training.csv_logger import MetricCSVLogger
from training.engine import train_one_epoch, validate
from training.runtime import (
    build_runtime,
    resolve_amp,
    resolve_device,
    seed_everything,
    to_scalar,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_encoder_repo() -> Path:
    return (REPO_ROOT / "../dinov3").resolve()


def _default_encoder_weights() -> Path:
    return (
        REPO_ROOT / "./data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    ).resolve()


def _default_official_delta_ckpt() -> Path:
    return (REPO_ROOT / "data/EoMT-L_640×640_InstanceSegmentation_DINOv3.bin").resolve()


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
    if args.network_impl == "dinov3":
        if not args.encoder_repo.exists():
            raise FileNotFoundError(f"DINOv3 repo path does not exist: {args.encoder_repo}")
        if not args.encoder_weights.exists():
            raise FileNotFoundError(
                f"DINOv3 checkpoint does not exist: {args.encoder_weights}"
            )
    elif args.network_impl == "original_bbox":
        # if not args.official_delta_ckpt.exists():
        #     raise FileNotFoundError(
        #         f"Official EoMT delta checkpoint does not exist: {args.official_delta_ckpt}"
        #     )
        pass
    else:
        raise ValueError(f"Unsupported --network-impl value: {args.network_impl}")
    if args.ckpt_path is not None and not args.ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint does not exist: {args.ckpt_path}")
    if (
        args.resume_from_checkpoint is not None
        and not args.resume_from_checkpoint.exists()
    ):
        raise FileNotFoundError(
            f"Resume checkpoint does not exist: {args.resume_from_checkpoint}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure PyTorch COCO instance training for EoMT."
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
        default="coco_instance_eomt_large_640_dinov3",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--network-impl",
        choices=("dinov3", "original_bbox"),
        default="dinov3",
    )
    parser.add_argument("--encoder-repo", type=Path, default=_default_encoder_repo())
    parser.add_argument("--encoder-model", default="dinov3_vits16")
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=_default_encoder_weights(),
    )
    parser.add_argument(
        "--official-backbone-name",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--official-delta-ckpt",
        type=Path,
        default=_default_official_delta_ckpt(),
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
    parser.add_argument("--owner-coefficient", type=float, default=1.0)
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
    parser.add_argument("--ckpt-path", type=Path, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    return parser


def parse_args(
    argv: list[str] | None = None,
    *,
    validate_paths: bool = True,
) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if len(args.attn_mask_annealing_start_steps) != args.num_blocks:
        raise ValueError("--attn-mask-annealing-start-steps must match --num-blocks")
    if len(args.attn_mask_annealing_end_steps) != args.num_blocks:
        raise ValueError("--attn-mask-annealing-end-steps must match --num-blocks")

    args.devices = _parse_devices(args.devices)
    args.img_size = tuple(args.img_size)
    args.scale_range = tuple(args.scale_range)
    args.warmup_steps = tuple(args.warmup_steps)
    args.data_path = args.data_path.resolve()
    args.log_dir = args.log_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.encoder_repo = args.encoder_repo.resolve()
    args.encoder_weights = args.encoder_weights.resolve()
    args.official_delta_ckpt = args.official_delta_ckpt.resolve()
    if args.ckpt_path is not None:
        args.ckpt_path = args.ckpt_path.resolve()
    if args.resume_from_checkpoint is not None:
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve()
    if args.precision is None:
        args.precision = _default_precision(args.accelerator)

    if validate_paths:
        _validate_paths(args)
    return args


def build_data_module(args: argparse.Namespace):
    from datasets.coco_instance import COCOInstance

    return COCOInstance(
        path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        num_classes=args.num_classes,
        check_empty_targets=args.check_empty_targets,
        cache_dir=args.cache_dir,
        force_rebuild_cache=args.force_rebuild_cache,
        verbose_cache=args.verbose_cache,
        save_cache=args.save_cache,

        color_jitter_enabled=args.color_jitter_enabled,
        scale_range=args.scale_range,

        # for overfiting
        # color_jitter_enabled=False,
        # scale_range=(1.0, 1.0),
        # overfit_indices=[0, 1],
        # overfit_repeat=2000,
    )


def build_model(args: argparse.Namespace):
    from training.instance_module import MaskClassificationInstance

    if args.network_impl == "original_bbox":
        raise ValueError("original_bbox no more use")
        # from models.eomt import freeze_module_as_buffers
        # from models.official_eomt import load_official_dinov3_delta
        # from models.original_eomt import EoMT as BboxEoMT
        # from models.vit import ViT
        # network = BboxEoMT(
        #     encoder=ViT(
        #         img_size=args.img_size,
        #         backbone_name=args.official_backbone_name,
        #     ),
        #     num_classes=args.num_classes,
        #     num_q=args.num_q,
        #     num_blocks=args.num_blocks,
        #     masked_attn_enabled=args.masked_attn_enabled,
        # )
        # if os.path.exists(args.official_delta_ckpt):
        #     load_official_dinov3_delta(network, str(args.official_delta_ckpt))
        # else:
        #     print(f"skip load_official_dinov3_delta of no exist {str(args.official_delta_ckpt)}")
        # # freeze_module_as_buffers(network)
        # network.init_bbox_head()
        # delta_weights = False
    else:
        from models.eomt import EoMT

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
            bbox_head_weight=args.bbox_head_weight,
        )
        delta_weights = args.delta_weights

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
        owner_coefficient=args.owner_coefficient,
        mask_thresh=args.mask_thresh,
        overlap_thresh=args.overlap_thresh,
        eval_top_k_instances=args.eval_top_k_instances,
        ckpt_path=str(args.ckpt_path) if args.ckpt_path is not None else None,
        delta_weights=delta_weights,
        load_ckpt_class_head=args.load_ckpt_class_head,
    )


def _format_param_count(count: int) -> str:
    return f"{count:,}"


def print_model_details(model, args: argparse.Namespace, optimizer) -> None:
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    trainable_tensors: list[tuple[str, int]] = []

    for name, param in model.named_parameters():
        numel = int(param.numel())
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
            trainable_tensors.append((name, numel))
        else:
            non_trainable_params += numel

    buffer_count = sum(int(buf.numel()) for buf in model.buffers())

    print("Model details:")
    print(f"  network_impl: {args.network_impl}")
    print(f"  total_params: {_format_param_count(total_params)}")
    print(f"  trainable_params: {_format_param_count(trainable_params)}")
    print(f"  non_trainable_params: {_format_param_count(non_trainable_params)}")
    print(f"  buffers: {_format_param_count(buffer_count)}")
    print(f"  optimizer_param_groups: {len(optimizer.param_groups)}")

    if trainable_tensors:
        print("  trainable_tensors:")
        for name, numel in trainable_tensors:
            print(f"    - {name}: {_format_param_count(numel)}")
    else:
        print("  trainable_tensors: none")


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
        "ignore",
        message=r"^Grad strides do not match bucket view strides.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*functools.partial will be a method descriptor in future Python versions*",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_runtime()
    seed_everything(args.seed)

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

    if "bf16" in args.precision:
        dtype = torch.bfloat16
    elif "fp16" in args.precision:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = build_model(args).to(dtype=dtype)
    model.max_epochs = args.max_epochs
    model.to(device)
    model.logger = argparse.Namespace(log_dir=str(run_dir))
    model.trainer = build_runtime(
        total_steps=len(train_loader) * args.max_epochs,
        run_dir=run_dir,
    )

    optim_config = model.configure_optimizers()
    optimizer = optim_config["optimizer"]
    scheduler = optim_config["lr_scheduler"]["scheduler"]
    print_model_details(model, args, optimizer)

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
            
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                with torch.amp.autocast('cuda', dtype=torch.float16):
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
    finally:
        logger.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
