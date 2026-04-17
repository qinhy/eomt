from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.detection import MeanAveragePrecision

from models import EoMT
from training.loss import MaskClassificationLoss
from training.scheduler import TwoStageWarmupPolySchedule


if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
    compiler_disable = torch.compiler.disable
else:
    def compiler_disable(fn):
        return fn


def rank_zero_info(message: str) -> None:
    print(message)


bold_green = "\033[1;32m"
reset = "\033[0m"


class TrainModule(nn.Module):
    def __init__(
        self,
        network: EoMT,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]],
        attn_mask_annealing_end_steps: Optional[list[int]],
        lr: float,
        llrd: float,
        llrd_l2_enabled: bool,
        lr_mult: float,
        weight_decay: float,
        poly_power: float,
        warmup_steps: tuple[int, int] | list[int],
        ckpt_path=None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ) -> None:
        super().__init__()

        self.network = network
        self.img_size = img_size
        self.num_classes = num_classes
        self.attn_mask_annealing_enabled = attn_mask_annealing_enabled
        self.attn_mask_annealing_start_steps = attn_mask_annealing_start_steps
        self.attn_mask_annealing_end_steps = attn_mask_annealing_end_steps
        self.lr = lr
        self.llrd = llrd
        self.lr_mult = lr_mult
        self.weight_decay = weight_decay
        self.poly_power = poly_power
        self.warmup_steps = tuple(warmup_steps)
        self.llrd_l2_enabled = llrd_l2_enabled

        self.criterion: MaskClassificationLoss = None
        self.strict_loading = False
        self.global_step = 0
        self.current_epoch = 0
        self.logger = None
        self.trainer = SimpleNamespace(
            callback_metrics={},
            sanity_checking=False,
            is_global_zero=True,
            default_root_dir=".",
        )
        self._logged_metrics: dict[str, object] = {}

        if delta_weights and ckpt_path:
            logging.info("Delta weights mode")
            self._zero_init_outside_encoder(skip_class_head=not load_ckpt_class_head)
            current_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
            if not load_ckpt_class_head:
                current_state_dict = {
                    k: v
                    for k, v in current_state_dict.items()
                    if "class_head" not in k and "class_predictor" not in k
                }
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            combined_state_dict = self._add_state_dicts(current_state_dict, ckpt)
            incompatible_keys = self.load_state_dict(combined_state_dict, strict=True)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)
        elif ckpt_path:
            ckpt = self._load_ckpt(ckpt_path, load_ckpt_class_head)
            incompatible_keys = self.load_state_dict(ckpt, strict=True)
            self._raise_on_incompatible(incompatible_keys, load_ckpt_class_head)

        self.log = compiler_disable(self.log)  # type: ignore[assignment]

    @property
    def device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device

        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device

        return torch.device("cpu")

    def save_hyperparameters(self, *args, **kwargs) -> None:
        pass

    def log(self, name, value, *args, **kwargs) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                value = float(value.item())
            else:
                value = value.cpu()

        self._logged_metrics[name] = value
        callback_metrics = getattr(self.trainer, "callback_metrics", None)
        if isinstance(callback_metrics, dict):
            callback_metrics[name] = value

    def consume_logged_metrics(self) -> dict[str, object]:
        metrics = dict(self._logged_metrics)
        self._logged_metrics.clear()
        return metrics

    def configure_optimizers(self):
        backbone = getattr(self.network.encoder, "backbone", self.network.encoder)
        backbone_name_by_id = {id(param): name for name, param in backbone.named_parameters()}
        backbone_param_ids = set(backbone_name_by_id.keys())

        backbone_param_groups = []
        other_param_groups = []

        backbone_blocks = len(backbone.blocks)
        l2_blocks = set(
            range(backbone_blocks - self.network.num_blocks, backbone_blocks)
        )

        for full_name, param in reversed(list(self.named_parameters())):
            lr = self.lr

            if id(param) in backbone_param_ids:
                local_name = backbone_name_by_id[id(param)]
                name_parts = local_name.split(".")

                is_block = False
                block_i = None
                for i, key in enumerate(name_parts):
                    if key == "blocks" and i + 1 < len(name_parts):
                        block_i = int(name_parts[i + 1])
                        is_block = True
                        break

                if is_block:
                    lr *= self.llrd ** (backbone_blocks - 1 - block_i)

                if local_name == "norm" or local_name.startswith("norm.") or "backbone.norm" in full_name:
                    lr = self.lr

                if (
                    is_block
                    and block_i in l2_blocks
                    and ((not self.llrd_l2_enabled) or (self.lr_mult != 1.0))
                ):
                    lr = self.lr

                backbone_param_groups.append(
                    {"params": [param], "lr": lr, "name": full_name}
                )
            else:
                other_param_groups.append(
                    {"params": [param], "lr": self.lr, "name": full_name}
                )

        optimizer = AdamW(backbone_param_groups + other_param_groups, weight_decay=self.weight_decay)
        scheduler = TwoStageWarmupPolySchedule(
            optimizer,
            num_backbone_params=len(backbone_param_groups),
            warmup_steps=self.warmup_steps,
            total_steps=self.trainer.estimated_stepping_batches,
            poly_power=self.poly_power,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def forward(self, imgs, imgs2=None):
        assert imgs.dtype == torch.uint8, "input image should be raw uint8 images"

        x = imgs / 255.0
        x2 = None
        if imgs2 is not None:
            assert imgs2.dtype == torch.uint8, "input image should be raw uint8 images"
            x2 = imgs2 / 255.0

        if x2 is None:
            return self.network.forward(x)
        return self.network.forward(x, x2)

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        mask_logits_per_block, class_logits_per_block, bbox_preds_per_block = self(imgs)

        losses_all_blocks = {}
        for i, (mask_logits, class_logits, bbox_preds) in enumerate(
            zip(mask_logits_per_block, class_logits_per_block, bbox_preds_per_block)
        ):
            losses = self.criterion(
                masks_queries_logits=mask_logits,
                class_queries_logits=class_logits,
                bbox_queries_preds=bbox_preds,
                targets=targets,
            )
            block_postfix = self.block_postfix(i)
            losses_all_blocks |= {
                f"{key}{block_postfix}": value for key, value in losses.items()
            }

        return self.criterion.loss_total(losses_all_blocks, self.log)

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def mask_annealing(self, start_iter, current_iter, final_iter):
        dtype = self.network.attn_mask_probs[0].dtype
        if current_iter < start_iter:
            return torch.ones(1, device=self.device, dtype=dtype)
        if current_iter >= final_iter:
            return torch.zeros(1, device=self.device, dtype=dtype)

        progress = (current_iter - start_iter) / (final_iter - start_iter)
        progress = torch.tensor(progress, device=self.device, dtype=dtype)
        return (1.0 - progress).pow(self.poly_power)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        if not self.attn_mask_annealing_enabled:
            return

        for i in range(self.network.num_blocks):
            self.network.attn_mask_probs[i] = self.mask_annealing(
                self.attn_mask_annealing_start_steps[i],
                self.global_step,
                self.attn_mask_annealing_end_steps[i],
            )

        for i, attn_mask_prob in enumerate(self.network.attn_mask_probs):
            self.log(f"attn_mask_prob_{i}", attn_mask_prob, on_step=True)

    def init_metrics_instance(self, num_blocks: int) -> None:
        self.metrics_segm = nn.ModuleList(
            [MeanAveragePrecision(iou_type="segm") for _ in range(num_blocks)]
        )
        self.metrics_bbox = nn.ModuleList(
            [MeanAveragePrecision(iou_type="bbox") for _ in range(num_blocks)]
        )

    @compiler_disable
    def update_metrics_instance(
        self,
        preds: list[dict],
        targets: list[dict],
        block_idx: int,
    ) -> None:
        segm_preds = [
            {"masks": pred["masks"], "labels": pred["labels"], "scores": pred["scores"]}
            for pred in preds
        ]
        segm_targets = []
        for target in targets:
            segm_target = {"masks": target["masks"], "labels": target["labels"]}
            if "iscrowd" in target:
                segm_target["iscrowd"] = target["iscrowd"]
            segm_targets.append(segm_target)

        self.metrics_segm[block_idx].update(segm_preds, segm_targets)

        bbox_preds = [
            {"boxes": pred["boxes"], "labels": pred["labels"], "scores": pred["scores"]}
            for pred in preds
        ]
        bbox_targets = []
        for target in targets:
            bbox_target = {"boxes": target["boxes"], "labels": target["labels"]}
            if "iscrowd" in target:
                bbox_target["iscrowd"] = target["iscrowd"]
            bbox_targets.append(bbox_target)

        self.metrics_bbox[block_idx].update(bbox_preds, bbox_targets)

    def block_postfix(self, block_idx: int) -> str:
        if not self.network.masked_attn_enabled:
            return ""
        return (
            f"_block_{-len(self.metrics_segm) + block_idx + 1}"
            if block_idx != self.network.num_blocks
            else ""
        )

    def _on_eval_epoch_end_instance(self, log_prefix: str) -> None:
        for i, (segm_metric, bbox_metric) in enumerate(
            zip(self.metrics_segm, self.metrics_bbox)
        ):
            segm_results = segm_metric.compute()
            bbox_results = bbox_metric.compute()
            segm_metric.reset()
            bbox_metric.reset()

            block_postfix = self.block_postfix(i)
            self.log(f"metrics/{log_prefix}_ap_all{block_postfix}", segm_results["map"])
            self.log(
                f"metrics/{log_prefix}_ap_small_all{block_postfix}",
                segm_results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_medium_all{block_postfix}",
                segm_results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_large_all{block_postfix}",
                segm_results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_50_all{block_postfix}",
                segm_results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_ap_75_all{block_postfix}",
                segm_results["map_75"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_all{block_postfix}",
                bbox_results["map"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_small_all{block_postfix}",
                bbox_results["map_small"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_medium_all{block_postfix}",
                bbox_results["map_medium"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_large_all{block_postfix}",
                bbox_results["map_large"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_50_all{block_postfix}",
                bbox_results["map_50"],
            )
            self.log(
                f"metrics/{log_prefix}_bbox_ap_75_all{block_postfix}",
                bbox_results["map_75"],
            )

    def _on_eval_end_instance(self, log_prefix: str) -> None:
        if self.trainer.sanity_checking:
            return
        rank_zero_info(
            f"{bold_green}mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_all'] * 100:.1f} | "
            f"mAP Small: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_small_all'] * 100:.1f} | "
            f"mAP Medium: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_medium_all'] * 100:.1f} | "
            f"mAP Large: {self.trainer.callback_metrics[f'metrics/{log_prefix}_ap_large_all'] * 100:.1f} | "
            f"Box mAP All: {self.trainer.callback_metrics[f'metrics/{log_prefix}_bbox_ap_all'] * 100:.1f}{reset}"
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["state_dict"] = {
            k.replace("._orig_mod", ""): v for k, v in checkpoint["state_dict"].items()
        }

    def _zero_init_outside_encoder(
        self,
        encoder_prefix: str = "network.encoder.",
        skip_class_head: bool = False,
    ) -> None:
        with torch.no_grad():
            total, zeroed = 0, 0
            for name, param in self.named_parameters():
                total += param.numel()
                if not name.startswith(encoder_prefix):
                    if skip_class_head and (
                        "class_head" in name or "class_predictor" in name
                    ):
                        continue
                    param.zero_()
                    zeroed += param.numel()
            msg = (
                f"Zeroed {zeroed:,} / {total:,} parameters "
                f"(everything not under '{encoder_prefix}'"
            )
            if skip_class_head:
                msg += ", skipping class head"
            msg += ")"
            logging.info(msg)

    def _add_state_dicts(self, state_dict1, state_dict2):
        summed = {}
        for key in state_dict1.keys():
            if key not in state_dict2:
                summed[key] = state_dict1[key]
                continue

            if state_dict1[key].shape != state_dict2[key].shape:
                raise ValueError(
                    f"Shape mismatch at {key}: "
                    f"{state_dict1[key].shape} vs {state_dict2[key].shape}"
                )

            summed[key] = state_dict1[key] + state_dict2[key]

        return summed

    def _load_ckpt(self, ckpt_path, load_ckpt_class_head):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        ckpt = {k: v for k, v in ckpt.items() if "criterion.empty_weight" not in k}
        if not load_ckpt_class_head:
            ckpt = {
                k: v
                for k, v in ckpt.items()
                if "class_head" not in k and "class_predictor" not in k
            }
        logging.info(f"Loaded {len(ckpt)} keys")
        return ckpt

    def _raise_on_incompatible(self, incompatible_keys, load_ckpt_class_head):
        def is_class_or_bbox_head(key: str) -> bool:
            return (
                ("class_head" in key)
                or ("class_predictor" in key)
                or ("bbox_head" in key)
                or ("bbox_predictor" in key)
            )

        if incompatible_keys.missing_keys:
            if not load_ckpt_class_head:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if not is_class_or_bbox_head(key)
                ]
            else:
                missing_keys = [
                    key
                    for key in incompatible_keys.missing_keys
                    if "bbox_head" not in key and "bbox_predictor" not in key
                ]
            if missing_keys:
                raise ValueError(f"Missing keys: {missing_keys}")

        if incompatible_keys.unexpected_keys:
            unexpected_keys = [
                key
                for key in incompatible_keys.unexpected_keys
                if "bbox_head" not in key and "bbox_predictor" not in key
            ]
            if unexpected_keys:
                raise ValueError(f"Unexpected keys: {unexpected_keys}")
