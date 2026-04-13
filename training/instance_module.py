from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import box_convert, masks_to_boxes

from datasets.coco_instance import COCOInstance
from models import EoMT
from training.base_module import TrainModule, compiler_disable
from training.loss import MaskClassificationLoss


class MaskClassificationInstance(TrainModule):
    def __init__(
        self,
        network: EoMT,
        img_size: tuple[int, int],
        num_classes: int,
        attn_mask_annealing_enabled: bool,
        attn_mask_annealing_start_steps: Optional[list[int]] = None,
        attn_mask_annealing_end_steps: Optional[list[int]] = None,
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
        poly_power: float = 0.9,
        warmup_steps: List[int] = [500, 1000],
        no_object_coefficient: float = 0.1,
        mask_coefficient: float = 5.0,
        dice_coefficient: float = 5.0,
        class_coefficient: float = 2.0,
        bbox_l1_coefficient: float = 5.0,
        bbox_giou_coefficient: float = 2.0,
        mask_thresh: float = 0.8,
        overlap_thresh: float = 0.8,
        eval_top_k_instances: int = 100,
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
        load_ckpt_class_head: bool = True,
    ) -> None:
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_classes,
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=load_ckpt_class_head,
        )

        self.save_hyperparameters(ignore=["_class_path", "network"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.eval_top_k_instances = eval_top_k_instances
        self.train_vis_interval = 100
        self.train_vis_dirname = "train_vis"
        self._train_vis_failed = False

        self.criterion = MaskClassificationLoss(
            num_points=num_points,
            oversample_ratio=oversample_ratio,
            importance_sample_ratio=importance_sample_ratio,
            mask_coefficient=mask_coefficient,
            dice_coefficient=dice_coefficient,
            class_coefficient=class_coefficient,
            bbox_l1_coefficient=bbox_l1_coefficient,
            bbox_giou_coefficient=bbox_giou_coefficient,
            num_labels=num_classes,
            no_object_coefficient=no_object_coefficient,
        )

        num_metric_blocks = self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        self.init_metrics_instance(num_metric_blocks)

    def _empty_prediction(self, img_size: tuple[int, int], device: torch.device):
        return {
            "masks": torch.zeros((0, *img_size), dtype=torch.bool, device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "scores": torch.zeros((0,), device=device),
            "boxes": torch.zeros((0, 4), device=device),
        }

    @compiler_disable
    def _predict_instances_from_layer(
        self,
        mask_logits: torch.Tensor,          # [Q, H, W]
        class_logits: torch.Tensor,         # [Q, C+1], bg is last class
        bbox_preds: torch.Tensor | None,    # [Q, 4] or None
        img_size: tuple[int, int],
        *,
        top_k: int | None = None,
        ignore_bg: bool = False,
        score_threshold: float | None = None,
        mask_threshold: float = 0.0,
    ) -> dict[str, torch.Tensor]:

        device = class_logits.device
        bg_idx = class_logits.shape[-1] - 1

        probs = class_logits.softmax(dim=-1)     # [Q, C+1]
        obj_probs = probs[:, :-1]                # [Q, C]

        # one label per query: best non-bg class
        obj_scores, labels = obj_probs.max(dim=-1)   # [Q], [Q]
        pred_all = probs.argmax(dim=-1)              # [Q]

        # Mode A: keep all queries, but rank by best object-class score
        # Mode B: first remove queries predicted as bg
        keep = torch.ones_like(obj_scores, dtype=torch.bool)
        if ignore_bg:
            keep = pred_all != bg_idx

        if score_threshold is not None:
            keep = keep & (obj_scores >= score_threshold)

        if keep.sum() == 0:
            return self._empty_prediction(img_size, device)

        mask_logits = mask_logits[keep]
        obj_scores = obj_scores[keep]
        labels = labels[keep]
        if bbox_preds is not None:
            bbox_preds = bbox_preds[keep]

        k = top_k if top_k is not None else self.eval_top_k_instances
        k = min(k, obj_scores.numel())
        if k == 0:
            return self._empty_prediction(img_size, device)

        topk_scores, topk_idx = obj_scores.topk(k, largest=True, sorted=False)

        mask_logits = mask_logits[topk_idx]
        labels = labels[topk_idx]
        if bbox_preds is not None:
            bbox_preds = bbox_preds[topk_idx]

        # convert masks for visualization / mask scoring
        masks = mask_logits > mask_threshold

        mask_scores = (
            mask_logits.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)

        scores = topk_scores * mask_scores

        order = scores.argsort(descending=False)
        masks = masks[order]
        labels = labels[order]
        scores = scores[order]

        if bbox_preds is not None:
            boxes = box_convert(bbox_preds[order], "cxcywh", "xyxy").clamp(0, 1)
            scale = torch.tensor(
                [img_size[1], img_size[0], img_size[1], img_size[0]],
                device=boxes.device,
                dtype=boxes.dtype,
            )
            boxes = boxes * scale
        elif len(masks) > 0:
            boxes = masks_to_boxes(masks)
        else:
            boxes = torch.zeros((0, 4), device=device)

        return {
            "masks": masks,
            "labels": labels,
            "scores": scores,
            "boxes": boxes,
        }

    def _target_for_metrics(self, target: dict) -> dict:
        if "boxes" in target:
            target_boxes = target["boxes"].to(device=self.device)
        else:
            target_boxes = masks_to_boxes(target["masks"])
        return {
            "masks": target["masks"],
            "labels": target["labels"],
            "boxes": target_boxes,
            "iscrowd": target["is_crowd"],
        }

    def _predict_instances_for_visualization(
        self,
        img: torch.Tensor,
        top_k: int = 10,
        ignore_bg: bool = False,
    ):
        img_size = img.shape[-2:]
        mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = self(
            img.unsqueeze(0)
        )

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1],
            size=img_size,
            mode="bilinear",
            align_corners=False,
        )[0]
        class_logits = class_logits_per_layer[-1][0]
        bbox_preds = bbox_preds_per_layer[-1][0]

        return self._predict_instances_from_layer(
            mask_logits,
            class_logits,
            bbox_preds,
            img_size,
            top_k=top_k,
            ignore_bg=ignore_bg,
        )

    def _save_train_visualization(self, batch, step: int) -> None:
        imgs, targets = batch
        random_idx = int(torch.randint(len(imgs), (1,), device=self.device).item())
        img = imgs[random_idx]
        target = targets[random_idx]

        pred_without_background = self._predict_instances_for_visualization(
            img,ignore_bg=True)
        pred_with_background = self._predict_instances_for_visualization(
            img,ignore_bg=False)

        target_boxes = target.get("boxes")
        if target_boxes is None:
            target_boxes = masks_to_boxes(target["masks"])

        gt_vis = COCOInstance.draw_one(
            img,
            target["masks"],
            target["labels"],
            target["is_crowd"],
            target_boxes,
        )
        pred_vis = COCOInstance.draw_one(
            img,
            pred_with_background["masks"],
            pred_with_background["labels"],
            None,
            pred_with_background["boxes"],
            pred_with_background["scores"],
        )
        pred_vis_without_background = COCOInstance.draw_one(
            img,
            pred_without_background["masks"],
            pred_without_background["labels"],
            None,
            pred_without_background["boxes"],
            pred_without_background["scores"],
        )
        pair_vis = np.concatenate(
            (gt_vis, pred_vis, pred_vis_without_background),
            axis=1,
        )

        log_dir = getattr(self.logger, "log_dir", None)
        save_dir = (
            Path(log_dir) if log_dir is not None else Path(self.trainer.default_root_dir)
        )
        save_dir = save_dir / self.train_vis_dirname
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pair_vis).save(save_dir / f"step_{step:07d}.png")

    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        for img, target in zip(imgs, targets):
            img_size = img.shape[-2:]
            mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = self(
                img.unsqueeze(0)
            )
            target_for_metrics = self._target_for_metrics(target)

            for block_idx, (mask_logits, class_logits, bbox_preds) in enumerate(
                zip(mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer)
            ):
                mask_logits = F.interpolate(
                    mask_logits,
                    size=img_size,
                    mode="bilinear",
                    align_corners=False,
                )[0]
                pred = self._predict_instances_from_layer(
                    mask_logits,
                    class_logits[0],
                    bbox_preds[0] if bbox_preds is not None else None,
                    img_size,
                )
                self.update_metrics_instance([pred], [target_for_metrics], block_idx)

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx=None,
        dataloader_idx=None,
    ):
        super().on_train_batch_end(outputs, batch, batch_idx, dataloader_idx)

        step = int(self.global_step)
        if step == 0 or step % self.train_vis_interval != 0:
            return
        if not self.trainer.is_global_zero or self._train_vis_failed:
            return

        was_training = self.training
        try:
            self.eval()
            with torch.inference_mode():
                self._save_train_visualization(batch, step)
        except Exception as exc:
            self._train_vis_failed = True
            print(f"Skipping train visualization after step {step}: {exc}")
        finally:
            if was_training:
                self.train()

    def on_validation_epoch_end(self):
        self._on_eval_epoch_end_instance("val")

    def on_validation_end(self):
        self._on_eval_end_instance("val")
