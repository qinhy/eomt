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
        owner_coefficient: float = 1.0,
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
            owner_coefficient=owner_coefficient,
        )

        num_metric_blocks = self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1
        self.init_metrics_instance(num_metric_blocks)

    def _save_train_visualization(self, batch, step: int) -> None:
        imgs, targets = batch
        random_idx = int(torch.randint(len(imgs), (1,), device=self.device).item())
        img = imgs[random_idx]
        target = targets[random_idx]

        mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer, owner_logits_per_layer = self.forward(img.unsqueeze(0))
        mask_logits, class_logits, bbox_preds, owner_logits = (mask_logits_per_layer[-1], class_logits_per_layer[-1],
                                                               bbox_preds_per_layer[-1], owner_logits_per_layer[-1])
        
        pred_without_background = self.network.predict(
                    mask_logits, class_logits, bbox_preds, owner_logits, img.shape[-2:],
                    top_k=self.eval_top_k_instances, ignore_bg=True, score_threshold=None)
        pred_with_background = self.network.predict(
                    mask_logits, class_logits, bbox_preds, owner_logits, img.shape[-2:],
                    top_k=self.eval_top_k_instances, ignore_bg=False, score_threshold=0.3)

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
            (gt_vis, pred_vis_without_background, pred_vis),
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
            img_size = img.shape[-2:] # (C, H, W)
            mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer, owner_logits_per_layer = self(
                img.unsqueeze(0)
            )
            target_for_metrics = self._target_for_metrics(target)

            for block_idx, (mask_logits, class_logits, bbox_preds, owner_logits) in enumerate(
                zip(mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer, owner_logits_per_layer)
            ):
                pred = self.network.predict(
                    mask_logits, class_logits, bbox_preds, owner_logits, img_size)
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
