# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Mask2Former repository
# by Facebook, Inc. and its affiliates, used under the Apache 2.0 License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import box_convert, masks_to_boxes

from datasets.coco_instance import COCOInstance
from models import EoMT
from training.mask_classification_loss import MaskClassificationLoss
from training.lightning_module import LightningModule


class MaskClassificationInstance(LightningModule):
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
    ):
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

        self.save_hyperparameters(ignore=["_class_path","network"])

        self.mask_thresh = mask_thresh
        self.overlap_thresh = overlap_thresh
        self.stuff_classes: List[int] = []
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

        self.init_metrics_instance(self.network.num_blocks + 1 if self.network.masked_attn_enabled else 1)

    def _empty_prediction(self, img_size: tuple[int, int], device: torch.device):
        return dict(
            masks=torch.zeros((0, *img_size), dtype=torch.bool, device=device),
            labels=torch.zeros((0,), dtype=torch.long, device=device),
            scores=torch.zeros((0,), dtype=torch.float32, device=device),
            boxes=torch.zeros((0, 4), dtype=torch.float32, device=device),
        )

    def _predict_instances_for_visualization(
        self,
        img: torch.Tensor,
        top_k: int = 10,
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
        )
        class_logits = class_logits_per_layer[-1]
        bbox_preds = bbox_preds_per_layer[-1]

        scores = class_logits[0].softmax(dim=-1)[:, :-1]
        top_k = min(top_k, self.eval_top_k_instances, scores.numel())
        if top_k == 0:
            return self._empty_prediction(img_size, img.device)

        labels = (
            torch.arange(scores.shape[-1], device=scores.device)
            .unsqueeze(0)
            .repeat(scores.shape[0], 1)
            .flatten(0, 1)
        )

        topk_scores, topk_indices = scores.flatten(0, 1).topk(top_k, sorted=False)
        labels = labels[topk_indices]

        query_indices = topk_indices // scores.shape[-1]
        mask_logits_img = mask_logits[0][query_indices]

        masks = mask_logits_img > 0
        mask_scores = (
            mask_logits_img.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        scores = topk_scores * mask_scores

        order = scores.argsort(descending=True)
        masks = masks[order]
        labels = labels[order]
        scores = scores[order]

        if bbox_preds is not None:
            bbox_preds_img = bbox_preds[0][query_indices][order]
            boxes = box_convert(bbox_preds_img, "cxcywh", "xyxy").clamp(0, 1)
            scale = torch.tensor(
                [img_size[1], img_size[0], img_size[1], img_size[0]],
                device=boxes.device,
                dtype=boxes.dtype,
            )
            boxes = boxes * scale
        elif len(masks) > 0:
            boxes = masks_to_boxes(masks.float())
        else:
            boxes = torch.zeros((0, 4), device=img.device, dtype=torch.float32)

        return dict(
            masks=masks,
            labels=labels,
            scores=scores,
            boxes=boxes,
        )

    def _save_train_visualization(self, batch, step: int):
        imgs, targets = batch
        random_idx = int(torch.randint(len(imgs), (1,), device=self.device).item())
        img = imgs[random_idx]
        target = targets[random_idx]

        pred = self._predict_instances_for_visualization(img)
        target_boxes = target.get("boxes")
        if target_boxes is None:
            target_boxes = masks_to_boxes(target["masks"].float())

        gt_vis = COCOInstance.draw_one(
            img,
            target["masks"],
            target["labels"],
            target["is_crowd"],
            target_boxes,
        )
        pred_vis = COCOInstance.draw_one(
            img,
            pred["masks"],
            pred["labels"],
            None,
            pred["boxes"],
        )
        pair_vis = np.concatenate((gt_vis, pred_vis), axis=1)

        log_dir = getattr(self.logger, "log_dir", None)
        save_dir = Path(log_dir) if log_dir is not None else Path(self.trainer.default_root_dir)
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

        img_sizes = [img.shape[-2:] for img in imgs]
        transformed_imgs = self.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = self(
            transformed_imgs
        )

        for i, (mask_logits, class_logits, bbox_preds) in enumerate(
            list(zip(mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer))
        ):
            mask_logits = F.interpolate(mask_logits, self.img_size, mode="bilinear")
            mask_logits = self.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            preds, targets_ = [], []
            for j in range(len(mask_logits)):
                scores = class_logits[j].softmax(dim=-1)[:, :-1]
                labels = (
                    torch.arange(scores.shape[-1], device=self.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                topk_indices = topk_indices // scores.shape[-1]
                mask_logits[j] = mask_logits[j][topk_indices]

                masks = mask_logits[j] > 0
                mask_scores = (
                    mask_logits[j].sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores
                if bbox_preds is not None:
                    bbox_preds_j = bbox_preds[j][topk_indices]
                    boxes = box_convert(bbox_preds_j, "cxcywh", "xyxy").clamp(0, 1)
                    scale = torch.tensor(
                        [
                            img_sizes[j][1],
                            img_sizes[j][0],
                            img_sizes[j][1],
                            img_sizes[j][0],
                        ],
                        device=boxes.device,
                        dtype=boxes.dtype,
                    )
                    boxes = boxes * scale
                else:
                    boxes = masks_to_boxes(masks.float())

                preds.append(
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                        boxes=boxes,
                    )
                )
                if "boxes" in targets[j]:
                    target_boxes = targets[j]["boxes"].to(
                        device=self.device, dtype=torch.float32
                    )
                else:
                    target_boxes = masks_to_boxes(targets[j]["masks"].float())
                targets_.append(
                    dict(
                        masks=targets[j]["masks"],
                        labels=targets[j]["labels"],
                        boxes=target_boxes,
                        iscrowd=targets[j]["is_crowd"],
                    )
                )

            self.update_metrics_instance(preds, targets_, i)
    
    def eval_step(
        self,
        batch,
        batch_idx=None,
        log_prefix=None,
    ):
        imgs, targets = batch

        for img, target in zip(imgs, targets):
            img_size = img.shape[-2:]  # (H, W)

            # batch size 1
            img_in = img.unsqueeze(0)

            mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = self(img_in)

            for i, (mask_logits, class_logits, bbox_preds) in enumerate(
                zip(mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer)
            ):
                # resize predicted masks directly to original image size
                mask_logits = F.interpolate(
                    mask_logits,
                    size=img_size,
                    mode="bilinear",
                    align_corners=False,
                )

                scores = class_logits[0].softmax(dim=-1)[:, :-1]

                labels = (
                    torch.arange(scores.shape[-1], device=scores.device)
                    .unsqueeze(0)
                    .repeat(scores.shape[0], 1)
                    .flatten(0, 1)
                )

                topk_scores, topk_indices = scores.flatten(0, 1).topk(
                    self.eval_top_k_instances, sorted=False
                )
                labels = labels[topk_indices]

                query_indices = topk_indices // scores.shape[-1]
                mask_logits_img = mask_logits[0][query_indices]

                masks = mask_logits_img > 0
                mask_scores = (
                    mask_logits_img.sigmoid().flatten(1) * masks.flatten(1)
                ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                scores = topk_scores * mask_scores

                if bbox_preds is not None:
                    bbox_preds_img = bbox_preds[0][query_indices]
                    boxes = box_convert(bbox_preds_img, "cxcywh", "xyxy").clamp(0, 1)

                    scale = torch.tensor(
                        [img_size[1], img_size[0], img_size[1], img_size[0]],
                        device=boxes.device,
                        dtype=boxes.dtype,
                    )
                    boxes = boxes * scale
                else:
                    boxes = masks_to_boxes(masks.float())

                preds = [
                    dict(
                        masks=masks,
                        labels=labels,
                        scores=scores,
                        boxes=boxes,
                    )
                ]

                if "boxes" in target:
                    target_boxes = target["boxes"].to(
                        device=self.device,
                        dtype=torch.float32,
                    )
                else:
                    target_boxes = masks_to_boxes(target["masks"].float())

                targets_ = [
                    dict(
                        masks=target["masks"],
                        labels=target["labels"],
                        boxes=target_boxes,
                        iscrowd=target["is_crowd"],
                    )
                ]

                self.update_metrics_instance(preds, targets_, i)

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
