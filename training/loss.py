# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional, Tuple
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import tv_tensors
from torchvision.ops import box_convert, generalized_box_iou, generalized_box_iou_loss

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerHungarianMatcher,
    Mask2FormerLoss,
)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [..., 4] in normalized cxcywh
    returns [..., 4] in normalized xyxy
    """
    cx, cy, w, h = boxes.unbind(dim=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h

    x_min = torch.minimum(x0, x1)
    y_min = torch.minimum(y0, y1)
    x_max = torch.maximum(x0, x1)
    y_max = torch.maximum(y0, y1)

    return torch.stack([x_min, y_min, x_max, y_max], dim=-1).clamp(0, 1)

class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        bbox_l1_coefficient: float = 0.0,
        bbox_giou_coefficient: float = 0.0,
        owner_coefficient: float = 0.0,
        owner_ignore_index: int = 255,
    ):
        # We intentionally do not call Mask2FormerLoss.__init__ because you are
        # using a manual constructor layout rather than HF config + weight_dict.
        nn.Module.__init__(self)

        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.bbox_l1_coefficient = bbox_l1_coefficient
        self.bbox_giou_coefficient = bbox_giou_coefficient
        self.owner_coefficient = owner_coefficient
        self.owner_ignore_index = owner_ignore_index

        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    def get_num_instances(self, class_labels: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Normalize exactly like mask loss should be normalized:
        average number of GT instances across all ranks, clamped to >= 1.
        """
        num_instances = sum(len(labels) for labels in class_labels)
        num_instances = torch.as_tensor(num_instances, dtype=torch.float, device=device)

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_instances)
            num_instances = num_instances / dist.get_world_size()

        return torch.clamp(num_instances, min=1.0)

    def _prepare_box_labels(
        self,
        targets: List[dict],
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[torch.Tensor]:
        """
        Assumes target['boxes'] is absolute pixel xyxy.
        Converts to normalized cxcywh in [0, 1].
        """
        box_labels = []
        for target in targets:
            boxes_xyxy: tv_tensors.BoundingBoxes = target["boxes"]
            h, w = boxes_xyxy.canvas_size
            scale = torch.tensor([w, h, w, h], device=device, dtype=dtype)

            boxes_xyxy = boxes_xyxy.to(device=device, dtype=dtype)
            boxes_cxcywh = box_convert(boxes_xyxy / scale, "xyxy", "cxcywh").clamp(0, 1)
            box_labels.append(boxes_cxcywh)

        return box_labels

    def build_owner_targets(
        self,
        targets: List[dict],
        indices,
        height: int,
        width: int,
        num_queries: int,
        device: torch.device,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build per-pixel owner targets for owner_queries_logits.

        Output shape: [B, H, W]
        Values:
          0..Q-1 : query id that owns the pixel
          Q      : background
          ignore_index : ambiguous overlapping GT pixels
        """
        if ignore_index is None:
            ignore_index = self.owner_ignore_index

        batch_size = len(targets)
        bg_id = num_queries

        owner_targets = torch.full(
            (batch_size, height, width),
            fill_value=bg_id,
            dtype=torch.long,
            device=device,
        )

        for b, (target, (src_idx, tgt_idx)) in enumerate(zip(targets, indices)):
            if src_idx.numel() == 0:
                continue

            gt_masks = target["masks"].to(device=device, dtype=torch.float32)  # [N_gt, H0, W0]

            if gt_masks.shape[-2:] != (height, width):
                gt_masks = F.interpolate(
                    gt_masks[:, None],
                    size=(height, width),
                    mode="nearest",
                )[:, 0]

            gt_masks = gt_masks > 0.5
            matched_masks = gt_masks[tgt_idx]  # [M, H, W]

            # Ambiguous pixels where multiple GT instances overlap.
            # Those cannot be represented by hard owner assignment, so ignore them.
            ambiguous = matched_masks.sum(dim=0) > 1
            if ambiguous.any():
                owner_targets[b][ambiguous] = ignore_index

            for q_idx, mask in zip(src_idx.tolist(), matched_masks):
                owner_targets[b][mask & (~ambiguous)] = q_idx

        return owner_targets

    def loss_owner(
        self,
        owner_queries_logits: torch.Tensor,  # [B, Q+1, H, W]
        targets: List[dict],
        indices,
        ignore_index: Optional[int] = None,
    ):
        """
        Per-pixel owner CE across queries + background.

        This is the key loss for non-overlap-by-design:
        each pixel chooses one query id or background.
        """
        if ignore_index is None:
            ignore_index = self.owner_ignore_index

        zero = owner_queries_logits.sum() * 0.0

        if owner_queries_logits is None:
            return {"loss_owner": zero}

        if owner_queries_logits.ndim != 4:
            raise ValueError(
                f"owner_queries_logits must have shape [B, Q+1, H, W], got {tuple(owner_queries_logits.shape)}"
            )

        batch_size, q_plus_bg, height, width = owner_queries_logits.shape
        num_queries = q_plus_bg - 1

        if num_queries <= 0:
            return {"loss_owner": zero}

        owner_targets = self.build_owner_targets(
            targets=targets,
            indices=indices,
            height=height,
            width=width,
            num_queries=num_queries,
            device=owner_queries_logits.device,
            ignore_index=ignore_index,
        )

        valid = owner_targets != ignore_index
        if not valid.any():
            return {"loss_owner": zero}

        loss_owner = F.cross_entropy(
            owner_queries_logits,
            owner_targets,
            ignore_index=ignore_index,
        )

        return {"loss_owner": loss_owner}

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: torch.Tensor,
        bbox_queries_preds: Optional[torch.Tensor] = None,
        owner_queries_logits: Optional[torch.Tensor] = None,
    ):
        device = masks_queries_logits.device
        dtype = masks_queries_logits.dtype

        # move labels/masks to the same device
        mask_labels = [t["masks"].to(device=device, dtype=dtype) for t in targets]
        class_labels = [t["labels"].to(device=device, dtype=torch.long) for t in targets]

        has_boxes = bbox_queries_preds is not None and all("boxes" in t for t in targets)
        box_labels = (
            self._prepare_box_labels(targets, device=device, dtype=dtype)
            if has_boxes
            else None
        )

        if bbox_queries_preds is not None:
            bbox_queries_preds = bbox_queries_preds.to(device=device, dtype=dtype)

        if owner_queries_logits is not None:
            owner_queries_logits = owner_queries_logits.to(device=device, dtype=dtype)

        indices = self.matcher.forward(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        num_instances = self.get_num_instances(class_labels, device=device)

        losses = {}

        loss_masks = super().loss_masks(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            indices=indices,
            num_masks=num_instances,
        )

        loss_labels = super().loss_labels(
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
            indices=indices,
        )

        losses.update(loss_masks)
        losses.update(loss_labels)

        if has_boxes:
            loss_boxes = self.loss_boxes(
                bbox_queries_preds=bbox_queries_preds,
                box_labels=box_labels,
                indices=indices,
                num_instances=num_instances,
            )
            losses.update(loss_boxes)

        if owner_queries_logits is not None:
            loss_owner = self.loss_owner(
                owner_queries_logits=owner_queries_logits,
                targets=targets,
                indices=indices,
            )
            losses.update(loss_owner)

        return losses

    def loss_boxes(
        self,
        bbox_queries_preds: torch.Tensor,   # [B, Q, 4], normalized cxcywh
        box_labels: list[torch.Tensor],     # list[[Ni, 4]], normalized cxcywh
        indices,
        num_instances: torch.Tensor,
    ):
        src_idx = self._get_predictions_permutation_indices(indices)
        zero = bbox_queries_preds.sum() * 0.0

        if src_idx[0].numel() == 0:
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = bbox_queries_preds[src_idx]
        target_boxes = torch.cat(
            [target[j] for target, (_, j) in zip(box_labels, indices)],
            dim=0,
        ).to(src_boxes)

        # 1) Drop pairs that are already NaN / Inf in cxcywh
        finite_cxcywh = (
            torch.isfinite(src_boxes).all(dim=1) &
            torch.isfinite(target_boxes).all(dim=1)
        )

        if not finite_cxcywh.any():
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = src_boxes[finite_cxcywh]
        target_boxes = target_boxes[finite_cxcywh]

        # 2) Convert to xyxy and clamp to image range
        src_xyxy = box_convert(src_boxes, "cxcywh", "xyxy").clamp(0, 1)
        tgt_xyxy = box_convert(target_boxes, "cxcywh", "xyxy").clamp(0, 1)

        # 3) Keep only valid xyxy pairs
        valid_xyxy = (
            torch.isfinite(src_xyxy).all(dim=1) &
            torch.isfinite(tgt_xyxy).all(dim=1) &
            (src_xyxy[:, 2] > src_xyxy[:, 0]) &
            (src_xyxy[:, 3] > src_xyxy[:, 1]) &
            (tgt_xyxy[:, 2] > tgt_xyxy[:, 0]) &
            (tgt_xyxy[:, 3] > tgt_xyxy[:, 1])
        )

        if not valid_xyxy.any():
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = src_boxes[valid_xyxy]
        target_boxes = target_boxes[valid_xyxy]
        src_xyxy = src_xyxy[valid_xyxy]
        tgt_xyxy = tgt_xyxy[valid_xyxy]

        # L1 in cxcywh
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="sum")

        # GIoU in xyxy
        loss_giou_vec = generalized_box_iou_loss(
            src_xyxy.float(),
            tgt_xyxy.float(),
            reduction="none",
            eps=1e-7,
        )

        finite_giou = torch.isfinite(loss_giou_vec)
        loss_giou = loss_giou_vec[finite_giou].sum() if finite_giou.any() else zero

        denom = num_instances.clamp_min(1)
        return {
            "loss_bbox": loss_bbox / denom,
            "loss_giou": loss_giou / denom,
        }

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        total = None
        progress_bar_totals = {
            "mask": None,
            "dice": None,
            "cls": None,
            "bbox": None,
            "giou": None,
            "owner": None,
        }

        for loss_key, loss in losses_all_layers.items():
            log_fn(
                f"losses/train_{loss_key}",
                loss.detach(),
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

            if "loss_mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
                progress_key = "mask"
            elif "loss_dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
                progress_key = "dice"
            elif "loss_cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
                progress_key = "cls"
            elif "loss_bbox" in loss_key:
                weighted_loss = loss * self.bbox_l1_coefficient
                progress_key = "bbox"
            elif "loss_giou" in loss_key:
                weighted_loss = loss * self.bbox_giou_coefficient
                progress_key = "giou"
            elif "loss_owner" in loss_key:
                weighted_loss = loss * self.owner_coefficient
                progress_key = "owner"
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            detached_weighted_loss = weighted_loss.detach()
            current_total = progress_bar_totals[progress_key]
            progress_bar_totals[progress_key] = (
                detached_weighted_loss
                if current_total is None
                else current_total + detached_weighted_loss
            )

            total = weighted_loss if total is None else total + weighted_loss

        log_fn(
            "loss_total",
            total.detach(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        for progress_key, value in progress_bar_totals.items():
            if value is None:
                continue
            log_fn(
                progress_key,
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
                prog_bar=True,
                logger=False,
            )

        log_fn(
            "loss",
            total.detach(),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
            prog_bar=True,
            logger=False,
        )

        return total  

