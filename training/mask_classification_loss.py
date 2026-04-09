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
from torchvision.ops import box_convert, generalized_box_iou

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
    sample_point,
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

class BoxAwareMask2FormerHungarianMatcher(nn.Module):
    """
    Mask2Former-style Hungarian matcher with optional bbox L1 + GIoU costs.

    Supports:
      - mask-only matching
      - box-only matching
      - hybrid mask+box matching

    Expected box format:
      - predictions: normalized cxcywh, shape [B, Q, 4]
      - labels: list of [Ni, 4] normalized cxcywh
    """

    def __init__(
        self,
        num_points: int = 12544,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        cost_bbox: float = 0.0,
        cost_giou: float = 0.0,
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("All matcher costs cannot be zero.")

        self.num_points = num_points
        self.cost_class = float(cost_class)
        self.cost_mask = float(cost_mask)
        self.cost_dice = float(cost_dice)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)

    def _validate_targets(
        self,
        class_labels: List[torch.Tensor],
        mask_labels: Optional[List[torch.Tensor]],
        box_labels: Optional[List[torch.Tensor]],
    ) -> None:
        if mask_labels is not None and len(mask_labels) != len(class_labels):
            raise ValueError("mask_labels and class_labels must have the same batch size")
        if box_labels is not None and len(box_labels) != len(class_labels):
            raise ValueError("box_labels and class_labels must have the same batch size")

        for batch_idx, labels in enumerate(class_labels):
            num_targets = labels.shape[0]

            if mask_labels is not None and mask_labels[batch_idx].shape[0] != num_targets:
                raise ValueError(
                    f"mask_labels[{batch_idx}] has {mask_labels[batch_idx].shape[0]} instances "
                    f"but class_labels[{batch_idx}] has {num_targets}"
                )

            if box_labels is not None and box_labels[batch_idx].shape[0] != num_targets:
                raise ValueError(
                    f"box_labels[{batch_idx}] has {box_labels[batch_idx].shape[0]} instances "
                    f"but class_labels[{batch_idx}] has {num_targets}"
                )

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: Optional[torch.Tensor],
        class_queries_logits: torch.Tensor,
        mask_labels: Optional[List[torch.Tensor]],
        class_labels: List[torch.Tensor],
        bbox_queries_preds: Optional[torch.Tensor] = None,
        box_labels: Optional[List[torch.Tensor]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        self._validate_targets(class_labels, mask_labels, box_labels)

        use_masks = (self.cost_mask > 0 or self.cost_dice > 0)
        use_boxes = (self.cost_bbox > 0 or self.cost_giou > 0)

        if use_masks and (masks_queries_logits is None or mask_labels is None):
            raise ValueError("Mask costs are enabled, but masks_queries_logits or mask_labels is missing.")
        if use_boxes and (bbox_queries_preds is None or box_labels is None):
            raise ValueError("Box costs are enabled, but bbox_queries_preds or box_labels is missing.")

        batch_size = class_queries_logits.shape[0]
        device = class_queries_logits.device
        indices: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for i in range(batch_size):
            target_classes = class_labels[i].to(device=device, dtype=torch.long)
            num_targets = target_classes.numel()

            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.int64, device=device)
                indices.append((empty, empty))
                continue

            pred_probs = class_queries_logits[i].softmax(-1)  # [Q, C+1]
            cost_matrix = self.cost_class * (-pred_probs[:, target_classes])  # [Q, N]

            if use_masks:
                pred_mask = masks_queries_logits[i][:, None]  # [Q, 1, H, W]
                tgt_mask = mask_labels[i].to(device=pred_mask.device, dtype=pred_mask.dtype)[:, None]  # [N, 1, H, W]

                point_coords = torch.rand(1, self.num_points, 2, device=device)

                pred_points = sample_point(
                    pred_mask,
                    point_coords.repeat(pred_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)  # [Q, P]

                tgt_points = sample_point(
                    tgt_mask,
                    point_coords.repeat(tgt_mask.shape[0], 1, 1),
                    align_corners=False,
                ).squeeze(1)  # [N, P]

                if self.cost_mask > 0:
                    cost_matrix = cost_matrix + self.cost_mask * pair_wise_sigmoid_cross_entropy_loss(
                        pred_points, tgt_points
                    )

                if self.cost_dice > 0:
                    cost_matrix = cost_matrix + self.cost_dice * pair_wise_dice_loss(pred_points, tgt_points)

            if use_boxes:
                pred_boxes = bbox_queries_preds[i]  # [Q, 4]
                tgt_boxes = box_labels[i].to(device=pred_boxes.device, dtype=pred_boxes.dtype)  # [N, 4]

                if self.cost_bbox > 0:
                    cost_matrix = cost_matrix + self.cost_bbox * torch.cdist(pred_boxes.float(), tgt_boxes.float(), p=1)

                if self.cost_giou > 0:
                    pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
                    tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
                    cost_matrix = cost_matrix + self.cost_giou * (-generalized_box_iou(pred_xyxy, tgt_xyxy))

            cost_matrix = torch.nan_to_num(cost_matrix, nan=0.0, posinf=1e10, neginf=-1e10)
            cost_matrix = cost_matrix.clamp(min=-1e10, max=1e10)

            src_idx, tgt_idx = linear_sum_assignment(cost_matrix.detach().cpu())
            indices.append(
                (
                    torch.as_tensor(src_idx, dtype=torch.int64, device=device),
                    torch.as_tensor(tgt_idx, dtype=torch.int64, device=device),
                )
            )

        return indices

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

        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = BoxAwareMask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
            cost_bbox=bbox_l1_coefficient,
            cost_giou=bbox_giou_coefficient,
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
            boxes_xyxy:tv_tensors.BoundingBoxes = target["boxes"]
            h, w = boxes_xyxy.canvas_size
            scale = torch.tensor([w, h, w, h], device=device, dtype=dtype)

            boxes_xyxy = boxes_xyxy.to(device=device, dtype=dtype)
            boxes_cxcywh = box_convert(boxes_xyxy / scale, "xyxy", "cxcywh").clamp(0, 1)
            box_labels.append(boxes_cxcywh)

        return box_labels

    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: torch.Tensor,
        bbox_queries_preds: Optional[torch.Tensor] = None,
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

        # IMPORTANT:
        # bbox_queries_preds must already be normalized cxcywh in [0,1].
        # If your bbox head outputs raw values, pass bbox_queries_preds.sigmoid().
        if bbox_queries_preds is not None:
            bbox_queries_preds = bbox_queries_preds.to(device=device, dtype=dtype)

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            bbox_queries_preds=bbox_queries_preds if has_boxes else None,
            box_labels=box_labels,
        )

        num_instances = self.get_num_instances(class_labels, device=device)

        losses = {}
        loss_masks = super().loss_masks(
                masks_queries_logits=masks_queries_logits,
                mask_labels=mask_labels,
                indices=indices,
                num_masks=num_instances,
        )
        loss_labels = self.loss_labels(
                class_queries_logits=class_queries_logits,
                class_labels=class_labels,
                indices=indices,
            )
        loss_boxes = self.loss_boxes(
                    bbox_queries_preds=bbox_queries_preds,
                    box_labels=box_labels,
                    indices=indices,
                    num_instances=num_instances,
        )
        losses.update(loss_masks)
        losses.update(loss_labels)
        if has_boxes:
            losses.update(loss_boxes)

        for key, value in losses.items():
            if not torch.isfinite(value):
                losses[key] = torch.tensor(0.0, device=device, dtype=dtype)
                                
        return losses

    def loss_boxes(
        self,
        bbox_queries_preds: torch.Tensor,   # [B, Q, 4], normalized cxcywh
        box_labels: List[torch.Tensor],     # list[[Ni, 4]], normalized cxcywh
        indices,
        num_instances: torch.Tensor,
    ):
        src_idx = self._get_predictions_permutation_indices(indices)

        if src_idx[0].numel() == 0:
            zero = bbox_queries_preds.sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = bbox_queries_preds[src_idx]
        target_boxes = torch.cat(
            [target[j] for target, (_, j) in zip(box_labels, indices)],
            dim=0,
        ).to(src_boxes)

        # L1 in cxcywh
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum()

        # GIoU in xyxy
        src_xyxy = box_convert(src_boxes, "cxcywh", "xyxy").clamp(0, 1)
        tgt_xyxy = box_convert(target_boxes, "cxcywh", "xyxy").clamp(0, 1)
        loss_giou = (1.0 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).sum()

        return {
            "loss_bbox": loss_bbox / num_instances,
            "loss_giou": loss_giou / num_instances,
        }

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        total = None
        progress_bar_totals = {
            "mask": None,
            "dice": None,
            "cls": None,
            "bbox": None,
            "giou": None,
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
