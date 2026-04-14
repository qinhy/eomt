from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torchvision.transforms import v2 as T, InterpolationMode
from torchvision.tv_tensors import TVTensor, wrap


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        zoom_out_p: float = 0.3,
        zoom_out_range: tuple[float, float] = (1.0, 1.5),
        iou_crop_enabled: bool = True,
        blur_p: float = 0.10,
        grayscale_p: float = 0.05,
        small_affine_p: float = 0.15,
        affine_translate: tuple[float, float] = (0.05, 0.05),
        affine_scale: tuple[float, float] = (0.9, 1.1),
        max_retries: int = 10,
        fill: int | float = 0,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.max_retries = max_retries

        common_ops: list[nn.Module] = []

        if color_jitter_enabled:
            common_ops.append(
                T.RandomPhotometricDistort(
                    brightness=(
                        1.0 - max_brightness_delta / 255.0,
                        1.0 + max_brightness_delta / 255.0,
                    ),
                    contrast=(
                        1.0 - max_contrast_factor,
                        1.0 + max_contrast_factor,
                    ),
                    saturation=(
                        1.0 - saturation_factor,
                        1.0 + saturation_factor,
                    ),
                    hue=(
                        -max_hue_delta / 360.0,
                        max_hue_delta / 360.0,
                    ),
                    p=0.5,
                )
            )

        if blur_p > 0:
            common_ops.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                    p=blur_p,
                )
            )

        if grayscale_p > 0:
            common_ops.append(T.RandomGrayscale(p=grayscale_p))

        self.common = T.Compose(common_ops)

        tail_ops: list[nn.Module] = []

        if small_affine_p > 0:
            tail_ops.append(
                T.RandomApply(
                    [
                        T.RandomAffine(
                            degrees=0.0,
                            translate=affine_translate,
                            scale=affine_scale,
                            fill=fill,
                        )
                    ],
                    p=small_affine_p,
                )
            )

        tail_ops.extend(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ScaleJitter(target_size=img_size, scale_range=scale_range),
                T.RandomCrop(size=img_size, pad_if_needed=True, fill=fill),
            ]
        )

        with_boxes_ops: list[nn.Module] = [
            T.RandomZoomOut(fill=fill, side_range=zoom_out_range, p=zoom_out_p),
        ]
        if iou_crop_enabled:
            with_boxes_ops.append(
                T.RandomIoUCrop(
                    min_scale=0.3,
                    max_scale=1.0,
                    min_aspect_ratio=0.5,
                    max_aspect_ratio=2.0,
                    trials=40,
                )
            )
        with_boxes_ops.extend(tail_ops)

        no_boxes_ops: list[nn.Module] = [
            T.RandomZoomOut(fill=fill, side_range=zoom_out_range, p=zoom_out_p),
            *tail_ops,
        ]

        self.with_boxes = T.Compose(with_boxes_ops)
        self.no_boxes = T.Compose(no_boxes_ops)

        self.clamp_boxes = T.ClampBoundingBoxes()
        self.sanitize_boxes = T.SanitizeBoundingBoxes(
            labels_getter=self._labels_getter
        )

    def _labels_getter(self, inputs: Any):
        _, target = inputs

        aligned = []
        for key in ("labels", "area", "iscrowd", "is_crowd"):
            value = target.get(key)
            if torch.is_tensor(value) and value.ndim > 0:
                aligned.append(value)

        return aligned

    def _num_instances(self, target: dict[str, Any]) -> int:
        for key in ("boxes", "masks", "labels", "iscrowd", "is_crowd", "area"):
            value = target.get(key)
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 0

    def _filter_instances(
        self,
        target: dict[str, Any],
        keep: Tensor,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for key, value in target.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == keep.numel():
                sliced = value[keep]
                out[key] = wrap(sliced, like=value) if isinstance(value, TVTensor) else sliced
            else:
                out[key] = value

        return out

    def _drop_crowd(self, target: dict[str, Any]) -> dict[str, Any]:
        crowd_key = "iscrowd" if "iscrowd" in target else "is_crowd" if "is_crowd" in target else None
        if crowd_key is None:
            return target

        keep = ~target[crowd_key].to(torch.bool)
        return self._filter_instances(target, keep)

    def _has_nonempty_boxes(self, target: dict[str, Any]) -> bool:
        boxes = target.get("boxes")
        return torch.is_tensor(boxes) and boxes.ndim > 0 and boxes.shape[0] > 0

    def _has_instances(self, target: dict[str, Any]) -> bool:
        if "boxes" in target and torch.is_tensor(target["boxes"]):
            return target["boxes"].shape[0] > 0

        if "masks" in target and torch.is_tensor(target["masks"]):
            return target["masks"].shape[0] > 0

        return self._num_instances(target) > 0

    def forward(
        self,
        img: Tensor,
        target: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        target = self._drop_crowd(target.copy())
        target_was_empty = not self._has_instances(target)

        last_img, last_target = img, target

        for _ in range(self.max_retries):
            out_img, out_target = self.common(img, target)

            aug = self.with_boxes if self._has_nonempty_boxes(out_target) else self.no_boxes
            out_img, out_target = aug(out_img, out_target)

            if "boxes" in out_target:
                out_img, out_target = self.clamp_boxes(out_img, out_target)
                out_img, out_target = self.sanitize_boxes(out_img, out_target)

            last_img, last_target = out_img, out_target

            if target_was_empty or self._has_instances(out_target):
                return out_img, out_target

        return last_img, last_target


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        fill: int | float = 0,
        
        # never use
        color_jitter_enabled=None,
        scale_range=None,
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.fill = fill

    def forward(
        self,
        img: Tensor,
        target: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        target = target.copy()

        h, w = img.shape[-2:]

        if h != w:
            side = max(h, w)
            pad_w = side - w
            pad_h = side - h

            left = int(torch.randint(0, pad_w + 1, ()).item()) if pad_w > 0 else 0
            top = int(torch.randint(0, pad_h + 1, ()).item()) if pad_h > 0 else 0
            right = pad_w - left
            bottom = pad_h - top

            img = T.functional.pad(img, padding=[left, top, right, bottom], fill=self.fill)

            boxes = target.get("boxes")
            if boxes is not None and torch.is_tensor(boxes):
                new_boxes = boxes.clone()
                new_boxes[..., 0::2] += left
                new_boxes[..., 1::2] += top
                if isinstance(boxes, TVTensor):
                    new_boxes = wrap(new_boxes, like=boxes, canvas_size=(side, side))
                target["boxes"] = new_boxes

            masks = target.get("masks")
            if masks is not None and torch.is_tensor(masks):
                target["masks"] = T.functional.pad(
                    masks, padding=[left, top, right, bottom], fill=0
                )

            src_h, src_w = side, side
        else:
            src_h, src_w = h, w

        scale_x = self.img_size[1] / src_w
        scale_y = self.img_size[0] / src_h

        img = T.functional.resize(
            img,
            size=self.img_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        boxes = target.get("boxes")
        if boxes is not None and torch.is_tensor(boxes):
            new_boxes = boxes.clone()
            new_boxes[..., 0::2] *= scale_x
            new_boxes[..., 1::2] *= scale_y
            if isinstance(boxes, TVTensor):
                new_boxes = wrap(new_boxes, like=boxes, canvas_size=self.img_size)
            target["boxes"] = new_boxes

        masks = target.get("masks")
        if masks is not None and torch.is_tensor(masks):
            target["masks"] = T.functional.resize(
                masks,
                size=self.img_size,
                interpolation=InterpolationMode.NEAREST,
            )

        if "area" in target and torch.is_tensor(target["area"]):
            target["area"] = target["area"] * (scale_x * scale_y)

        return img, target
