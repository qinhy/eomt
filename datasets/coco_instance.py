# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union
import numpy as np
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from pycocotools import mask as coco_mask
import torch

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.dataset import Dataset

CLASS_MAPPING = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


class COCOInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled=False,
        scale_range=(0.1, 2.0),
        check_empty_targets=True,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, object],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        bboxes_by_id: dict[int, list[float]],
        width: int,
        height: int,
        **kwargs
    ):
        masks, labels, is_crowd, boxes = [], [], [], []

        for ann_id, cls_id in labels_by_id.items():
            if cls_id not in CLASS_MAPPING:
                continue

            # Best practical choice for training
            if is_crowd_by_id[ann_id]:
                continue

            segmentation = polygons_by_id[ann_id]

            # polygon or RLE
            if isinstance(segmentation, dict):
                # already RLE-like
                rle = segmentation
            else:
                rles = coco_mask.frPyObjects(segmentation, height, width)
                rle = coco_mask.merge(rles) if isinstance(rles, list) else rles

            mask = coco_mask.decode(rle)
            if mask.ndim == 3:
                mask = mask.any(axis=2)

            masks.append(tv_tensors.Mask(torch.as_tensor(mask, dtype=torch.bool)))
            labels.append(CLASS_MAPPING[cls_id])
            is_crowd.append(False)

            x, y, w, h = bboxes_by_id[ann_id]
            boxes.append([x, y, x + w, y + h])

        boxes = tv_tensors.BoundingBoxes(
            torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            format="XYXY",
            canvas_size=(height, width),
        )

        return masks, labels, is_crowd, boxes

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_parser": self.target_parser,
            "only_annotations_json": True,
            "check_empty_targets": self.check_empty_targets,
        }
        self.train_dataset = Dataset(
            transforms=self.transforms,
            img_folder_path_in_zip="train2017",
            annotations_json_path_in_zip="annotations/instances_train2017.json",
            target_zip_path=Path(self.path, "annotations_trainval2017.zip"),
            zip_path=Path(self.path, "train2017.zip"),
            **dataset_kwargs,
        )
        self.val_dataset = Dataset(
            img_folder_path_in_zip="val2017",
            annotations_json_path_in_zip="annotations/instances_val2017.json",
            target_zip_path=Path(self.path, "annotations_trainval2017.zip"),
            zip_path=Path(self.path, "val2017.zip"),
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
    
    @staticmethod
    def draw_one(img, masks, labels, is_crowd, boxes):
        """
        Args:
            img:
                torch.Tensor or np.ndarray
                shape: (C,H,W) or (H,W,C)
                value range: [0,1] or [0,255]
            masks:
                list[tv_tensors.Mask] | torch.Tensor
                each mask shape: (H,W)
            labels:
                list[int] | torch.Tensor
            is_crowd:
                list[bool] | torch.Tensor
            boxes:
                tv_tensors.BoundingBoxes | torch.Tensor
                shape: (N,4), XYXY

        Returns:
            np.ndarray: RGB uint8 image, shape (H,W,3)
        """
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenCV is only required for COCOInstance.draw_one(). "
                "Install `opencv-python` to use this visualization helper."
            ) from exc

        def to_numpy_image(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()

                if x.ndim != 3:
                    raise ValueError(f"img must be 3D, got {tuple(x.shape)}")

                # CHW -> HWC
                if x.shape[0] in (1, 3):
                    x = x.permute(1, 2, 0)

                x = x.numpy()

            else:
                x = np.asarray(x)

            if x.ndim != 3:
                raise ValueError(f"img must be 3D, got {x.shape}")

            if x.dtype != np.uint8:
                x = x.astype(np.float32)
                if x.max() <= 1.0:
                    x = x * 255.0
                x = np.clip(x, 0, 255).astype(np.uint8)

            if x.shape[2] == 1:
                x = np.repeat(x, 3, axis=2)

            return x.copy()

        def to_bool_mask(mask):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()
            else:
                mask = np.asarray(mask)

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError(f"each mask must be 2D, got {mask.shape}")

            return mask.astype(bool)

        def to_boxes_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            if hasattr(x, "as_subclass"):  # tv_tensors
                return torch.as_tensor(x).detach().cpu().numpy()
            return np.asarray(x)

        def color_for_label(label: int):
            palette = np.array(
                [
                    [230, 25, 75],
                    [60, 180, 75],
                    [255, 225, 25],
                    [0, 130, 200],
                    [245, 130, 48],
                    [145, 30, 180],
                    [70, 240, 240],
                    [240, 50, 230],
                    [210, 245, 60],
                    [250, 190, 190],
                    [0, 128, 128],
                    [230, 190, 255],
                    [170, 110, 40],
                    [255, 250, 200],
                    [128, 0, 0],
                    [170, 255, 195],
                    [128, 128, 0],
                    [255, 215, 180],
                    [0, 0, 128],
                    [128, 128, 128],
                ],
                dtype=np.uint8,
            )
            return palette[int(label) % len(palette)]

        img = to_numpy_image(img)
        out = img.astype(np.float32)

        if boxes is None:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
        else:
            boxes_np = to_boxes_numpy(boxes).reshape(-1, 4)

        if labels is None:
            labels = [0] * len(masks)
        elif isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()

        if is_crowd is None:
            is_crowd = [False] * len(masks)
        elif isinstance(is_crowd, torch.Tensor):
            is_crowd = is_crowd.detach().cpu().tolist()

        alpha = 0.45

        for i, mask in enumerate(masks):
            mask = to_bool_mask(mask)
            if not mask.any():
                continue

            label = int(labels[i])
            crowd = bool(is_crowd[i])
            color = color_for_label(label).astype(np.float32)

            # mask overlay
            out[mask] = out[mask] * (1.0 - alpha) + color * alpha

            # contour
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color.tolist(), 2)

            # bbox
            if i < len(boxes_np):
                x1, y1, x2, y2 = boxes_np[i]
                x1, y1, x2, y2 = map(lambda v: int(round(float(v))), (x1, y1, x2, y2))

                thickness = 1 if crowd else 2
                cv2.rectangle(out, (x1, y1), (x2, y2), color.tolist(), thickness)

                text = f"{label}"
                if crowd:
                    text += " crowd"

                (tw, th), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                ty = max(y1, th + 4)

                cv2.rectangle(
                    out,
                    (x1, ty - th - 4),
                    (x1 + tw + 4, ty + baseline - 2),
                    color.tolist(),
                    -1,
                )
                cv2.putText(
                    out,
                    text,
                    (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return np.clip(out, 0, 255).astype(np.uint8)