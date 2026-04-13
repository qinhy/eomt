# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


from pathlib import Path
from typing import Union

import numpy as np
import torch
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader, Subset
from torchvision import tv_tensors

from datasets.base import DataModule
from datasets.transforms import Transforms
from datasets.zip_dataset import ZipDataset

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
INV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
COCO_CLASS_NAMES = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}

def label_to_name(label: int, bg_idx: int = len(CLASS_MAPPING)) -> str:
    label = int(label)
    if bg_idx is not None and label == bg_idx:
        return "none-object"
    real_cls_id = INV_CLASS_MAPPING.get(label, label)
    return COCO_CLASS_NAMES.get(real_cls_id, f"{real_cls_id}")

class COCOInstance(DataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 16,
        img_size: tuple[int, int] = (640, 640),
        num_classes: int = 80,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.1, 2.0),
        check_empty_targets: bool = True,
        cache_dir: str | Path = "./dataset_cache",
        force_rebuild_cache: bool = False,
        verbose_cache: bool = True,
        save_cache: bool = True,
        overfit_indices: list[int] | None = None,
        overfit_use_train_for_val: bool = True,
        overfit_repeat: int = 1,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.overfit_indices = overfit_indices
        self.overfit_repeat = overfit_repeat
        self.overfit_use_train_for_val = overfit_use_train_for_val

        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        self.cache_kwargs = {
            "cache_dir": cache_dir,
            "force_rebuild_cache": force_rebuild_cache,
            "verbose_cache": verbose_cache,
            "save_cache": save_cache,
        }

    @staticmethod
    def target_parser(
        polygons_by_id: dict[int, object],
        labels_by_id: dict[int, int],
        is_crowd_by_id: dict[int, bool],
        bboxes_by_id: dict[int, list[float]],
        width: int,
        height: int,
        **kwargs,
    ):
        masks, labels, is_crowd, boxes = [], [], [], []

        for ann_id, cls_id in labels_by_id.items():
            if cls_id not in CLASS_MAPPING:
                continue
            if is_crowd_by_id[ann_id]:
                continue

            segmentation = polygons_by_id[ann_id]
            if isinstance(segmentation, dict):
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

    def setup(self, stage: Union[str, None] = None):
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_parser": self.target_parser,
            "only_annotations_json": True,
            "check_empty_targets": self.check_empty_targets,
            **self.cache_kwargs,
        }

        self.train_dataset = ZipDataset(
            transforms=self.transforms,
            img_folder_path_in_zip=Path("train2017"),
            annotations_json_path_in_zip=Path("annotations/instances_train2017.json"),
            target_zip_path=self.path / "annotations_trainval2017.zip",
            zip_path=self.path / "train2017.zip",
            **dataset_kwargs,
        )
        self.val_dataset = ZipDataset(
            img_folder_path_in_zip=Path("val2017"),
            annotations_json_path_in_zip=Path("annotations/instances_val2017.json"),
            target_zip_path=self.path / "annotations_trainval2017.zip",
            zip_path=self.path / "val2017.zip",
            **dataset_kwargs,
        )
        if self.overfit_indices is not None:
            base_train = Subset(self.train_dataset, self.overfit_indices)
            self.train_dataset = base_train

            if self.overfit_repeat > 1:
                self.train_dataset = torch.utils.data.ConcatDataset(
                    [base_train] * self.overfit_repeat
                )

            self.val_dataset = Subset(self.train_dataset.datasets[0] if isinstance(self.train_dataset, torch.utils.data.ConcatDataset) else self.train_dataset,
                                    range(len(self.overfit_indices)))
        return self

    def train_dataloader(self):
        drop_last = False if self.overfit_indices is not None else True
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=drop_last,
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
    def draw_one(img, masks, labels, is_crowd, boxes, scores=None):
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

        def to_bool_mask(mask, threshold=0.5, from_logits=False):
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().float().cpu().numpy()
            else:
                mask = np.asarray(mask)

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 2:
                raise ValueError(f"each mask must be 2D, got {mask.shape}")

            if mask.dtype == np.bool_:
                return mask

            if from_logits:
                mask = 1.0 / (1.0 + np.exp(-mask))

            return mask > threshold

        def to_boxes_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.detach().float().cpu().numpy()
            if hasattr(x, "as_subclass"):
                return torch.as_tensor(x).detach().float().cpu().numpy()
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
            boxes_np = None
        else:
            boxes_np = to_boxes_numpy(boxes).reshape(-1, 4)

        masks_seq = list(masks) if torch.is_tensor(masks) and masks.ndim == 3 else masks
        labels_seq = (
            labels.detach().cpu().tolist()
            if isinstance(labels, torch.Tensor)
            else list(labels)
        )
        scores_seq = (
            scores.detach().cpu().tolist()
            if isinstance(scores, torch.Tensor)
            else (list(scores) if scores is not None else None)
        )
        crowd_seq = (
            is_crowd.detach().cpu().tolist()
            if isinstance(is_crowd, torch.Tensor)
            else (list(is_crowd) if is_crowd is not None else None)
        )

        for i in range(len(labels_seq)):
            mask = to_bool_mask(masks_seq[i])
            color = color_for_label(labels_seq[i]).astype(np.float32)

            if mask.any():
                out[mask] = out[mask] * 0.5 + color * 0.5

            if boxes_np is not None and i < len(boxes_np):
                x1, y1, x2, y2 = boxes_np[i].astype(int).tolist()
                cv2.rectangle(out, (x1, y1), (x2, y2), color.tolist(), 2)

                label_name = label_to_name(labels_seq[i])
                text = label_name
                if scores_seq is not None:
                    text += f" {scores_seq[i]:.2f}"
                if crowd_seq is not None and crowd_seq[i]:
                    text += " crowd"

                (tw, th), baseline = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                )
                y_text = max(0, y1 - th - baseline - 2)
                cv2.rectangle(
                    out,
                    (x1, y_text),
                    (x1 + tw + 4, y_text + th + baseline + 4),
                    color.tolist(),
                    thickness=-1,
                )
                cv2.putText(
                    out,
                    text,
                    (x1 + 2, y_text + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return np.clip(out, 0, 255).astype(np.uint8)
