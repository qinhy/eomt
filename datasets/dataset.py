# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import re
import json
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional
from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
    ):
        self.zip_path = zip_path
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_zip_path = target_zip_path
        self.target_zip_path_in_zip = target_zip_path_in_zip
        self.target_instance_zip_path = target_instance_zip_path
        self.target_folder_path_in_zip = target_folder_path_in_zip
        self.target_instance_folder_path_in_zip = target_instance_folder_path_in_zip

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None
        img_zip, target_zip, target_instance_zip = self._load_zips()

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.bboxes_by_id = {}
        self.image_sizes_by_name = {}

        if annotations_json_path_in_zip is not None:
            with zipfile.ZipFile(target_zip_path or zip_path) as outer_target_zip:
                with outer_target_zip.open(
                    str(annotations_json_path_in_zip), "r"
                ) as file:
                    annotation_data = json.load(file)

            image_id_to_file_name = {}
            for image in annotation_data["images"]:
                image_id_to_file_name[image["id"]] = image["file_name"]
                self.image_sizes_by_name[image["file_name"]] = (
                    image["width"],
                    image["height"],
                )

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}

                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}

                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}
                    if img_filename not in self.bboxes_by_id:
                        self.bboxes_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation[
                        "category_id"
                    ]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation[
                        "segmentation"
                    ]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(
                        annotation["iscrowd"]
                    )
                    if "bbox" in annotation:
                        self.bboxes_by_id[img_filename][annotation["id"]] = annotation[
                            "bbox"
                        ]

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        target_zip_filenames = target_zip.namelist()

        for img_info in sorted(img_zip.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info, img_folder_path_in_zip, img_stem_suffix, img_suffix
            ):
                continue

            img_path = Path(img_info.filename)
            if not only_annotations_json:
                rel_path = img_path.relative_to(img_folder_path_in_zip)
                target_parent = target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(img_stem_suffix, target_stem_suffix)

                target_filename = (target_parent / f"{target_stem}{target_suffix}").as_posix()

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue

                if not self.labels_by_id[img_path.name]:
                    continue

                if check_empty_targets and only_annotations_json:
                    width, height = self.image_sizes_by_name[img_path.name]
                    parsed_target = self.target_parser(
                        polygons_by_id=self.polygons_by_id.get(img_path.name, {}),
                        labels_by_id=self.labels_by_id.get(img_path.name, {}),
                        is_crowd_by_id=self.is_crowd_by_id.get(img_path.name, {}),
                        bboxes_by_id=self.bboxes_by_id.get(img_path.name, {}),
                        width=width,
                        height=height,
                    )
                    if not parsed_target[1]:
                        continue
            else:
                if target_filename not in target_zip_filenames:
                    continue

                if check_empty_targets:
                    with target_zip.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            if target_instance_zip is not None:
                target_instance_filename = (
                    target_instance_folder_path_in_zip / (target_stem + target_suffix)
                ).as_posix()

                if check_empty_targets:
                    with target_instance_zip.open(
                        target_instance_filename
                    ) as target_instance:
                        extrema = Image.open(target_instance).getextrema()
                        if all(min_val == max_val for min_val, max_val in extrema):
                            _, labels, _ = self.target_parser(
                                target=tv_tensors.Mask(
                                    Image.open(target_zip.open(target_filename))
                                ),
                                target_instance=tv_tensors.Mask(
                                    Image.open(target_instance)
                                ),
                                stuff_classes=self.stuff_classes,
                            )
                            if not labels:
                                continue

            self.imgs.append(img_path.as_posix())

            if not only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_zip is not None:
                self.targets_instance.append(target_instance_filename)

    def __getitem__(self, index: int):
        img_zip, target_zip, target_instance_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_zip.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_zip.open(
                self.targets_instance[index]
            ) as target_instance:
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance), dtype=torch.long
                )

        parsed_target = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(Path(self.imgs[index]).name, {}),
            labels_by_id=self.labels_by_id.get(Path(self.imgs[index]).name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(Path(self.imgs[index]).name, {}),
            bboxes_by_id=self.bboxes_by_id.get(Path(self.imgs[index]).name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )
        if len(parsed_target) == 3:
            masks, labels, is_crowd = parsed_target
            boxes = None
        elif len(parsed_target) == 4:
            masks, labels, is_crowd, boxes = parsed_target
        else:
            raise ValueError("target_parser must return 3 or 4 values")

        if masks:
            mask_tensor = tv_tensors.Mask(torch.stack(masks))
        else:
            mask_tensor = tv_tensors.Mask(
                torch.empty((0, *img.shape[-2:]), dtype=torch.bool)
            )

        target = {
            "masks": mask_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }
        if boxes is not None:
            target["boxes"] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _load_zips(
        self,
    ) -> Tuple[zipfile.ZipFile, zipfile.ZipFile, Optional[zipfile.ZipFile]]:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)
        if worker not in self.target_zip:
            if self.target_zip_path:
                self.target_zip[worker] = zipfile.ZipFile(self.target_zip_path)
                if self.target_zip_path_in_zip:
                    with self.target_zip[worker].open(
                        str(self.target_zip_path_in_zip)
                    ) as target_zip_stream:
                        nested_zip_data = BytesIO(target_zip_stream.read())
                    self.target_zip[worker].close()
                    self.target_zip[worker] = zipfile.ZipFile(nested_zip_data)
            else:
                self.target_zip[worker] = self.zip[worker]
        if (
            self.target_instance_zip_path is not None
            and worker not in self.target_instance_zip
        ):
            self.target_instance_zip[worker] = zipfile.ZipFile(
                self.target_instance_zip_path
            )

        return (
            self.zip[worker],
            self.target_zip[worker],
            self.target_instance_zip[worker] if self.target_instance_zip_path else None,
        )

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)

        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def valid_member(
        img_info: zipfile.ZipInfo,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        return (
            Path(img_info.filename).is_relative_to(img_folder_path_in_zip)
            and img_info.filename.endswith(img_stem_suffix + img_suffix)
            and not img_info.is_dir()
        )

    def __len__(self):
        return len(self.imgs)

    def close(self):
        if self.zip is not None:
            for item in self.zip.values():
                item.close()
            self.zip = None

        if self.target_zip is not None:
            for item in self.target_zip.values():
                item.close()
            self.target_zip = None

        if self.target_instance_zip is not None:
            for item in self.target_instance_zip.values():
                item.close()
            self.target_instance_zip = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state


# ---------------------------------------------------------------
# Cached dataset manifest wrapper for zip-based datasets
# Derived from the user's Dataset implementation
# ---------------------------------------------------------------

import hashlib
import inspect
import json
import os
import re
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import get_worker_info
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class Dataset(torch.utils.data.Dataset):
    """
    Drop-in replacement for the original Dataset class, with a persistent cache
    for the expensive dataset manifest built during __init__.

    What is cached:
    - labels_by_id
    - polygons_by_id
    - is_crowd_by_id
    - bboxes_by_id
    - image_sizes_by_name
    - imgs
    - targets
    - targets_instance

    What this avoids on later runs:
    - re-parsing the annotations JSON
    - re-scanning and sorting zip members
    - re-building target filename lists
    - re-running empty-target filtering during __init__

    Notes:
    - The first run is still slow because it builds the cache.
    - Later runs load the manifest directly from disk.
    - Cache invalidation uses a fingerprint of:
        * source zip file(s) size + mtime
        * key dataset constructor args
        * best-effort fingerprint of target_parser
    """

    CACHE_FORMAT_VERSION = 1

    def __init__(
        self,
        zip_path: Path,
        img_suffix: str,
        target_parser: Callable,
        check_empty_targets: bool,
        transforms: Optional[Callable] = None,
        only_annotations_json: bool = False,
        target_suffix: str = None,
        stuff_classes: Optional[list[int]] = None,
        img_stem_suffix: str = "",
        target_stem_suffix: str = "",
        target_zip_path: Optional[Path] = None,
        target_zip_path_in_zip: Optional[Path] = None,
        target_instance_zip_path: Optional[Path] = None,
        img_folder_path_in_zip: Path = Path("./"),
        target_folder_path_in_zip: Path = Path("./"),
        target_instance_folder_path_in_zip: Path = Path("./"),
        annotations_json_path_in_zip: Optional[Path] = None,
        # cache-related args
        cache_dir="./dataset_cache",   # optional
        force_rebuild_cache=False,     # set True once if you want to refresh
        verbose_cache=True,            # prints cache hit/build info
        save_cache: bool = True,
    ):
        self.zip_path = Path(zip_path)
        self.target_parser = target_parser
        self.transforms = transforms
        self.only_annotations_json = only_annotations_json
        self.stuff_classes = stuff_classes
        self.target_zip_path = Path(target_zip_path) if target_zip_path is not None else None
        self.target_zip_path_in_zip = (
            Path(target_zip_path_in_zip) if target_zip_path_in_zip is not None else None
        )
        self.target_instance_zip_path = (
            Path(target_instance_zip_path) if target_instance_zip_path is not None else None
        )
        self.target_folder_path_in_zip = Path(target_folder_path_in_zip)
        self.target_instance_folder_path_in_zip = Path(target_instance_folder_path_in_zip)

        self.img_suffix = img_suffix
        self.check_empty_targets = check_empty_targets
        self.target_suffix = target_suffix
        self.img_stem_suffix = img_stem_suffix
        self.target_stem_suffix = target_stem_suffix
        self.img_folder_path_in_zip = Path(img_folder_path_in_zip)
        self.annotations_json_path_in_zip = (
            Path(annotations_json_path_in_zip)
            if annotations_json_path_in_zip is not None
            else None
        )

        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.zip_path.parent / ".dataset_cache"
        self.force_rebuild_cache = force_rebuild_cache
        self.verbose_cache = verbose_cache
        self.save_cache = save_cache

        self.zip = None
        self.target_zip = None
        self.target_instance_zip = None

        self.labels_by_id = {}
        self.polygons_by_id = {}
        self.is_crowd_by_id = {}
        self.bboxes_by_id = {}
        self.image_sizes_by_name = {}

        self.imgs = []
        self.targets = []
        self.targets_instance = []

        if not self.force_rebuild_cache and self._try_load_cache():
            return

        self._build_manifest()

        if self.save_cache:
            self._save_cache()

    # ------------------------
    # Public data API
    # ------------------------

    def __getitem__(self, index: int):
        img_zip, target_zip, target_instance_zip = self._load_zips()

        with img_zip.open(self.imgs[index]) as img:
            img = tv_tensors.Image(Image.open(img).convert("RGB"))

        target = None
        if not self.only_annotations_json:
            with target_zip.open(self.targets[index]) as target_file:
                target = tv_tensors.Mask(Image.open(target_file), dtype=torch.long)

            if img.shape[-2:] != target.shape[-2:]:
                target = F.resize(
                    target,
                    list(img.shape[-2:]),
                    interpolation=F.InterpolationMode.NEAREST,
                )

        target_instance = None
        if self.targets_instance:
            with target_instance_zip.open(self.targets_instance[index]) as target_instance_file:
                target_instance = tv_tensors.Mask(
                    Image.open(target_instance_file), dtype=torch.long
                )

        parsed_target = self.target_parser(
            target=target,
            target_instance=target_instance,
            stuff_classes=self.stuff_classes,
            polygons_by_id=self.polygons_by_id.get(Path(self.imgs[index]).name, {}),
            labels_by_id=self.labels_by_id.get(Path(self.imgs[index]).name, {}),
            is_crowd_by_id=self.is_crowd_by_id.get(Path(self.imgs[index]).name, {}),
            bboxes_by_id=self.bboxes_by_id.get(Path(self.imgs[index]).name, {}),
            width=img.shape[-1],
            height=img.shape[-2],
        )

        if len(parsed_target) == 3:
            masks, labels, is_crowd = parsed_target
            boxes = None
        elif len(parsed_target) == 4:
            masks, labels, is_crowd, boxes = parsed_target
        else:
            raise ValueError("target_parser must return 3 or 4 values")

        if masks:
            mask_tensor = tv_tensors.Mask(torch.stack(masks))
        else:
            mask_tensor = tv_tensors.Mask(
                torch.empty((0, *img.shape[-2:]), dtype=torch.bool)
            )

        target = {
            "masks": mask_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }
        if boxes is not None:
            target["boxes"] = boxes

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def close(self):
        if self.zip is not None:
            for item in self.zip.values():
                item.close()
            self.zip = None

        if self.target_zip is not None:
            for item in self.target_zip.values():
                item.close()
            self.target_zip = None

        if self.target_instance_zip is not None:
            for item in self.target_instance_zip.values():
                item.close()
            self.target_instance_zip = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["zip"] = None
        state["target_zip"] = None
        state["target_instance_zip"] = None
        return state

    # ------------------------
    # Cache internals
    # ------------------------

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "cache_format_version": self.CACHE_FORMAT_VERSION,
            "labels_by_id": self.labels_by_id,
            "polygons_by_id": self.polygons_by_id,
            "is_crowd_by_id": self.is_crowd_by_id,
            "bboxes_by_id": self.bboxes_by_id,
            "image_sizes_by_name": self.image_sizes_by_name,
            "imgs": self.imgs,
            "targets": self.targets,
            "targets_instance": self.targets_instance,
        }

    def _load_manifest_payload(self, payload: dict[str, Any]) -> None:
        if payload.get("cache_format_version") != self.CACHE_FORMAT_VERSION:
            raise RuntimeError("Cache format version mismatch")

        self.labels_by_id = payload["labels_by_id"]
        self.polygons_by_id = payload["polygons_by_id"]
        self.is_crowd_by_id = payload["is_crowd_by_id"]
        self.bboxes_by_id = payload["bboxes_by_id"]
        self.image_sizes_by_name = payload["image_sizes_by_name"]
        self.imgs = payload["imgs"]
        self.targets = payload["targets"]
        self.targets_instance = payload["targets_instance"]

    def _cache_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key()
        zip_stem = self.zip_path.stem
        return self.cache_dir / f"{zip_stem}.{key}.pt"

    def _cache_key(self) -> str:
        payload = {
            "cache_format_version": self.CACHE_FORMAT_VERSION,
            "zip_path": str(self.zip_path.resolve()),
            "target_zip_path": str(self.target_zip_path.resolve()) if self.target_zip_path else None,
            "target_instance_zip_path": (
                str(self.target_instance_zip_path.resolve())
                if self.target_instance_zip_path
                else None
            ),
            "target_zip_path_in_zip": str(self.target_zip_path_in_zip) if self.target_zip_path_in_zip else None,
            "img_suffix": self.img_suffix,
            "target_suffix": self.target_suffix,
            "check_empty_targets": self.check_empty_targets,
            "only_annotations_json": self.only_annotations_json,
            "stuff_classes": self.stuff_classes,
            "img_stem_suffix": self.img_stem_suffix,
            "target_stem_suffix": self.target_stem_suffix,
            "img_folder_path_in_zip": str(self.img_folder_path_in_zip),
            "target_folder_path_in_zip": str(self.target_folder_path_in_zip),
            "target_instance_folder_path_in_zip": str(self.target_instance_folder_path_in_zip),
            "annotations_json_path_in_zip": (
                str(self.annotations_json_path_in_zip)
                if self.annotations_json_path_in_zip is not None
                else None
            ),
            "file_deps": self._file_dependencies_fingerprint(),
            "target_parser": self._fingerprint_callable(self.target_parser),
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:20]

    def _file_dependencies_fingerprint(self) -> dict[str, Any]:
        deps = {}
        for name, path in [
            ("zip_path", self.zip_path),
            ("target_zip_path", self.target_zip_path),
            ("target_instance_zip_path", self.target_instance_zip_path),
        ]:
            deps[name] = self._fingerprint_file(path)
        return deps

    @staticmethod
    def _fingerprint_file(path: Optional[Path]) -> Optional[dict[str, Any]]:
        if path is None:
            return None
        p = Path(path)
        st = p.stat()
        return {
            "path": str(p.resolve()),
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
        }

    @staticmethod
    def _fingerprint_callable(fn: Callable) -> dict[str, Any]:
        out = {
            "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "name": getattr(fn, "__name__", None),
            "source_sha256": None,
        }
        try:
            source = inspect.getsource(fn)
            out["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        except Exception:
            out["source_sha256"] = hashlib.sha256(repr(fn).encode("utf-8")).hexdigest()
        return out

    def _try_load_cache(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path.exists():
            if self.verbose_cache:
                print(f"[Dataset cache] miss: {cache_path}")
            return False

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self._load_manifest_payload(payload)
            if self.verbose_cache:
                print(f"[Dataset cache] hit: {cache_path}")
            return True
        except Exception as exc:
            if self.verbose_cache:
                print(f"[Dataset cache] failed to load {cache_path}: {exc}")
            return False

    def _save_cache(self) -> None:
        cache_path = self._cache_path()
        payload = self._manifest_payload()

        fd, tmp_name = tempfile.mkstemp(
            prefix=cache_path.name + ".",
            suffix=".tmp",
            dir=str(cache_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, cache_path)
            if self.verbose_cache:
                print(f"[Dataset cache] saved: {cache_path}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    # ------------------------
    # Manifest build path
    # ------------------------

    def _build_manifest(self) -> None:
        img_zip, target_zip, target_instance_zip = self._load_zips()

        if self.annotations_json_path_in_zip is not None:
            with zipfile.ZipFile(self.target_zip_path or self.zip_path) as outer_target_zip:
                with outer_target_zip.open(self.annotations_json_path_in_zip.as_posix(), "r") as file:
                    annotation_data = json.load(file)

            image_id_to_file_name = {}
            for image in annotation_data["images"]:
                image_id_to_file_name[image["id"]] = image["file_name"]
                self.image_sizes_by_name[image["file_name"]] = (
                    image["width"],
                    image["height"],
                )

            for annotation in annotation_data["annotations"]:
                img_filename = image_id_to_file_name[annotation["image_id"]]

                if "segments_info" in annotation:
                    self.labels_by_id[img_filename] = {
                        segment_info["id"]: segment_info["category_id"]
                        for segment_info in annotation["segments_info"]
                    }
                    self.is_crowd_by_id[img_filename] = {
                        segment_info["id"]: bool(segment_info["iscrowd"])
                        for segment_info in annotation["segments_info"]
                    }
                else:
                    if img_filename not in self.labels_by_id:
                        self.labels_by_id[img_filename] = {}
                    if img_filename not in self.polygons_by_id:
                        self.polygons_by_id[img_filename] = {}
                    if img_filename not in self.is_crowd_by_id:
                        self.is_crowd_by_id[img_filename] = {}
                    if img_filename not in self.bboxes_by_id:
                        self.bboxes_by_id[img_filename] = {}

                    self.labels_by_id[img_filename][annotation["id"]] = annotation["category_id"]
                    self.polygons_by_id[img_filename][annotation["id"]] = annotation["segmentation"]
                    self.is_crowd_by_id[img_filename][annotation["id"]] = bool(annotation["iscrowd"])

                    if "bbox" in annotation:
                        self.bboxes_by_id[img_filename][annotation["id"]] = annotation["bbox"]

        target_zip_filenames = set(target_zip.namelist()) if target_zip is not None else set()

        for img_info in sorted(img_zip.infolist(), key=self._sort_key):
            if not self.valid_member(
                img_info,
                self.img_folder_path_in_zip,
                self.img_stem_suffix,
                self.img_suffix,
            ):
                continue

            img_path = Path(img_info.filename)
            target_filename = None
            target_stem = None

            if not self.only_annotations_json:
                rel_path = img_path.relative_to(self.img_folder_path_in_zip)
                target_parent = self.target_folder_path_in_zip / rel_path.parent
                target_stem = rel_path.stem.replace(
                    self.img_stem_suffix,
                    self.target_stem_suffix,
                )
                target_filename = (
                    target_parent / f"{target_stem}{self.target_suffix}"
                ).as_posix()

            if self.labels_by_id:
                if img_path.name not in self.labels_by_id:
                    continue

                if not self.labels_by_id[img_path.name]:
                    continue

                if self.check_empty_targets and self.only_annotations_json:
                    width, height = self.image_sizes_by_name[img_path.name]
                    parsed_target = self.target_parser(
                        polygons_by_id=self.polygons_by_id.get(img_path.name, {}),
                        labels_by_id=self.labels_by_id.get(img_path.name, {}),
                        is_crowd_by_id=self.is_crowd_by_id.get(img_path.name, {}),
                        bboxes_by_id=self.bboxes_by_id.get(img_path.name, {}),
                        width=width,
                        height=height,
                    )
                    if not parsed_target[1]:
                        continue
            else:
                if target_filename not in target_zip_filenames:
                    continue

                if self.check_empty_targets:
                    with target_zip.open(target_filename) as target_file:
                        min_val, max_val = Image.open(target_file).getextrema()
                        if min_val == max_val:
                            continue

            target_instance_filename = None
            if target_instance_zip is not None:
                if target_stem is None:
                    rel_path = img_path.relative_to(self.img_folder_path_in_zip)
                    target_stem = rel_path.stem.replace(
                        self.img_stem_suffix,
                        self.target_stem_suffix,
                    )

                target_instance_filename = (
                    self.target_instance_folder_path_in_zip / (target_stem + self.target_suffix)
                ).as_posix()

                if self.check_empty_targets:
                    with target_instance_zip.open(target_instance_filename) as target_instance_file:
                        extrema = Image.open(target_instance_file).getextrema()
                        if all(min_val == max_val for min_val, max_val in extrema):
                            with target_zip.open(target_filename) as target_file:
                                _, labels, _ = self.target_parser(
                                    target=tv_tensors.Mask(Image.open(target_file)),
                                    target_instance=tv_tensors.Mask(Image.open(target_instance_file)),
                                    stuff_classes=self.stuff_classes,
                                )
                            if not labels:
                                continue

            self.imgs.append(img_path.as_posix())

            if not self.only_annotations_json:
                self.targets.append(target_filename)

            if target_instance_filename is not None:
                self.targets_instance.append(target_instance_filename)

        if self.verbose_cache:
            print(
                f"[Dataset cache] built manifest with {len(self.imgs)} samples "
                f"from {self.zip_path.name}"
            )

    # ------------------------
    # Zip loading path
    # ------------------------

    def _load_zips(self) -> Tuple[zipfile.ZipFile, zipfile.ZipFile, Optional[zipfile.ZipFile]]:
        worker = get_worker_info()
        worker = worker.id if worker else None

        if self.zip is None:
            self.zip = {}
        if self.target_zip is None:
            self.target_zip = {}
        if self.target_instance_zip is None and self.target_instance_zip_path:
            self.target_instance_zip = {}

        if worker not in self.zip:
            self.zip[worker] = zipfile.ZipFile(self.zip_path)

        if worker not in self.target_zip:
            if self.target_zip_path:
                self.target_zip[worker] = zipfile.ZipFile(self.target_zip_path)
                if self.target_zip_path_in_zip:
                    with self.target_zip[worker].open(str(self.target_zip_path_in_zip)) as target_zip_stream:
                        nested_zip_data = BytesIO(target_zip_stream.read())
                    self.target_zip[worker].close()
                    self.target_zip[worker] = zipfile.ZipFile(nested_zip_data)
            else:
                self.target_zip[worker] = self.zip[worker]

        if self.target_instance_zip_path is not None and worker not in self.target_instance_zip:
            self.target_instance_zip[worker] = zipfile.ZipFile(self.target_instance_zip_path)

        return (
            self.zip[worker],
            self.target_zip[worker],
            self.target_instance_zip[worker] if self.target_instance_zip_path else None,
        )

    @staticmethod
    def _sort_key(m: zipfile.ZipInfo):
        match = re.search(r"\d+", m.filename)
        return (int(match.group()) if match else float("inf"), m.filename)

    @staticmethod
    def valid_member(
        img_info: zipfile.ZipInfo,
        img_folder_path_in_zip: Path,
        img_stem_suffix: str,
        img_suffix: str,
    ):
        return (
            Path(img_info.filename).is_relative_to(img_folder_path_in_zip)
            and img_info.filename.endswith(img_stem_suffix + img_suffix)
            and not img_info.is_dir()
        )
    
