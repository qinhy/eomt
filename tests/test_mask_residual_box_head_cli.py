from __future__ import annotations

import unittest
from pathlib import Path

import torch

from scripts.train_mask_residual_box_head import build_instance_batch, parse_args


class MaskResidualBoxHeadCliTest(unittest.TestCase):
    def test_parse_args_without_path_validation(self):
        args = parse_args(
            [
                "--data-path",
                "/tmp/coco",
                "--accelerator",
                "cpu",
            ],
            validate_paths=False,
        )

        self.assertEqual(args.img_size, (640, 640))
        self.assertEqual(args.data_path, Path("/tmp/coco").resolve())
        self.assertEqual(args.cache_dir.name, "dataset_cache")
        self.assertEqual(args.precision, "32-true")

    def test_build_instance_batch_uses_binary_masks_for_boxes(self):
        target = {
            "masks": torch.tensor(
                [
                    [
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 0, 0],
                    ],
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 0, 0],
                    ],
                ],
                dtype=torch.bool,
            )
        }

        mask_batch, box_batch = build_instance_batch([target], device=torch.device("cpu"))

        self.assertEqual(mask_batch.shape, (2, 1, 3, 3))
        expected = torch.tensor(
            [
                [2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
            ],
            dtype=box_batch.dtype,
        )
        self.assertTrue(torch.allclose(box_batch, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
