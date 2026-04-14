from __future__ import annotations

import unittest
from pathlib import Path

from scripts.train_coco_instance import parse_args


class TrainCliTest(unittest.TestCase):
    def test_parse_args_without_path_validation(self):
        args = parse_args(["--data-path", "/tmp/coco"], validate_paths=False)

        self.assertEqual(args.img_size, (640, 640))
        self.assertEqual(args.warmup_steps, (2000, 3000))
        self.assertEqual(args.data_path, Path("/tmp/coco").resolve())
        self.assertEqual(args.cache_dir.name, "dataset_cache")
        self.assertEqual(args.network_impl, "dinov3")

    def test_parse_args_supports_original_bbox_mode(self):
        args = parse_args(
            [
                "--data-path",
                "/tmp/coco",
                "--network-impl",
                "original_bbox",
                "--official-delta-ckpt",
                "/tmp/official.bin",
            ],
            validate_paths=False,
        )

        self.assertEqual(args.network_impl, "original_bbox")
        self.assertEqual(args.official_delta_ckpt, Path("/tmp/official.bin").resolve())

    def test_parse_args_rejects_mismatched_anneal_steps(self):
        with self.assertRaises(ValueError):
            parse_args(
                ["--data-path", "/tmp/coco", "--num-blocks", "3"],
                validate_paths=False,
            )


if __name__ == "__main__":
    unittest.main()
