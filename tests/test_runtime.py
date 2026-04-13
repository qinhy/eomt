from __future__ import annotations

import unittest
from pathlib import Path

import torch

from training.checkpointing import resolve_run_dir, sanitize_state_dict
from training.runtime import build_runtime, resolve_amp, to_scalar


class RuntimeUtilsTest(unittest.TestCase):
    def test_to_scalar_handles_common_inputs(self):
        self.assertEqual(to_scalar(torch.tensor(3.5)), 3.5)
        self.assertEqual(to_scalar(True), 1.0)
        self.assertIsNone(to_scalar(torch.tensor([1.0, 2.0])))

    def test_sanitize_state_dict_strips_compile_prefix(self):
        state_dict = {"model._orig_mod.weight": torch.tensor([1.0])}
        sanitized = sanitize_state_dict(state_dict)
        self.assertIn("model.weight", sanitized)

    def test_build_runtime_and_resolve_run_dir(self):
        run_dir = resolve_run_dir(
            Path("/tmp/logs"),
            "exp",
            Path("/tmp/logs/exp/checkpoints/last.pt"),
        )
        runtime = build_runtime(total_steps=123, run_dir=run_dir)

        self.assertEqual(run_dir, Path("/tmp/logs/exp"))
        self.assertEqual(runtime.estimated_stepping_batches, 123)
        self.assertEqual(runtime.default_root_dir, "/tmp/logs/exp")

    def test_resolve_amp_cpu_disables_amp(self):
        amp_enabled, amp_dtype, use_scaler = resolve_amp(
            torch.device("cpu"),
            "bf16-true",
        )
        self.assertFalse(amp_enabled)
        self.assertIsNone(amp_dtype)
        self.assertFalse(use_scaler)


if __name__ == "__main__":
    unittest.main()
