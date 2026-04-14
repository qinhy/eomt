from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from training.base_module import TrainModule


class _OneArgNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        return x + self.scale


class TrainModuleForwardTest(unittest.TestCase):
    def test_forward_supports_networks_with_single_input(self):
        module = TrainModule(
            network=_OneArgNetwork(),
            img_size=(8, 8),
            num_classes=1,
            attn_mask_annealing_enabled=False,
            attn_mask_annealing_start_steps=None,
            attn_mask_annealing_end_steps=None,
            lr=1e-4,
            llrd=1.0,
            llrd_l2_enabled=False,
            lr_mult=1.0,
            weight_decay=0.0,
            poly_power=0.9,
            warmup_steps=(0, 0),
        )

        imgs = torch.zeros((1, 3, 2, 2), dtype=torch.uint8)
        output = module(imgs)

        self.assertTrue(torch.allclose(output, torch.ones_like(output)))


if __name__ == "__main__":
    unittest.main()
