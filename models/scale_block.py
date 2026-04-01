# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import math
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import LayerNorm2d


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d):
        super().__init__()

        self.conv1 = conv1_layer(
            embed_dim,
            embed_dim,
            kernel_size=2,
            stride=2,
        )
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x

class FSRCNN(nn.Module):
    """
    FSRCNN architecture matching the common yjn870 implementation:
    FSRCNN(scale_factor=2, num_channels=1, d=56, s=12, m=4)
    """
    def __init__(self, scale_factor: int, num_channels: int = 1, d: int = 56, s: int = 12, m: int = 4):
        super().__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d),
        )

        mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            mid_part.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=3 // 2),
                nn.PReLU(s),
            ])
        mid_part.extend([
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),
        ])
        self.mid_part = nn.Sequential(*mid_part)

        self.last_part = nn.ConvTranspose2d(
            d,
            num_channels,
            kernel_size=9,
            stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=math.sqrt(2 / (m.out_channels * m.weight[0][0].numel())),
                )
                nn.init.zeros_(m.bias)

        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(
                    m.weight,
                    mean=0.0,
                    std=math.sqrt(2 / (m.out_channels * m.weight[0][0].numel())),
                )
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.last_part.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class FSRCNNx2YOnlyRGBWrapper(nn.Module):
    """
    Wraps a 1-channel pretrained FSRCNN x2 model so it can accept RGB input.

    Input:
        x: [N, 3, H, W] float tensor in [0, 1]

    Output:
        [N, 3, 2H, 2W] float tensor in [0, 1]
    """
    def __init__(self, y_model: nn.Module):
        super().__init__()
        self.y_model = y_model.eval()
        for p in self.y_model.parameters():
            p.requires_grad = False

    @staticmethod
    def _rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
        # Uses the same formulas as the yjn870 repo utils.py, but in torch.
        # x: [N, 3, H, W] in [0, 1]
        x255 = x * 255.0
        r = x255[:, 0:1]
        g = x255[:, 1:2]
        b = x255[:, 2:3]

        y  = 16.0  + (64.738 * r + 129.057 * g +  25.064 * b) / 256.0
        cb = 128.0 + (-37.945 * r -  74.494 * g + 112.439 * b) / 256.0
        cr = 128.0 + (112.439 * r -  94.154 * g -  18.285 * b) / 256.0
        return torch.cat([y, cb, cr], dim=1)

    @staticmethod
    def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
        # Inverse formulas from the same repo utils.py
        y = ycbcr[:, 0:1]
        cb = ycbcr[:, 1:2]
        cr = ycbcr[:, 2:3]

        r = 298.082 * y / 256.0 + 408.583 * cr / 256.0 - 222.921
        g = 298.082 * y / 256.0 - 100.291 * cb / 256.0 - 208.120 * cr / 256.0 + 135.576
        b = 298.082 * y / 256.0 + 516.412 * cb / 256.0 - 276.836

        rgb255 = torch.cat([r, g, b], dim=1).clamp(0.0, 255.0)
        return rgb255 / 255.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected [N, 3, H, W], got {tuple(x.shape)}")

        x = x.clamp(0.0, 1.0)

        ycbcr = self._rgb_to_ycbcr(x)
        y = ycbcr[:, 0:1] / 255.0       # model expects Y in [0, 1]
        cbcr = ycbcr[:, 1:3]            # keep in [0, 255] domain

        y_sr = self.y_model(y).clamp(0.0, 1.0) * 255.0
        out_h, out_w = y_sr.shape[-2:]

        cbcr_sr = F.interpolate(
            cbcr,
            size=(out_h, out_w),
            mode="bicubic",
            align_corners=False,
        )

        out_ycbcr = torch.cat([y_sr, cbcr_sr], dim=1)
        out_rgb = self._ycbcr_to_rgb(out_ycbcr)
        return out_rgb


def _extract_state_dict(ckpt_obj):
    """
    Accepts a few common checkpoint formats:
    - plain state_dict
    - {'state_dict': ...}
    - {'model_state_dict': ...}
    - full nn.Module
    """
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()

    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt_obj)}")

    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        ckpt_obj = ckpt_obj["state_dict"]
    elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
        ckpt_obj = ckpt_obj["model_state_dict"]

    state_dict = {}
    for k, v in ckpt_obj.items():
        if not torch.is_tensor(v):
            continue
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        state_dict[nk] = v
    return state_dict


def build_or_load_fsrcnn_x2(
    checkpoint_path: str | None = None,
    device: str | torch.device = "cpu",
    wrap_rgb: bool = True,
) -> nn.Module:
    """
    Returns a ready-to-use x2 FSRCNN model.

    - If wrap_rgb=True:
        input  : [N, 3, H, W] in [0,1]
        output : [N, 3, 2H, 2W] in [0,1]

    - If wrap_rgb=False:
        input  : [N, 1, H, W] in [0,1]
        output : [N, 1, 2H, 2W] in [0,1]
    """
    # Pretrained x2 weights from yjn870 use the 1-channel model shape.
    y_model = FSRCNN(scale_factor=2, num_channels=1, d=56, s=12, m=4)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        y_model.load_state_dict(state_dict, strict=True)

    y_model.eval()

    model = FSRCNNx2YOnlyRGBWrapper(y_model) if wrap_rgb else y_model
    model = model.to(device).eval()

    for p in model.parameters():
        p.requires_grad = False

    return model