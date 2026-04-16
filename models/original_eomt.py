import torch
import torch.nn as nn
from torchvision.ops import box_convert, masks_to_boxes
from models.eomt import EoMT
 
class MaskResidualBoxHead(nn.Module):
    def __init__(self, hidden=64, delta_scale=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.delta_scale = delta_scale
        self._coord_cache = {}

    def get_coords(self, H, W, device, dtype, N):
        key = (H, W, device.type, device.index, str(dtype))
        if key not in self._coord_cache:
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device, dtype=dtype),
                torch.linspace(-1, 1, W, device=device, dtype=dtype),
                indexing="ij",
            )
            coord = torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]
            self._coord_cache[key] = coord
        return self._coord_cache[key].expand(N, -1, -1, -1)

    def forward(self, mask_logits, boxes_cxcywh=None):
        B, Q, H, W = mask_logits.shape
        N = B * Q

        if boxes_cxcywh is None:
            with torch.no_grad():
                masks = mask_logits.reshape(N, H, W) > 0.0
                empty_mask = ~masks.flatten(1).any(dim=1)

                base_boxes = torch.zeros(
                    (N, 4),
                    device=mask_logits.device,
                    dtype=mask_logits.dtype,
                )

                if (~empty_mask).any():
                    non_empty_boxes = masks_to_boxes(masks[~empty_mask])
                    base_boxes[~empty_mask] = non_empty_boxes.to(
                        mask_logits.device,
                        mask_logits.dtype,
                    )

                base_boxes[:, [0, 2]] /= max(W - 1, 1)
                base_boxes[:, [1, 3]] /= max(H - 1, 1)
                boxes_cxcywh = box_convert(
                    base_boxes,
                    in_fmt="xyxy",
                    out_fmt="cxcywh",
                ).clamp(0, 1)
                
        else:
            boxes_cxcywh = boxes_cxcywh.reshape(N, 4).to(mask_logits.device, mask_logits.dtype)

        x = mask_logits.sigmoid().reshape(N, 1, H, W)
        coord = self.get_coords(H, W, x.device, x.dtype, N)
        x = torch.cat([x, coord], dim=1)

        # small bounded residual in normalized box space
        delta = self.delta_scale * torch.tanh(self.net(x))

        out_boxes_cxcywh = (boxes_cxcywh + delta).clamp(0.0, 1.0)
        return out_boxes_cxcywh.reshape(B, Q, 4)
    
class EoMT(EoMT):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__(
            encoder,
            num_classes,
            num_q,
            num_blocks,
            masked_attn_enabled,
        )
        self.init_bbox_head()
    
    def init_bbox_head(self):
        self.bbox_head = MaskResidualBoxHead()

    def _predict_bbox(self, mask_logits: torch.Tensor):
        return self.bbox_head(mask_logits) # normalized cxcywh in [0,1].
    
    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        bbox_head_enabled = hasattr(self, "bbox_head")
        if bbox_head_enabled:bbox_preds_per_layer = []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                if bbox_head_enabled:
                    bbox_preds = self._predict_bbox(mask_logits)
                    bbox_preds_per_layer.append(bbox_preds)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        if bbox_head_enabled:
            bbox_preds = self._predict_bbox(mask_logits)
            bbox_preds_per_layer.append(bbox_preds)
            return (
                mask_logits_per_layer,
                class_logits_per_layer,
                bbox_preds_per_layer,
            )

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
