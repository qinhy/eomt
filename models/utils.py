import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, masks_to_boxes

class ClassHead(nn.Module):
    def __init__(self, num_classes, dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.class_head = nn.Linear(dim, num_classes + 1)

    def forward(self, x):
        return self.class_head(x)

class MaskHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.mask_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x, ):
        return self.mask_head(x)

class BboxHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, x, mask_logits=None, raw_logits=False):
        if not raw_logits:
            if mask_logits is not None:
                return self.bbox_head(x).sigmoid()*0.1 + self.masks_to_boxes_cxcywh(mask_logits)
            else:
                return self.bbox_head(x).sigmoid()
        return self.bbox_head(x)
    
    @staticmethod
    def masks_to_boxes_cxcywh(mask_logits: torch.Tensor):
        B, Q, H, W = mask_logits.shape
        N = B * Q
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
            return boxes_cxcywh.reshape(B, Q, 4)
        
def patch_tokens_shape(patch_tokens,grid_size=None):    
    B, N, D = patch_tokens.shape
    if grid_size is not None:
        H, W = grid_size
    else:
        H = W = int(math.sqrt(N))

    if N != H * W:
        raise ValueError(f"N={N} does not match grid_size={grid_size}")
    return B, N, D, H, W

class OwnerHead(nn.Module):
    def __init__(self, dim=768, owner_tau: float = 0.07):
        super().__init__()
        self.owner_tau = owner_tau

        self.owner_query_proj = nn.Linear(dim, dim)
        self.owner_pixel_proj = nn.Linear(dim, dim)
        self.owner_bg_head = nn.Linear(dim, 1)

    def forward(
        self,
        query_mask_feats: torch.Tensor,   # [B, Q, D]
        patch_tokens: torch.Tensor,       # [B, N, D], N = H*W
        grid_size: tuple[int, int]=None,  # (H, W)
    ):
        B, Q, D = query_mask_feats.shape
        B2, N, D2, H, W = patch_tokens_shape(patch_tokens,grid_size)
        if B != B2 or D != D2:
            raise ValueError(
                f"Shape mismatch: query_mask_feats={query_mask_feats.shape}, "
                f"patch_tokens={patch_tokens.shape}"
            )

        owner_q = self.owner_query_proj(query_mask_feats)   # [B, Q, D]
        owner_p = self.owner_pixel_proj(patch_tokens)       # [B, N, D]

        owner_q = F.normalize(owner_q, dim=-1)
        owner_p = F.normalize(owner_p, dim=-1)

        # query-vs-pixel affinity: [B, Q, N]
        owner_fg_logits = torch.einsum("bqd,bnd->bqn", owner_q, owner_p)
        owner_fg_logits = owner_fg_logits / max(self.owner_tau, 1e-6)

        # reshape to [B, Q, H, W]
        owner_fg_logits = owner_fg_logits.view(B, Q, H, W)

        # background logit per pixel: [B, N, 1] -> [B, 1, H, W]
        owner_bg_logits = self.owner_bg_head(patch_tokens)          # [B, N, 1]
        owner_bg_logits = owner_bg_logits.transpose(1, 2)           # [B, 1, N]
        owner_bg_logits = owner_bg_logits.view(B, 1, H, W)          # [B, 1, H, W]

        # final: [B, Q+1, H, W]
        owner_queries_logits = torch.cat([owner_fg_logits, owner_bg_logits], dim=1)
        return owner_queries_logits
    
class CenterHead(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.center_pred_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.center_reg_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, 2),
        )

    def forward(
        self,
        patch_tokens: torch.Tensor,        # [B, N, D]
        grid_size: tuple[int, int]=None,   # (H, W)
    ):
        B, N, D, H, W = patch_tokens_shape(patch_tokens,grid_size)
        center_pred_logits = self.center_pred_head(patch_tokens)   # [B, N, 1]
        center_reg = self.center_reg_head(patch_tokens)            # [B, N, 2]

        center_pred_logits = center_pred_logits.transpose(1, 2).reshape(B, 1, H, W)  # [B,1,H,W]
        center_reg = center_reg.permute(0, 2, 1).reshape(B, 2, H, W)                 # [B,2,H,W]

        return center_pred_logits, center_reg

