import math
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import box_convert, masks_to_boxes

from dinov3.layers.block import SelfAttentionBlock
from dinov3.models.vision_transformer import DinoVisionTransformer
from models.scale_block import FSRCNNx2YOnlyRGBWrapper, ScaleBlock, build_or_load_fsrcnn_x2

REPO_ROOT = Path(__file__).resolve().parents[1]
FSRCNN_X2_WEIGHTS = REPO_ROOT / "data" / "fsrcnn_x2.pth"

def parameter_to_buffer(
    module: nn.Module,
    name: str,
    *,
    clone: bool = True,
    persistent: bool = True,
):
    p = getattr(module, name)

    if not isinstance(p, nn.Parameter):
        raise TypeError(f"{name!r} is not an nn.Parameter")

    t = p.detach()
    if clone:
        t = t.clone()

    # Remove from registered parameters
    delattr(module, name)

    # Re-register under the same name as a buffer
    module.register_buffer(name, t, persistent=persistent)
    return module


def freeze_module_as_buffers(
    module: nn.Module,
    *,
    clone: bool = True,
    persistent: bool = True,
):
    """
    Convert every parameter in `module` (including all child submodules)
    into a buffer under the same attribute name.

    Returns:
        list[str]: parameter names (relative to `module`) that were converted
    """
    # Snapshot names first, because we are going to mutate registrations
    names = [name for name, _ in module.named_parameters(recurse=True)]

    for full_name in names:
        *parts, leaf = full_name.split(".")
        submod = module
        for part in parts:
            submod = getattr(submod, part)
        parameter_to_buffer(submod, leaf, clone=clone, persistent=persistent)

    return names

def token2map(x: torch.Tensor, grid_size):
    if x is None:
        return None
    B, N, D = x.shape
    gh, gw = grid_size
    assert N == gh * gw, f"Token count {N} does not match grid {gh}x{gw}"
    return x.transpose(1, 2).reshape(B, D, gh, gw)

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


class RandomResizeToMultipleOf16:
    def __init__(self, scale=(0.5, 1.0), patch_size=16):
        self.scale = scale
        self.patch_size = patch_size
    def __call__(self, img):
        # img: [..., H, W]
        h, w = img.shape[-2], img.shape[-1]
        s = random.uniform(*self.scale)
        new_h = max(self.patch_size, round(h * s / self.patch_size) * self.patch_size)
        new_w = max(self.patch_size, round(w * s / self.patch_size) * self.patch_size)
        return torchvision.transforms.functional.resize(img, [new_h, new_w])
    
class EoMT(nn.Module):
    class Index:
        def __init__(self,num_q,num_backbone_prefix_tokens):
            self.query_start = 0
            self.query_end = num_q
            self.prefix_start = self.query_end
            self.prefix_end = self.prefix_start + num_backbone_prefix_tokens
            self.patch_start = self.prefix_end

    """
    Drop-in rewrite of the pasted EoMT class with a consistent token layout:
        [queries][backbone_prefix][patches]

    This matches the current prepend operation:
        x = torch.cat([query_tokens, x], dim=1)
    """

    def __init__(
        self,
        encoder_weights,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        bbox_head_enabled=False,
        bbox_head_weight=0.1,
        upscale=False,
        fsrcnnx2=False,
        encoder_repo='../dinov3',
        encoder_model='dinov3_vitl16',
        precision="bf16-true",
        owner_head_enabled=True,
        owner_tau: float = 0.07,
        owner_fusion_alpha: float = 0.5,
        owner_score_thresh: float = 0.3,
        owner_min_area: int = 16,
    ):
        super().__init__()
        if "bf16" in precision:
            self.dtype = dtype = torch.bfloat16
        elif "fp16" in precision:
            self.dtype = dtype = torch.float16
        else:
            self.dtype = dtype = torch.float32

        # print(f"load model of {encoder_model},{encoder_weights}")
        self.encoder: DinoVisionTransformer = torch.hub.load(encoder_repo, encoder_model,
                                                             source='local',weights=encoder_weights).to(dtype=dtype)

        self.patch_size = self.encoder.patch_size
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.bbox_head_enabled = bbox_head_enabled
        self.bbox_head_weight = bbox_head_weight

        self.owner_head_enabled = owner_head_enabled
        self.owner_tau = owner_tau
        self.owner_fusion_alpha = owner_fusion_alpha
        self.owner_score_thresh = owner_score_thresh
        self.owner_min_area = owner_min_area

        D = self.encoder.embed_dim
        self.num_classes = num_classes
        self.num_backbone_prefix_tokens = self.encoder.n_storage_tokens+1        
        self.idx = self.Index(num_q+1,self.num_backbone_prefix_tokens)

        self.fsrcnnx2 = None
        if fsrcnnx2:
            self.fsrcnnx2:FSRCNNx2YOnlyRGBWrapper = build_or_load_fsrcnn_x2(
                checkpoint_path=str(FSRCNN_X2_WEIGHTS)
            ).to(dtype=dtype)
            freeze_module_as_buffers(self.fsrcnnx2)

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks).to(dtype=dtype))

        self.q = nn.Embedding(num_q+1, D).to(dtype=dtype) # num_q + bg = num_q+1

        # class head 
        DD = num_classes + 1
        # mask head 
        DD += D
        if self.owner_head_enabled:
            DD += D
        if self.bbox_head_enabled:
        # bbox head 
            DD += 4
            self.bbox_res = masks_to_boxes_cxcywh
        self.output_head = nn.Sequential(
            nn.Linear(D, int(DD*0.6)),  nn.GELU(),
            nn.Linear(int(DD*0.6), int(DD*0.8)), nn.GELU(),
            nn.Linear(int(DD*0.8), DD),
        ).to(dtype=dtype)

        patch_size = self.encoder.patch_size
        max_patch_size = patch_size
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        if upscale:
            self.upscale = nn.Sequential(*[ScaleBlock(D) for _ in range(num_upscale)]).to(dtype=dtype)
        else:
            self.upscale = None

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1).to(dtype=dtype)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1).to(dtype=dtype)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def _normalize_image(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return (x - self.pixel_mean) / self.pixel_std
        
    def _attn_mask(
        self,
        num_tokens: int,                     # total token count N from x: [B, N, D]
        mask_logits: torch.Tensor,           # [B, num_queries, h, w]
        grid_size: Tuple[int, int],    # (patch_h, patch_w)
        late_block_idx: int,
    ) -> torch.Tensor:
        idx = self.idx
        batch_size, num_queries, _, _ = mask_logits.shape

        # Start from full attention everywhere.
        attn_mask = torch.ones(
            batch_size, num_tokens, num_tokens, dtype=torch.bool, device=mask_logits.device,
        )

        # Resize query masks to the patch-token grid, then flatten spatial dims:
        # [B, Q, H, W] -> [B, Q, patch_h, patch_w] -> [B, Q, num_patches]
        patch_mask = F.interpolate( mask_logits, size=grid_size, mode="bilinear", align_corners=False,
        ).flatten(2)

        # Restrict query -> patch attention using the predicted masks.
        attn_mask[ :, idx.query_start:idx.query_start+num_queries, idx.patch_start:,
        ] = patch_mask > 0

        # Randomly disable masking for some queries, according to the annealed probability.
        keep_mask_prob = float(self.attn_mask_probs[late_block_idx])
        attn_mask = self._disable_attn_mask(
            attn_mask=attn_mask,
            keep_mask_prob=keep_mask_prob,
            query_start=idx.query_start,
            query_end=idx.query_end,
            patch_start=idx.patch_start,
        )
        return attn_mask


    @torch.compiler.disable
    def _disable_attn_mask(
        self,
        attn_mask: torch.Tensor,
        keep_mask_prob: float,
        query_start: int,
        query_end: int,
        patch_start: int,
    ) -> torch.Tensor:
        # keep_mask_prob == 1.0 means masking stays enabled for all queries.
        if keep_mask_prob >= 1.0:
            return attn_mask

        batch_size = attn_mask.shape[0]
        num_queries = query_end - query_start

        # True means: disable masked attention for this query and fall back to full attention.
        disable_mask_for_query = (
            torch.rand(batch_size, num_queries, device=attn_mask.device) > keep_mask_prob
        )

        # View over query -> patch attention region: [B, Q, num_patches]
        query_to_patch_mask = attn_mask[:, query_start:query_end, patch_start:]

        # If masking is disabled for a query, allow attention to all patches.
        # True = allowed attention
        # False = blocked attention
        query_to_patch_mask[disable_mask_for_query] = True

        return attn_mask
    
    def _predict(
        self,
        query_tokens: torch.Tensor, # (B,Q,D)
        patch_tokens_map: torch.Tensor, # (B,D,Hp,Wp)
        patch_tokens_map_x2: Optional[torch.Tensor] = None, # (B,D, Hp*n, Wp*n)
    ):
        # Q = num_q +1
        output_logits = self.output_head(query_tokens) # (B,Q,DD)

        cls_num = self.num_classes + 1
        class_logits = output_logits[:, :, : cls_num] # (B,Q,cls_num)
        D = self.encoder.embed_dim
        query_mask_feats = output_logits[:, :, cls_num : (cls_num + D)]  # (B,Q,D)
                
        if patch_tokens_map_x2 is None:
            if self.upscale is not None:
                patch_tokens_map_x2 = self.upscale(patch_tokens_map)
            else:
                patch_tokens_map_x2 = patch_tokens_map            

        # old mask head: keep it for mask/dice loss + masked attention
        binary_mask = mask_logits = torch.einsum("bqd,bdhw->bqhw", query_mask_feats, patch_tokens_map_x2) # (B,Q,Hp*n,Wp*n)

        # new owner head: final non-overlap competition head
        owner_queries_logits = None
        if self.owner_head_enabled:
            owner_queries = output_logits[:, :, (cls_num + D) : (cls_num + D) + D]  # (B,Q,D)
            owner_queries_logits = torch.einsum("bqd,bdhw->bqhw", owner_queries, patch_tokens_map_x2) # (B,Q,Hp*n,Wp*n)
            
            # fuse with mask_logits
            # owner_queries_logits = mask_logits = owner_queries_logits*self.owner_fusion_alpha + mask_logits*(1.0-self.owner_fusion_alpha)            
            # binary_mask = F.one_hot(owner_queries_logits.argmax(dim=1),
            #                         num_classes=owner_queries_logits.shape[1]).to(dtype=owner_queries_logits.dtype) # (B,Hp*n,Wp*n,Q)
            # binary_mask = binary_mask.permute(0,3,1,2) # (B,Q,Hp*n,Wp*n)

        bbox_preds = None
        if self.bbox_head_enabled:
            bbox_preds = output_logits[:, :, -4 :].sigmoid() # normalized cxcywh in [0,1].
            bbox_preds = bbox_preds*(self.bbox_head_weight) + (1.0-self.bbox_head_weight)*self.bbox_res(mask_logits) # normalized cxcywh in [0,1]. # (B,Q,4)
        return mask_logits[:,:self.num_q,:], class_logits[:,:self.num_q,:], bbox_preds[:,:self.num_q,:], owner_queries_logits

    def forward_dinov3_phase1(self, x: torch.Tensor):
        backbone = self.encoder
        num_layers = len(backbone.blocks)
        if not (1 <= self.num_blocks <= num_layers):
            raise ValueError(f"self.num_blocks must be in [1, {num_layers}], got {self.num_blocks}")

        split_idx = num_layers - self.num_blocks
        early_layers = backbone.blocks[:split_idx]
        late_layers = backbone.blocks[split_idx:]

        x, (Hp, Wp) = backbone.prepare_tokens_with_masks(x) # [CLS][registers][patches]
        rope = backbone.rope_embed(H=Hp, W=Wp)
        for block in early_layers:
            x = block._forward(x, rope=rope)
        return x, late_layers, rope, (Hp, Wp)

    def forward_dinov3(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ):
        backbone: DinoVisionTransformer = self.encoder
        B, C, H, W = x.shape
        idx = self.idx

        x, late_layers, rope, grid_size = self.forward_dinov3_phase1(x)

        if x2 is not None:
            x2, _, x2_rope, grid_size_x2 = self.forward_dinov3_phase1(x2)
            
        query_tokens = self.q.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([query_tokens, x], dim=1)
        # [queries][cls][register][img patches]
        B, N, D = x.shape

        res = []
        attn_mask = None
        for late_block_idx, block in enumerate(late_layers):
            block: SelfAttentionBlock = block
            
            if self.masked_attn_enabled:
                norm_x = backbone.norm(x)

                if x2 is not None:
                    norm_x2 = backbone.norm(x2)

                if self.training:
                    mask_logits, class_logits, bbox_preds, owner_queries_logits = self._predict(
                        query_tokens=norm_x[:,idx.query_start:idx.query_end],
                        patch_tokens_map=token2map(norm_x[:,idx.patch_start:],grid_size),
                        patch_tokens_map_x2=token2map(norm_x2[:,self.num_backbone_prefix_tokens:] if x2 is not None else None, grid_size_x2),
                    )
                    res.append((mask_logits, class_logits, bbox_preds, owner_queries_logits))

                    attn_mask = self._attn_mask(N,mask_logits,
                                                grid_size=grid_size,
                                                late_block_idx=late_block_idx)

            layer_attn_mask = attn_mask
            if layer_attn_mask is not None:
                layer_attn_mask = layer_attn_mask[:, None].expand(
                    -1, block.attn.num_heads, -1, -1
                )

            x = block._forward(x,rope,layer_attn_mask)
            # x = block(x, layer_attn_mask, position_embeddings=rope)
            if x2 is not None:
                x2 = block._forward(x2,x2_rope)
                # x2 = block(x2, None, position_embeddings=x2_rope)

        norm_x = backbone.norm(x)
        if x2 is not None:
            norm_x2 = backbone.norm(x2)

        mask_logits, class_logits, bbox_preds, owner_queries_logits = self._predict(
            query_tokens=norm_x[:,idx.query_start:idx.query_end],
            patch_tokens_map=token2map(norm_x[:,idx.patch_start:],grid_size),
            patch_tokens_map_x2=token2map(norm_x2[:,self.num_backbone_prefix_tokens:] if x2 is not None else None, grid_size_x2),
        )
        res.append((mask_logits, class_logits, bbox_preds, owner_queries_logits))

        mask_logits_per_layer = [m for m, _, _, _ in res]
        class_logits_per_layer = [c for _, c, _, _ in res]
        bbox_preds_per_layer = [b for _, _, b, _ in res]
        owner_logits_per_layer = [o for _, _, _, o in res]
        return mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer, owner_logits_per_layer

    def forward(self, x: torch.Tensor, x2: Optional[torch.Tensor] = None):
        x = x.to(dtype=self.dtype)

        if not x.is_floating_point():
            x = x.float().div(255.0)

        if self.fsrcnnx2 is not None and x2 is None:
            x2 = self.fsrcnnx2(x).detach()

        if self.training and x2 is not None:
            x2 = RandomResizeToMultipleOf16(scale=(0.7, 1.3), patch_size=self.patch_size)(x2).detach()

        if x2 is not None:
            x2 = x2.to(dtype=self.dtype)

        x = self._normalize_image(x)
        x2 = self._normalize_image(x2)

        if x2 is not None:
            return self.forward_dinov3(x, x2)

        return self.forward_dinov3(x)
