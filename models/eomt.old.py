# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import List, Optional, Tuple
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer
from models.scale_block import ScaleBlock
from models.vit import ViT

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

class MLP(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def token2map(x:torch.Tensor,grid_size):
    if x is None:return None
    B,N,D = x.shape
    return x.transpose(1, 2).reshape(B,D,grid_size[0],grid_size[1])

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
        encoder: ViT,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        bbox_head_enabled=False,
        upscale=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.patch_size = ph, pw = self.encoder.patch_size
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled
        self.bbox_head_enabled = bbox_head_enabled
        D = self.encoder.embed_dim

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, D)
        self.class_head = nn.Linear(D, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, D), nn.GELU(),
            nn.Linear(D, D),
        )
        if self.bbox_head_enabled:
            self.bbox_head = nn.Sequential(
                nn.Linear(D, D), nn.GELU(),
                nn.Linear(D, D), nn.GELU(),
                nn.Linear(D, 4),
            )
        else:
            self.bbox_head = None

        patch_size = encoder.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)
        if upscale:
            self.upscale = nn.Sequential(*[ScaleBlock(D) for _ in range(num_upscale)])
        else:
            self.upscale = None

        self.is_dinov3 = "dinov3" in self.encoder.backbone_name.lower()
        if self.is_dinov3:
            self.num_backbone_prefix_tokens = self.encoder.backbone.config.num_register_tokens+1
        
        self.idx = self.Index(num_q,self.num_backbone_prefix_tokens)

    def freeze_encoder(self):
        freeze_module_as_buffers(self.encoder)

    def _normalize_image(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is None:
            return None
        return (x - self.encoder.pixel_mean) / self.encoder.pixel_std

    def _attn_mask(
        self,
        N:int, # from x of B,N,D
        mask_logits: torch.Tensor, # B,q,h,w
        grid_size: Tuple[int,int],
        late_block_idx: int,
    ):
        idx = self.idx
        B,q,h,w = mask_logits.shape
        attn_mask = torch.ones(
            B,N,N,
            dtype=torch.bool,
            device=mask_logits.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            size=grid_size,
            mode="bilinear",
            align_corners=False,
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[:, idx.query_start:idx.query_end, idx.patch_start:] = interpolated > 0

        attn_mask = self._disable_attn_mask(
            attn_mask,
            float(self.attn_mask_probs[late_block_idx]),
            idx.query_start,
            idx.query_end,
            idx.patch_start,
        )
        return attn_mask

    @torch.compiler.disable
    def _disable_attn_mask(
        self,
        attn_mask: torch.Tensor,
        prob: float,
        query_start: int,
        query_end: int,
        patch_start: int,
    ):
        if prob < 1:
            random_queries = (
                torch.rand(
                    attn_mask.shape[0],
                    query_end - query_start,
                    device=attn_mask.device,
                ) > prob
            )
            attn_mask[:, query_start:query_end, patch_start:][random_queries] = True

        return attn_mask

    def _predict(
        self,
        query_tokens: torch.Tensor,
        patch_tokens_map: torch.Tensor,
        patch_tokens_map_x4: Optional[torch.Tensor] = None,
    ):
        class_logits = self.class_head(query_tokens)

        bbox_preds = None
        if self.bbox_head is not None:
            bbox_preds = self.bbox_head(query_tokens).sigmoid()
        
        if patch_tokens_map_x4 is None:
            if self.upscale is None:
                raise RuntimeError(
                    "patch_tokens_map_x4 is None and self.upscale is None. "
                    "Provide x4 features or enable the upscale module."
                )
            patch_tokens_map_x4 = self.upscale(patch_tokens_map)
        else:
            patch_tokens_map_x4 = patch_tokens_map_x4

        query_mask_feats = self.mask_head(query_tokens)
        mask_logits = torch.einsum("bqc,bchw->bqhw", query_mask_feats, patch_tokens_map_x4)

        return mask_logits, class_logits, bbox_preds

    def forward_dinov3_phase1(self, x: torch.Tensor):
        backbone: DINOv3ViTModel = self.encoder.backbone
        num_layers = len(backbone.layer)
        if not (1 <= self.num_blocks <= num_layers):
            raise ValueError(
                f"self.num_blocks must be in [1, {num_layers}], got {self.num_blocks}"
            )

        split_idx = num_layers - self.num_blocks
        early_layers = backbone.layer[:split_idx]
        late_layers = backbone.layer[split_idx:]

        rope = backbone.rope_embeddings(x)
        x = backbone.embeddings(x)  # [CLS][registers][patches]

        for block in early_layers:
            x = block(x, None, position_embeddings=rope)
        return x, late_layers, rope

    def forward_dinov3(
        self,
        x: torch.Tensor,
        x4: Optional[torch.Tensor] = None,
    ):
        backbone: DINOv3ViTModel = self.encoder.backbone
        B, C, H, W = x.shape
        idx = self.idx

        x, late_layers, rope = self.forward_dinov3_phase1(x)
        grid_size = (H // self.patch_size[0], W // self.patch_size[1])

        if x4 is not None:            
            _, _, H4, W4 = x4.shape
            x4, _, x4_rope = self.forward_dinov3_phase1(x4)            
            grid_size_x4 = (H4 // self.patch_size[0], W4 // self.patch_size[1])
            
        query_tokens = self.q.weight.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([query_tokens, x], dim=1)
        # [queries][cls][register][img patches]
        B, N, D = x.shape

        res = []
        attn_mask = None
        for late_block_idx, block in enumerate(late_layers):
            block: DINOv3ViTLayer = block
            
            if self.masked_attn_enabled:
                norm_x = backbone.norm(x)

                if x4 is not None:
                    norm_x4 = backbone.norm(x4)
                    
                mask_logits, class_logits, bbox_preds = self._predict(
                    query_tokens=norm_x[:,idx.query_start:idx.query_end],
                    patch_tokens_map=token2map(norm_x[:,idx.patch_start:],grid_size),
                    patch_tokens_map_x4=token2map(norm_x4[:,self.num_backbone_prefix_tokens:] if x4 is not None else None, grid_size_x4),
                )
                res.append((mask_logits, class_logits, bbox_preds))

                attn_mask = self._attn_mask(N,mask_logits,
                                            grid_size=grid_size,
                                            late_block_idx=late_block_idx)

            layer_attn_mask = attn_mask
            if layer_attn_mask is not None:
                layer_attn_mask = layer_attn_mask[:, None].expand(
                    -1, block.attention.num_heads, -1, -1
                )

            x = block(x, layer_attn_mask, position_embeddings=rope)
            if x4 is not None:
                x4 = block(x4, None, position_embeddings=x4_rope)

        norm_x = backbone.norm(x)

        if x4 is not None:
            norm_x4 = backbone.norm(x4)

        mask_logits, class_logits, bbox_preds = self._predict(
            query_tokens=norm_x[:,idx.query_start:idx.query_end],
            patch_tokens_map=token2map(norm_x[:,idx.patch_start:],grid_size),
            patch_tokens_map_x4=token2map(norm_x4[:,self.num_backbone_prefix_tokens:] if x4 is not None else None, grid_size_x4),
        )
        res.append((mask_logits, class_logits, bbox_preds))

        mask_logits_per_layer = [m for m, _, _ in res]
        class_logits_per_layer = [c for _, c, _ in res]
        bbox_preds_per_layer = [b for _, _, b in res]
        return mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer

    def forward(
        self,
        x: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        x4: Optional[torch.Tensor] = None,
    ):
        x = self._normalize_image(x)
        x2 = self._normalize_image(x2)
        x4 = self._normalize_image(x4)

        if not self.is_dinov3:
            raise NotImplementedError(
                "This rewrite only covers the DINOv3 path because that is the only path present in the pasted file."
            )

        if x4 is not None:
            return self.forward_dinov3(x, x4)

        if x2 is not None:
            x4_from_x2 = F.interpolate(x2,scale_factor=2,mode="bilinear",align_corners=False,)
            return self.forward_dinov3(x, x4_from_x2)

        if self.upscale is not None:
            return self.forward_dinov3(x)

        x4 = F.interpolate( x, scale_factor=2, mode="bilinear", align_corners=False,)
        x_half = F.interpolate( x, scale_factor=0.5, mode="bilinear", align_corners=False,)
        mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = self.forward_dinov3(x_half,x4,)
        mask_logits_per_layer = [
            F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
            for m in mask_logits_per_layer
        ]
        return mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer
        # else:
        #     return self.forward_vit(x)

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
        return x
    
    def forward_vit(self, x: torch.Tensor):
        backbone: timm.models.vision_transformer.VisionTransformer = self.encoder.backbone        
        rope = None
        x = backbone.patch_embed(x)
        x = backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer, bbox_preds_per_layer = [], [], []

        for i, block in enumerate(backbone.blocks):
            block: timm.models.vision_transformer.Block = block

            if i == len(backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits, bbox_preds = self._predict(
                    backbone.norm(x)
                )
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)
                bbox_preds_per_layer.append(bbox_preds)

                attn_mask = self._attn_mask(x, mask_logits, i)

            attn = block.attn
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)

            x = x + block.ls1(attn_out)
            mlp_out = block.mlp(block.norm2(x))
            x = x + block.ls2(mlp_out)

        mask_logits, class_logits, bbox_preds = self._predict(backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)
        bbox_preds_per_layer.append(bbox_preds)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
            bbox_preds_per_layer,
        )

if __name__ == "__main__":
    model = EoMT(num_q=200,
                 num_classes=80,
                 bbox_head_enabled=True,
                 encoder=ViT(
                    img_size=(320,320),
                    backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
            ).cuda()
    img = torch.randn(1, 3, 320, 320).cuda()
    res = model(img)
    print(res)