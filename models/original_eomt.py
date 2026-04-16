import torch
import torch.nn as nn
from models.eomt import MaskResidualBoxHead
from models.official_eomt import EoMT


class EoMT(EoMT):
    
    def init_bbox_head(self):
        self.bbox_head = MaskResidualBoxHead()

    def _predict_bbox(self, mask_logits: torch.Tensor):
        return self.bbox_head(mask_logits)
        # bbox_preds = bbox_logits.sigmoid() # normalized cxcywh in [0,1].
        # return bbox_preds
    
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
    

# uv run ./scripts/train_coco_instance.py --network-impl original_bbox --data-path D:\images.cocodataset.org --official-delta-ckpt ./data/EoMT-L_640×640_InstanceSegmentation_DINOv3.bin --official-backbone-name facebook/dinov3-vitl16-pretrain-lvd1689m --batch-size 1 --num-workers 0 --accelerator cuda --devices 1 --experiment-name bbox_head_only_original_eomt