import torch
from torchvision.ops import box_convert

from datasets.coco_instance import COCOInstance
from models.eomt_dinov3 import EoMT
from training.mask_classification_loss import MaskClassificationLoss
from torch.nn.attention import sdpa_kernel, SDPBackend

model = EoMT(num_q=200,
                num_classes=80,
                bbox_head_enabled=True,
                encoder_repo='../dinov3',
                encoder_model='dinov3_vits16',
                encoder_weights='../BitNetCNN/data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
        ).cuda()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
    img = torch.randn(1, 3, 2880, 2880).cuda()
    res = model(img)
print(res)

# dinov3_vits16
# 160 400MB
# 320 900MB
# 640 3GB
# 720 3.5GB

# dataset = COCOInstance(path="D:/images.cocodataset.org",num_workers=0,batch_size=1)
# dataset.setup()
# tl = dataset.train_dataloader()
# batch = next(iter(tl))
# imgs,targets = batch
# imgs = imgs.cuda()

# masks, labels, is_crowd, boxes = targets[0]["masks"], targets[0]["labels"], targets[0]["is_crowd"], targets[0]["boxes"]
# vis = COCOInstance.draw_one(imgs[0], masks, labels, is_crowd, boxes)
# mask_logits_per_block, class_logits_per_block, bbox_preds_per_block = model(imgs)


# criterion = MaskClassificationLoss(num_points=12544,
#                                     oversample_ratio=3.0,
#                                     importance_sample_ratio=0.75,
#                                     mask_coefficient=5.0,
#                                     dice_coefficient=5.0,
#                                     class_coefficient=2.0,
#                                     num_labels=80,
#                                     no_object_coefficient=0.1,
#                                     bbox_l1_coefficient=5.0,
#                                     bbox_giou_coefficient=2.0).cuda()

# losses = criterion(masks_queries_logits=mask_logits_per_block[-1],
#                     class_queries_logits=class_logits_per_block[-1],
#                     bbox_queries_preds=bbox_preds_per_block[-1],
#                     targets=targets)
# print(losses)


# with torch.no_grad():
#     # pick image 0 from the batch
#     pred_logits = class_logits_per_block[-1][0]   # (Q, C) or (Q, C+1)
#     pred_masks  = mask_logits_per_block[-1][0]    # (Q, Hm, Wm)
#     pred_boxes  = bbox_preds_per_block[-1][0]     # (Q, 4)

#     # class scores
#     probs = pred_logits.softmax(-1)

#     # if your classifier has a "no-object" last class, use probs[..., :-1]
#     scores, pred_labels = probs.max(-1)

#     # keep only confident queries
#     keep = scores > 0.5

#     pred_labels = pred_labels[keep]
#     pred_masks  = pred_masks[keep]
#     pred_boxes  = pred_boxes[keep]

#     # resize masks to image size if needed
#     H, W = imgs[0].shape[-2:] if imgs[0].ndim == 3 else imgs[0].shape[:2]
#     if pred_masks.shape[-2:] != (H, W):
#         pred_masks = torch.nn.functional.interpolate(
#             pred_masks[:, None],
#             size=(H, W),
#             mode="bilinear",
#             align_corners=False,
#         )[:, 0]

#     # logits -> bool masks
#     pred_masks = pred_masks.sigmoid() > 0.5

#     # predictions do not have is_crowd; use all False
#     pred_is_crowd = [False] * len(pred_labels)

#     # If pred_boxes are already XYXY in image coordinates, keep as-is.
#     # If they are normalized cxcywh, convert first:
#     #
#     pred_boxes = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
#     pred_boxes[:, [0, 2]] *= W
#     pred_boxes[:, [1, 3]] *= H
    
#     # If they are normalized xyxy:
    
#     # pred_boxes[:, [0, 2]] *= W
#     # pred_boxes[:, [1, 3]] *= H

#     vis = COCOInstance.draw_one(
#         imgs[0],
#         pred_masks,
#         pred_labels,
#         pred_is_crowd,
#         pred_boxes,
#     )

