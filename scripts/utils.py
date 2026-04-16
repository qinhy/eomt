import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    x_min = torch.minimum(x0, x1)
    y_min = torch.minimum(y0, y1)
    x_max = torch.maximum(x0, x1)
    y_max = torch.maximum(y0, y1)
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1).clamp(0, 1)

def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    x_min = torch.minimum(x0, x1)
    y_min = torch.minimum(y0, y1)
    x_max = torch.maximum(x0, x1)
    y_max = torch.maximum(y0, y1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = x_max - x_min
    h = y_max - y_min
    return torch.stack([cx, cy, w, h], dim=-1).clamp(0, 1)

def save_mask_with_box(
    mask,
    box,
    save_path,
    threshold=0.0,
    box_normalized=True,
    box_color="red",
    linewidth=2,
    title=None,
):
    """
    Save one mask with one box overlaid.

    Args:
        mask: Tensor or ndarray of shape [H, W].
              Can be logits, probabilities, or binary mask.
        box: Tensor/list/ndarray with 4 values [x1, y1, x2, y2].
        save_path: Output image path, e.g. "vis/example.png".
        threshold: If mask is not binary, use mask > threshold for visualization.
        box_normalized: If True, box is assumed in [0, 1] xyxy and will be
                        scaled to image size. If False, box is assumed pixel xyxy.
        box_color: Matplotlib color for the box.
        linewidth: Box line width.
        title: Optional plot title.
    """
    print(mask,box)
    if torch.is_tensor(mask):
        mask = mask.detach().float().cpu()
        if mask.ndim != 2:
            raise ValueError(f"mask must have shape [H, W], got {tuple(mask.shape)}")
        mask_np = mask.numpy()
    else:
        mask_np = np.asarray(mask, dtype=np.float32)
        if mask_np.ndim != 2:
            raise ValueError(f"mask must have shape [H, W], got {mask_np.shape}")

    H, W = mask_np.shape

    if torch.is_tensor(box):
        box = box.detach().float().cpu().tolist()
    else:
        box = list(box)

    if len(box) != 4:
        raise ValueError(f"box must have 4 values [x1, y1, x2, y2], got {box}")

    x1, y1, x2, y2 = map(float, box)

    if box_normalized:
        x1 *= W
        x2 *= W
        y1 *= H
        y2 *= H

    # Binary visualization from logits/probabilities/binary mask
    vis_mask = (mask_np > threshold).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(vis_mask, cmap="gray")

    rect = Rectangle(
        (x1, y1),
        max(x2 - x1, 1e-6),
        max(y2 - y1, 1e-6),
        fill=False,
        edgecolor=box_color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)

    if title is not None:
        ax.set_title(title)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def summ(model, verbose=True, include_buffers=True):
    info = []
    for name, module in model.named_modules():
        # parameters for this module only (no children)
        params = list(module.parameters(recurse=False))
        nparams = sum(p.numel() for p in params)

        # collect dtypes from params (and optionally buffers)
        tensors = params
        if include_buffers:
            tensors += list(module.buffers(recurse=False))

        dtypes = {t.dtype for t in tensors}
        if dtypes:
            # e.g. "float32", "float16, int8"
            dtype_str = ", ".join(
                sorted(str(dt).replace("torch.", "") for dt in dtypes)
            )
        else:
            dtype_str = "-" + " "*14

        out_chs = ""
        if hasattr(module,'out_channels'):
            out_chs = f"out_channels={module.out_channels}"
            
        row = (name, module.__class__.__name__, nparams, dtype_str)
        info.append(row)

        if verbose:
            print(f"{name:35} {module.__class__.__name__:25} "
                  f"params={nparams:8d}  dtypes={dtype_str:15} {out_chs}")
    return info
