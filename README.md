# EoMT Fork for COCO Instance Segmentation

This repository is a working fork of **EoMT** (Encoder-only Mask Transformer) from the CVPR 2025 Highlight paper **"Your ViT is Secretly an Image Segmentation Model"**.

In this fork, the maintained workflow is **pure PyTorch training for COCO instance segmentation** built around the custom final model in [`models/eomt.py`](./models/eomt.py). The upstream implementation is preserved in [`models/official_eomt.py`](./models/official_eomt.py) so the fork-specific changes stay explicit.

- Paper: <https://arxiv.org/abs/2503.19108>
- Upstream repo: <https://github.com/tue-mps/eomt>

## What This Fork Maintains

The codebase is currently centered on one practical training path:

- COCO instance segmentation
- Mask and bounding-box prediction from the same network
- Pure PyTorch training and validation loops
- Single-device execution
- CSV logging, checkpoint save/resume, and training visualizations
- Zip-backed COCO loading with persistent manifest caching

Reference code from upstream is still present, but the main path to use is:

```bash
python3 -m scripts.train_coco_instance --help
```

## Model File Map

This fork keeps multiple EoMT variants in the repository on purpose.

- [`models/eomt.py`](./models/eomt.py)
  The custom final model used by the maintained training path. It loads a DINOv3 backbone through `torch.hub.load(...)`, inserts learnable query tokens into the late transformer blocks, predicts classes, masks, and boxes from a shared output head, and can use an FSRCNN x2 auxiliary image branch.

- [`models/official_eomt.py`](./models/official_eomt.py)
  The preserved upstream official EoMT implementation. Treat this as the reference version when comparing the fork against the original project.

- [`models/original_eomt.py`](./models/original_eomt.py)
  A small adapter around the official model used for bbox-head experiments and compatibility work.

The default network used by [`scripts/train_coco_instance.py`](./scripts/train_coco_instance.py) is the custom forked model in `models/eomt.py` via `--network-impl dinov3`.

## Fast View: Original EoMT

Quick reference for the original/reference model:

```txt
EoMT(Original):
  trainable_tensors:
    - network.encoder.cls_token: 1,024
    - network.encoder.storage_tokens: 4,096
    - network.encoder.mask_token: 1,024
    - network.encoder.patch_embed.proj.weight: 786,432
    - network.encoder.patch_embed.proj.bias: 1,024
    - network.encoder.blocks.0.norm1.weight: 1,024
    - network.encoder.blocks.0.norm1.bias: 1,024
    ...encoder(dinov3) layers
    - network.encoder.blocks.23.ls2.gamma: 1,024
    - network.encoder.norm.weight: 1,024
    - network.encoder.norm.bias: 1,024

    - network.q.weight: 204,800
    - network.output_head.0.weight: 1,135,616
    - network.output_head.0.bias: 1,109
    - network.output_head.2.weight: 1,229,881
    - network.output_head.2.bias: 1,109
    - network.output_head.4.weight: 1,229,881
    - network.output_head.4.bias: 1,109
```

![alt text](./docs/e1.png)

## Main Differences From the Official Model

Compared with [`models/official_eomt.py`](./models/official_eomt.py), the maintained model in [`models/eomt.py`](./models/eomt.py):

- loads DINOv3 from a local repo checkout with `torch.hub.load`
- uses a unified output head that emits class logits, mask features, and bbox values together
- optionally builds an x2 auxiliary image with FSRCNN (`data/fsrcnn_x2.pth`)
- derives a base bbox from the predicted mask and learns a residual on top of it
- is wired into the repo's pure PyTorch trainer instead of the upstream training stack

## Installation

Python `3.10` to `3.13` is supported by the project metadata.

Recommended with `uv`:

```bash
uv sync
```

Fallback with `pip`:

```bash
python3 -m pip install -r requirements.txt
```

If you are not using the pinned `uv` environment, make sure your local PyTorch install matches your CUDA or CPU setup before training.

## Required Data and Weights

### 1. COCO archives

Point `--data-path` at a directory containing:

```text
/path/to/coco/
  train2017.zip
  val2017.zip
  annotations_trainval2017.zip
```

The COCO loader is implemented in [`datasets/coco_instance.py`](./datasets/coco_instance.py) and reads images plus annotation JSON directly from the zip files through [`datasets/zip_dataset.py`](./datasets/zip_dataset.py).

### 2. A local upstream DINOv3 checkout

The maintained model still loads the backbone with:

```python
torch.hub.load(encoder_repo, encoder_model, source="local", weights=encoder_weights)
```

That means you need a **separate local DINOv3 repo checkout** with the expected `hubconf.py` entrypoints.

Default path used by the trainer:

```text
../dinov3
```

### 3. DINOv3 pretrained weights

The maintained training path expects a local checkpoint for the backbone.

Default path:

```text
./data/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

If your checkpoint lives elsewhere, override it with `--encoder-weights`.

### 4. Optional official delta checkpoint

The training script also exposes an official/reference path through `--network-impl original_bbox`.

Default delta checkpoint path for that path:

```text
./data/EoMT-L_640×640_InstanceSegmentation_DINOv3.bin
```

### Important note about `dinov3/` in this repo

This repository already contains a local [`dinov3/`](./dinov3/) source tree for internal imports and utilities. That does **not** replace the external `--encoder-repo` checkout required by `torch.hub.load(...)`.

## Training the Maintained Model

Example:

```bash
uv run python -m scripts.train_coco_instance \
  --data-path /path/to/coco \
  --encoder-repo /path/to/dinov3 \
  --encoder-weights /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --batch-size 1 \
  --num-workers 4 \
  --accelerator cuda \
  --devices 1 \
  --experiment-name eomt_dinov3_coco
```

Resume from a saved training checkpoint:

```bash
uv run python -m scripts.train_coco_instance \
  --data-path /path/to/coco \
  --encoder-repo /path/to/dinov3 \
  --encoder-weights /path/to/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --resume-from-checkpoint logs/eomt_dinov3_coco/checkpoints/last.pt
```

Useful flags:

- `--img-size 640 640` sets the training image size
- `--precision` defaults to `bf16-true` on GPU and `32-true` on CPU
- `--compile` enables `torch.compile` for the network
- `--ckpt-path` loads model weights before training starts
- `--delta-weights` makes `--ckpt-path` behave like a delta checkpoint added on top of the current initialization
- `--force-rebuild-cache` rebuilds dataset manifest caches
- `--check-val-every-n-epoch` controls validation frequency
- `--devices 1` is the supported execution mode

## Auxiliary Script

This repo also includes:

```bash
python3 -m scripts.train_mask_residual_box_head --help
```

[`scripts/train_mask_residual_box_head.py`](./scripts/train_mask_residual_box_head.py) trains the small residual bbox head directly from COCO binary masks and ground-truth boxes. It is useful for bbox-head experiments, but it is not the main README workflow.

## Outputs

Each run writes to:

```text
logs/<experiment-name>/
```

Typical contents:

- `hparams.json`
- `metrics.csv`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `train_vis/*.png`

Dataset manifest cache files are written under:

```text
dataset_cache/
```

## Repo Layout

```text
scripts/train_coco_instance.py        Maintained training entrypoint
scripts/train_mask_residual_box_head.py  Auxiliary bbox-head training script
models/eomt.py                        Custom final forked model
models/official_eomt.py               Preserved upstream official model
models/original_eomt.py               Adapter around the official model
models/scale_block.py                 Upscaling and FSRCNN helpers
training/instance_module.py           COCO instance training/eval logic
training/loss.py                      Mask/class/box losses
training/engine.py                    Epoch and validation loops
training/checkpointing.py             Save/resume helpers
training/csv_logger.py                CSV metrics logger
datasets/coco_instance.py             COCO datamodule
datasets/zip_dataset.py               Zip-backed dataset with manifest caching
dinov3/                               Local DINOv3 source used by imports/utilities
docs/                                 Upstream paper/project-page assets
inference.ipynb                       Notebook for local experimentation
```

## Sanity Checks

Once dependencies are installed, useful smoke checks are:

```bash
python3 -m scripts.train_coco_instance --help
python3 -m scripts.train_mask_residual_box_head --help
python3 -m unittest discover -s tests
```

## Citation

If you use this work, cite the original EoMT paper:

```bibtex
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## License

This repository is released under the [MIT License](./LICENSE).
