# Per-species pipeline

This directory implements the per-species model described in the paper ("@@@": Section "@@@").

In this setting, separate Mask R-CNN models are trained independently for each species using only the image patches derived from that species. This pipeline serves as an upper-bound benchmark for within-domain performance, because training and evaluation data share the same species-specific anatomical characteristics.

## Files

```text
pipelines/per_species/
├─ preprocess.py
├─ train.py
├─ inference.py
└─ README.md
```

## What this pipeline does

1. `preprocess.py`
   - reads image/mask pairs separately for each species
   - resizes original images and masks to 1536 × 1536 by default
   - creates 256 × 256 and 512 × 512 patches
   - splits patches into train / valid / test within each original image
   - writes image and mask patches
   - creates COCO JSON annotation files for each species

2. `train.py`
   - trains one Mask R-CNN model for one species
   - reads COCO-formatted annotations
   - saves the best checkpoint based on validation loss

3. `inference.py`
   - loads a trained per-species checkpoint
   - runs single-frame or multi-frame inference
   - saves raw prediction images, cleaned prediction images, and regionprops tables

## Input data

The preprocessing script expects the original dataset to be organised by species:

```text
dataset/
├─ acapod/
│  └─ original/
│     ├─ image/
│     │  ├─ acapod3_01_img.png
│     │  ├─ acapod3_02_img.png
│     │  └─ ...
│     └─ mask/
│        ├─ acapod3_01_msk.png
│        ├─ acapod3_02_msk.png
│        └─ ...
├─ alllit/
├─ burspi/
├─ euclon/
├─ exostr/
└─ hakdac/
```

Images and masks are matched by their shared image ID:

```text
image: <ID>_img.png
mask : <ID>_msk.png
```

Example:

```text
image: acapod3_01_img.png
mask : acapod3_01_msk.png
```

## Preprocessing output

`preprocess.py` creates the following structure for each species:

```text
dataset/
└─ acapod/
   ├─ png_256_thin/
   │  ├─ image/
   │  │  ├─ train/
   │  │  ├─ valid/
   │  │  └─ test/
   │  └─ mask/
   │     ├─ train/
   │     ├─ valid/
   │     └─ test/
   │
   └─ png_512_thin/
      ├─ image/
      │  ├─ train/
      │  ├─ valid/
      │  └─ test/
      └─ mask/
         ├─ train/
         ├─ valid/
         └─ test/
```

Patch files are saved using notebook-style filenames:

```text
{patch_number:04}_{original_image_id}.png
```

Example:

```text
dataset/acapod/png_256_thin/image/train/0000_acapod3_01.png
dataset/acapod/png_256_thin/mask/train/0000_acapod3_01.png
```

The mask patch uses the same basename as the image patch, following the original notebook design.

## COCO annotation files

By default, `preprocess.py` also creates COCO JSON annotation files in the image split directories:

```text
dataset/acapod/png_256_thin/image/train/train_annot_coco_256.json
dataset/acapod/png_256_thin/image/valid/valid_annot_coco_256.json
dataset/acapod/png_256_thin/image/test/test_annot_coco_256.json

dataset/acapod/png_512_thin/image/train/train_annot_coco_512.json
dataset/acapod/png_512_thin/image/valid/valid_annot_coco_512.json
dataset/acapod/png_512_thin/image/test/test_annot_coco_512.json
```

These files are required by `train.py`.

## How to run

Run commands from the repository root, or specify `--dataset-root` explicitly.

### 1. Check preprocessing without writing files

For one species:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --dry-run
```

For all species directories under `dataset/`:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species all \
  --dry-run
```

If the dataset is located at `dataset/` relative to the current working directory, this is enough:

```bash
python pipelines/per_species/preprocess.py \
  --species acapod \
  --dry-run
```

### 2. Run preprocessing

For one species:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species acapod
```

For all species:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species all
```

Default settings:

```text
dataset-root = dataset
image-size   = 1536
patch-sizes  = 256 512
seed         = None
COCO JSON    = created
```

To make the train / valid / test split reproducible:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --seed 0
```

To create patches without COCO JSON:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --no-create-coco-json
```

### 3. Train

Train a 256-patch model for one species:

```bash
python pipelines/per_species/train.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 256 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/256_thin_acapod_700_epoch.tar
```

Train a 512-patch model for one species:

```bash
python pipelines/per_species/train.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 512 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/512_thin_acapod_700_epoch.tar
```

For a safe one-epoch test run:

```bash
python pipelines/per_species/train.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 256 \
  --epochs 1 \
  --batch-size 1 \
  --checkpoint-file ./tmp_checkpoints/ps_acapod_test_1epoch.tar
```

### 4. Inference

Single-frame inference:

```bash
python pipelines/per_species/inference.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 256 \
  --checkpoint-file ./model/256_thin_acapod_700_epoch.tar \
  --mode single \
  --test-index 0 \
  --output-root ./inference_results_per_species
```

Single-frame inference with an explicitly specified image:

```bash
python pipelines/per_species/inference.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 256 \
  --checkpoint-file ./model/256_thin_acapod_700_epoch.tar \
  --mode single \
  --input-image /path/to/image.png \
  --output-root ./inference_results_per_species
```

Multi-frame inference:

```bash
python pipelines/per_species/inference.py \
  --dataset-root /path/to/dataset \
  --species acapod \
  --image-size 256 \
  --checkpoint-file ./model/256_thin_acapod_700_epoch.tar \
  --mode multi \
  --input-image /path/to/full_size_image.png \
  --output-root ./inference_results_per_species
```

For multi-frame inference, specifying `--input-image` is recommended.

## Inference output

`inference.py` writes outputs under the specified `--output-root`.

Example:

```text
inference_results_per_species/
└─ 256_per_species_acapod_<checkpoint_name>/
   ├─ prediction_raw/
   │  └─ <image_id>_prediction_raw.png
   ├─ prediction_cleaned/
   │  └─ <image_id>_pred_cleaned.png
   └─ <image_id>_regionprops.csv
```

## Notes

- This pipeline trains one model per species.
- It is intended as a within-domain benchmark.
- It does not test cross-species generalisation directly.
- The authoritative implementation is in `pipelines/per_species/`.
- Jupyter notebooks under `notebooks/per_species/` are retained for inspection, visualisation, and figure generation.
- This pipeline does not currently use `config.yaml`.
- Outputs are controlled through CLI arguments such as `--dataset-root`, `--species`, `--checkpoint-file`, and `--output-root`.

## Paper mapping

- Model name in paper: Per-species model
- Corresponding directory: `pipelines/per_species/`
- Related figures/tables: "@@@"
