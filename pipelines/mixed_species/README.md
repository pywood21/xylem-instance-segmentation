# Mixed-species pipeline

This directory implements the mixed-species model described in the paper ("@@@": Section "@@@").

In this setting, a single Mask R-CNN model is trained using image patches pooled across all species, without explicitly modelling species or domain labels. This pipeline serves as a standard multi-domain baseline for evaluating cross-species generalisation under heterogeneous xylem anatomical structures.

## Files

```text
pipelines/mixed_species/
├─ preprocess.py
├─ train.py
├─ inference.py
└─ README.md
```

## What this pipeline does

1. `preprocess.py`
   - reads pooled multi-species image/mask pairs
   - resizes original images and masks to 1536 × 1536 by default
   - creates 256 × 256 and 512 × 512 patches
   - splits patches into train / valid / test
   - writes image and mask patches
   - creates COCO JSON annotation files

2. `train.py`
   - trains one Mask R-CNN model on pooled multi-species patches
   - reads COCO-formatted annotations
   - saves the best checkpoint based on validation loss

3. `inference.py`
   - loads a trained mixed-species checkpoint
   - runs single-frame or multi-frame inference
   - saves raw prediction images, cleaned prediction images, and regionprops tables

## Input data

The preprocessing script expects the original pooled dataset to be organised as follows:

```text
dataset/
└─ original/
   ├─ image/
   │  ├─ acapod3_01_img.png
   │  ├─ acapod3_02_img.png
   │  └─ ...
   └─ mask/
      ├─ acapod3_01_msk.png
      ├─ acapod3_02_msk.png
      └─ ...
```

Images and masks are paired by sorted order, following the original notebooks.

## Preprocessing output

`preprocess.py` creates the following structure:

```text
dataset/
├─ png_256/
│  ├─ image/
│  │  ├─ train/
│  │  ├─ valid/
│  │  └─ test/
│  └─ mask/
│     ├─ train/
│     ├─ valid/
│     └─ test/
│
└─ png_512/
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
{patch_number:04}_{original_image_basename}.png
```

Example:

```text
dataset/png_256/image/train/0000_acapod3_01_img.png
dataset/png_256/mask/train/0000_acapod3_01_img.png
```

The mask patch uses the same basename as the image patch, following the original notebook design.

## COCO annotation files

By default, `preprocess.py` also creates COCO JSON annotation files in the image split directories:

```text
dataset/png_256/image/train/train_annot_coco_256.json
dataset/png_256/image/valid/valid_annot_coco_256.json
dataset/png_256/image/test/test_annot_coco_256.json

dataset/png_512/image/train/train_annot_coco_512.json
dataset/png_512/image/valid/valid_annot_coco_512.json
dataset/png_512/image/test/test_annot_coco_512.json
```

These files are required by `train.py`.

## How to run

Run commands from the repository root, or specify `--dataset-root` explicitly.

### 1. Check preprocessing without writing files

```bash
python pipelines/mixed_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --dry-run
```

If the dataset is located at `dataset/` relative to the current working directory, this is enough:

```bash
python pipelines/mixed_species/preprocess.py --dry-run
```

### 2. Run preprocessing

```bash
python pipelines/mixed_species/preprocess.py \
  --dataset-root /path/to/dataset
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
python pipelines/mixed_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --seed 0
```

To create patches without COCO JSON:

```bash
python pipelines/mixed_species/preprocess.py \
  --dataset-root /path/to/dataset \
  --no-create-coco-json
```

### 3. Train

For a 256-patch model:

```bash
python pipelines/mixed_species/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/256_multispecies_700_epoch.tar
```

For a 512-patch model:

```bash
python pipelines/mixed_species/train.py \
  --dataset-root /path/to/dataset \
  --image-size 512 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/512_multispecies_700_epoch.tar
```

For a safe one-epoch test run:

```bash
python pipelines/mixed_species/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --epochs 1 \
  --batch-size 1 \
  --checkpoint-file ./tmp_checkpoints/ms_test_1epoch.tar
```

### 4. Inference

Single-frame inference:

```bash
python pipelines/mixed_species/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_multispecies_700_epoch.tar \
  --mode single \
  --test-index 0 \
  --output-root ./inference_results_mixed_species
```

Single-frame inference with an explicitly specified image:

```bash
python pipelines/mixed_species/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_multispecies_700_epoch.tar \
  --mode single \
  --input-image /path/to/image.png \
  --output-root ./inference_results_mixed_species
```

Multi-frame inference:

```bash
python pipelines/mixed_species/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_multispecies_700_epoch.tar \
  --mode multi \
  --input-image /path/to/full_size_image.png \
  --output-root ./inference_results_mixed_species
```

For multi-frame inference, specifying `--input-image` is recommended.

## Inference output

`inference.py` writes outputs under the specified `--output-root`.

Example:

```text
inference_results_mixed_species/
└─ 256_mixed_species_<checkpoint_name>/
   ├─ prediction_raw/
   │  └─ <image_id>_prediction_raw.png
   ├─ prediction_cleaned/
   │  └─ <image_id>_pred_cleaned.png
   └─ <image_id>_regionprops.csv
```

## Notes

- This pipeline does not explicitly model species identity or domain labels.
- It provides a baseline for cross-species generalisation.
- Performance may be affected by interspecific anatomical heterogeneity.
- The authoritative implementation is in `pipelines/mixed_species/`.
- Jupyter notebooks under `notebooks/mixed_species/` are retained for inspection, visualisation, and figure generation.
- This pipeline does not currently use `config.yaml`.
- Outputs are controlled through CLI arguments such as `--dataset-root`, `--checkpoint-file`, and `--output-root`.

## Paper mapping

- Model name in paper: Mixed-species model
- Corresponding directory: `pipelines/mixed_species/`
- Related figures/tables: "@@@"
