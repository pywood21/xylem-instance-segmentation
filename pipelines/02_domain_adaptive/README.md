# Domain-adaptive pipeline

This directory implements the domain-adaptive model described in the paper ("@@@": Section "@@@").

In this setting, a single Mask R-CNN model is trained on pooled multi-species image patches, with an adversarial domain adaptation component. The model uses a gradient reversal layer (GRL) and a domain classifier to encourage domain-invariant feature learning across species.

This pipeline is intended to test whether cross-species generalisation can be improved by reducing species-specific feature bias while retaining instance segmentation performance.

## Files

```text
pipelines/domain_adaptive/
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
   - extracts species/domain labels from patch filenames
   - adds a gradient reversal layer and domain classifier
   - combines Mask R-CNN detection/segmentation losses with domain-adversarial loss
   - saves the best checkpoint based on validation loss

3. `inference.py`
   - loads a trained domain-adaptive checkpoint
   - runs single-frame, multi-frame, or batch inference
   - saves raw prediction images, cleaned prediction images, and regionprops tables
   - optionally exports cleaned predictions to COCO JSON

## Input data

The preprocessing script expects the original pooled dataset to be organised as follows:

```text
dataset/
└─ original/
   ├─ image/
   │  ├─ acapod3_01_img.png
   │  ├─ alllit3_01_img.png
   │  ├─ burspi2_01_img.png
   │  └─ ...
   └─ mask/
      ├─ acapod3_01_msk.png
      ├─ alllit3_01_msk.png
      ├─ burspi2_01_msk.png
      └─ ...
```

Images and masks are paired by sorted order, following the original notebooks.

Domain labels are inferred from the image filenames during training. For example:

```text
0000_acapod3_01_img.png -> acapod
0000_alllit3_01_img.png -> alllit
0000_burspi2_01_img.png -> burspi
```

## Preprocessing output

By default, `preprocess.py` creates the following structure:

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

## Domain labels

The default domain/species order is:

```text
alllit acapod burspi euclon exostr hakdac
```

This order is used to convert species names into domain IDs.

To change the order or species set, pass `--domain-names` explicitly when running `train.py`:

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --domain-names alllit acapod burspi euclon exostr hakdac
```

## How to run

Run commands from the repository root, or specify `--dataset-root` explicitly.

### 1. Check preprocessing without writing files

```bash
python pipelines/domain_adaptive/preprocess.py \
  --dataset-root /path/to/dataset \
  --dry-run
```

If the dataset is located at `dataset/` relative to the current working directory, this is enough:

```bash
python pipelines/domain_adaptive/preprocess.py --dry-run
```

### 2. Run preprocessing

```bash
python pipelines/domain_adaptive/preprocess.py \
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
python pipelines/domain_adaptive/preprocess.py \
  --dataset-root /path/to/dataset \
  --seed 0
```

To create patches without COCO JSON:

```bash
python pipelines/domain_adaptive/preprocess.py \
  --dataset-root /path/to/dataset \
  --no-create-coco-json
```

## Training

### 3. Train a 256-patch model

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/256_domain_adaptive_700_epoch.tar
```

### 4. Train a 512-patch model

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root /path/to/dataset \
  --image-size 512 \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/512_domain_adaptive_700_epoch.tar
```

### 5. Safe one-epoch test run

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --epochs 1 \
  --batch-size 1 \
  --checkpoint-file ./tmp_checkpoints/domain_test_1epoch.tar
```

## Optional: using `_thin` data

If the data are stored under:

```text
dataset/png_256_thin/
dataset/png_512_thin/
```

instead of:

```text
dataset/png_256/
dataset/png_512/
```

add:

```bash
--data-suffix _thin
```

Example:

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --data-suffix _thin \
  --epochs 700 \
  --batch-size 8 \
  --checkpoint-file ./model/256_thin_domain_adaptive_700_epoch.tar
```

## Inference

### 6. Single-frame inference

```bash
python pipelines/domain_adaptive/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_domain_adaptive_700_epoch.tar \
  --mode single \
  --test-index 0 \
  --output-root ./inference_results_domain_adaptive
```

### 7. Single-frame inference with an explicitly specified image

```bash
python pipelines/domain_adaptive/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_domain_adaptive_700_epoch.tar \
  --mode single \
  --input-image /path/to/image.png \
  --output-root ./inference_results_domain_adaptive
```

### 8. Multi-frame inference

```bash
python pipelines/domain_adaptive/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_domain_adaptive_700_epoch.tar \
  --mode multi \
  --input-image /path/to/full_size_image.png \
  --output-root ./inference_results_domain_adaptive
```

For multi-frame inference, specifying `--input-image` is required.

### 9. Batch multi-frame inference

```bash
python pipelines/domain_adaptive/inference.py \
  --dataset-root /path/to/dataset \
  --image-size 256 \
  --checkpoint-file ./model/256_domain_adaptive_700_epoch.tar \
  --mode batch \
  --input-dir /path/to/performance_check/image \
  --output-root ./inference_results_domain_adaptive
```

If `--input-dir` is omitted, batch mode expects:

```text
dataset/performance_check/image/
```

## Inference output

`inference.py` writes outputs under the specified `--output-root`.

Example:

```text
inference_results_domain_adaptive/
└─ 256_domain_adaptive_<checkpoint_name>/
   ├─ prediction_raw/
   │  └─ <image_id>_prediction_raw.png
   ├─ prediction_cleaned/
   │  └─ <image_id>_pred_cleaned.png
   └─ cell_props/
      └─ <image_id>_cell_props.csv
```

If `--export-coco` is used, cleaned prediction JSON files are also written:

```text
prediction_cleaned_json/
└─ <image_id>_annot_coco.json
```

## Gray value convention

The prediction masks follow the same gray value convention as the preprocessing and COCO conversion steps:

```text
vessel      -> 255
ray         -> 191
fiber       -> 127
parenchyma  -> 63
background  -> 0
```

## Notes

- This pipeline uses pooled multi-species data.
- Species/domain labels are not stored in a separate CSV file; they are extracted from filenames.
- The domain-adaptive component is introduced in `train.py`, not in `preprocess.py`.
- The model includes a gradient reversal layer and a domain classifier.
- The checkpoint contains both the Mask R-CNN model and the domain classifier.
- The authoritative implementation is in `pipelines/domain_adaptive/`.
- Jupyter notebooks under `notebooks/domain_adaptive/` are retained for inspection, visualisation, and figure generation.
- This pipeline does not currently use `config.yaml`.
- Outputs are controlled through CLI arguments such as `--dataset-root`, `--checkpoint-file`, `--data-suffix`, and `--output-root`.

## Paper mapping

- Model name in paper: Domain-adaptive model
- Corresponding directory: `pipelines/domain_adaptive/`
- Related figures/tables: "@@@"
