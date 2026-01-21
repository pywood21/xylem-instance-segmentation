# Mixed-species pipeline

This directory implements the **mixed-species model** described in the paper ("@@@": Section "@@@").  
In this setting, **a single model is trained using image patches pooled across all species**, without explicitly modelling species or domain labels.

This pipeline serves as a standard multi-domain baseline, testing the model’s ability to generalise across species under heterogeneous anatomical structures.

## What this pipeline does

- Trains a single Mask R-CNN model on pooled multi-species data
- Performs inference using the shared checkpoint across all species
- Writes outputs to the common `runs/` structure for consistent cross-pipeline evaluation

## How to run

### 1) Configure
Edit `config.yaml` to specify paths and experiment parameters.

Key items to check (names may differ depending on your config):
- `data_root`: path to pooled image patches
- `ann_path` / `coco_json`: path to COCO annotations (if used)
- `out_dir`: output base directory (recommended: `runs/mixed_species/`)
- `seed`: random seed
- `device`: `cuda` or `cpu`

### 2) Train
From the repository root:

    python pipelines/mixed_species/train.py --config pipelines/mixed_species/config.yaml

Expected behaviour:
- Trains a single model using all available species data
- Saves checkpoints under:

    runs/mixed_species/<experiment_id>/ckpt/

### 3) Inference
From the repository root:

    python pipelines/mixed_species/inference.py --config pipelines/mixed_species/config.yaml

Expected behaviour:
- Loads the shared checkpoint
- Runs inference across all species
- Writes predictions under:

    runs/mixed_species/<experiment_id>/preds/

## Outputs (convention)

This pipeline follows the shared output convention under `runs/`:

    runs/
      mixed_species/
        <experiment_id>/
          ckpt/      trained checkpoints
          preds/     predicted instance masks (and/or COCO-style outputs)
          scores/    evaluation outputs (optional; usually written by evaluation/)

`evaluation/score_all.py` assumes outputs are organised consistently across pipelines.

## Notes

- This pipeline does **not** explicitly model species or domain identity.
- It provides a baseline for cross-species generalisation, but performance may be affected by interspecific anatomical heterogeneity.
- Jupyter notebooks under `notebooks/mixed_species/` are provided for inspection and figure generation; they are not the authoritative implementation.

## Paper mapping

- Model name in paper: **Mixed-species model**
- Corresponding directory: `pipelines/mixed_species/`
- Related figures/tables: "@@@"

