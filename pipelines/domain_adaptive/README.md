
# Domain-adaptive pipeline

This directory implements the **domain-adaptive model** described in the paper ("@@@": Section "@@@").  
In this setting, **a single model is trained on pooled multi-species data using an adversarial domain adaptation framework**, aiming to learn domain-invariant feature representations.

The model incorporates a **gradient reversal layer (GRL)** and a **domain classifier** to reduce species-specific (domain-specific) biases while maintaining segmentation performance.

## What this pipeline does

- Trains a single Mask R-CNN–based model on pooled multi-species data
- Applies adversarial domain adaptation to encourage domain-invariant features
- Performs inference using the shared checkpoint across all species
- Writes outputs to the common `runs/` structure for consistent cross-pipeline evaluation

## How to run

### 1) Configure
Edit `config.yaml` to specify paths and experiment parameters.

Key items to check (names may differ depending on your config):
- `data_root`: path to pooled image patches
- `ann_path` / `coco_json`: path to COCO annotations (if used)
- `domain_labels`: mapping from samples to domain/species IDs
- `grl_lambda`: strength of gradient reversal
- `out_dir`: output base directory (recommended: `runs/domain_adaptive/`)
- `seed`: random seed
- `device`: `cuda` or `cpu`

### 2) Train
From the repository root:

    python pipelines/domain_adaptive/train.py --config pipelines/domain_adaptive/config.yaml

Expected behaviour:
- Trains a single domain-adaptive model using all species
- Optimises both segmentation loss and domain classification loss
- Saves checkpoints under:

    runs/domain_adaptive/<experiment_id>/ckpt/

### 3) Inference
From the repository root:

    python pipelines/domain_adaptive/inference.py --config pipelines/domain_adaptive/config.yaml

Expected behaviour:
- Loads the shared domain-adaptive checkpoint
- Runs inference across all species
- Writes predictions under:

    runs/domain_adaptive/<experiment_id>/preds/

## Outputs (convention)

This pipeline follows the shared output convention under `runs/`:

    runs/
      domain_adaptive/
        <experiment_id>/
          ckpt/      trained checkpoints
          preds/     predicted instance masks (and/or COCO-style outputs)
          scores/    evaluation outputs (optional; usually written by evaluation/)

`evaluation/score_all.py` assumes outputs are organised consistently across pipelines.

## Notes

- This pipeline explicitly trades off within-species optimisation for improved cross-species consistency.
- The adversarial objective encourages domain invariance, which may slightly reduce peak performance on any single species.
- Jupyter notebooks under `notebooks/domain_adaptive/` are provided for inspection and figure generation; they are not the authoritative implementation.

## Paper mapping

- Model name in paper: **Domain-adaptive model**
- Key components: Gradient reversal layer, domain classifier
- Corresponding directory: `pipelines/domain_adaptive/`
- Related figures/tables: "@@@"
