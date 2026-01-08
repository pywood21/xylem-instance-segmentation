# xylem-instance-segmentation

This repository provides three image-analysis pipelines for instance segmentation of xylem cells across multiple species, together with a unified evaluation framework.

## Overview

We implement three modelling strategies described in the paper:

1. Per-species model: independent models trained for each species.
2. Mixed-species model: a single model trained on pooled multi-species data.
3. Domain-adaptive model: a domain-adversarial model trained with gradient reversal.

Each pipeline can be executed independently and evaluated using a common scoring framework.

## Repository structure

- src/ : shared source code (datasets, models, metrics, post-processing)
- pipelines/ : executable training and inference pipelines
  - per_species/
  - mixed_species/
  - domain_adaptive/
- notebooks/ : notebooks for data preparation, visualisation, and figure generation
- evaluation/ : cross-pipeline evaluation and aggregation
- examples/ : minimal example data

## Reproducibility

To train a model:
python pipelines/per_species/train.py --config pipelines/per_species/config.yaml

To run inference:
python pipelines/per_species/inference.py --config pipelines/per_species/config.yaml

To evaluate all pipelines:
python evaluation/score_all.py --runs_dir runs/

## Notebooks

Jupyter notebooks are provided for demonstration, data inspection, and figure generation.
They are not required for reproducing the main results, which can be obtained by running the pipeline scripts.

## Notebooks

Jupyter notebooks are provided for demonstration, data inspection, and figure generation.
They are not required for reproducing the main results, which can be obtained by running the pipeline scripts.

## Requirements

- Python >= 3.9
- PyTorch
- CUDA (optional, for GPU acceleration)

## License

This repository is released under the MIT License.

## Citation

If you use this code, please cite the corresponding paper.









