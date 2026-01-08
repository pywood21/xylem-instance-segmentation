# Xylem instance segmentation

This repository provides three image-analysis pipelines for instance segmentation of xylem cells across multiple species, together with a unified evaluation framework.

## Overview

This code accompanies the following paper:

- Title: "@@@"
- Authors: "@@@"
- Journal / Preprint: "@@@"
- Year: "@@@"
- DOI / arXiv: "@@@"

We implement three modelling strategies described in the paper:

1. Per-species model  
   Independent models trained separately for each species using only image patches derived from that species. This serves as an upper-bound benchmark for within-domain performance, as both training and evaluation data share identical domain characteristics.

2. Mixed-species model  
   A single model trained on pooled multi-species data without explicit consideration of domain labels, representing a standard multi-domain baseline.

3. Domain-adaptive model  
   A single model trained on pooled multi-species data using an adversarial domain adaptation framework with a gradient reversal layer to encourage domain-invariant feature learning.

Each pipeline can be executed independently and evaluated using a common scoring framework.

## Repository structure

    src/                Shared source code (datasets, models, metrics, post-processing)
    pipelines/          Executable training and inference pipelines
      per_species/
      mixed_species/
      domain_adaptive/
    notebooks/          Notebooks for data preparation, visualisation, and figure generation
    evaluation/         Cross-pipeline evaluation and aggregation
    examples/           Minimal example data and format specifications

## Reproducibility

Training:

    python pipelines/per_species/train.py --config pipelines/per_species/config.yaml

Inference:

    python pipelines/per_species/inference.py --config pipelines/per_species/config.yaml

Cross-pipeline evaluation:

    python evaluation/score_all.py --runs_dir runs/

## Output format

All pipelines write results to a common directory structure under runs/:

    runs/
      per_species/
        <experiment_id>/
          ckpt/           Trained model checkpoints
          preds/          Predicted instance masks
          scores/         Evaluation outputs (e.g. scores.csv)

The evaluation scripts assume this structure when aggregating results across pipelines.

## Notebooks

Jupyter notebooks are provided for demonstration, data inspection, and figure generation.  
They are not required for reproducing the main results, which can be obtained by running the pipeline scripts directly.

Notebook directories mirror the pipeline names (e.g. notebooks/per_species/) to indicate correspondence, but notebooks do not constitute the authoritative implementation.

## Data availability

The original microscopy images used in the study are "@@@".

- Data availability status: "@@@"
- Restrictions (if any): "@@@"

The examples/ directory provides minimal example data and specifications of expected input and annotation formats to allow code execution and inspection.

## Requirements

- Python >= 3.9
- PyTorch "@@@"
- CUDA "@@@" (required / optional)

Exact dependency versions are listed in requirements.txt or environment.yml.

## License

This repository is released under the MIT License.

## Citation

If you use this code, please cite the corresponding paper:

    "@@@"

(Alternatively, see CITATION.cff for citation metadata.)










