# Evaluation

This directory contains the **authoritative evaluation scripts** used to compute all quantitative results reported in the manuscript.

## Purpose

- Compute evaluation metrics for all pipelines in a consistent manner
- Aggregate results across species and pipelines
- Write evaluation outputs (CSV files) for downstream analysis and visualisation

All numerical results reported in the paper are derived from scripts in this directory.

## Scripts

- `score_all.py`  
  Computes evaluation metrics for each pipeline based on outputs stored under `runs/`.

- `aggregate.py`  
  Aggregates evaluation results across species and pipelines and writes summary CSV files.

## Inputs

- Model outputs stored under `runs/`
- The directory structure described in the repository README is assumed

## Outputs

- CSV files containing evaluation metrics (e.g. `scores.csv`)
- Aggregated summary tables used for figure and table generation

## Notes

- Jupyter notebooks are **not used** to compute evaluation metrics.
- Notebooks under `notebooks/` are provided only for inspection, comparison, and figure generation.
