# Xylem instance segmentation and network analysis

Code accompanying:

Izumi, N., Wright, I. J., Chhajed, S., Ziemińska, K., and Sugiyama, J.
From segmentation to networks: Mask R-CNN and graph analysis uncover
vessel–parenchyma organisation in angiosperm xylem.
DOI: https://doi.org/10.1007/s00468-026-02838-8

## Repository structure

- PS_model/: Per-species model workflow
- MS_model/: Mixed-species model workflow
- DA_model/: Domain-adaptive model workflow
- network_analysis/: Cell-contact graphs and connectivity analyses
- score_comparison/: Comparison of saved model evaluation scores
- requirements.txt: Python dependencies

## Installation and execution

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The workflows were developed with Python 3.12, PyTorch 2.4.1, and
torchvision 0.19.1. GPU acceleration is recommended for training and inference.
Run numbered notebooks in workflow order within the relevant model directory.
Update input, checkpoint, and output paths to match your local setup.

Run score_comparison/00_score_comparison.ipynb from the repository root,
which contains PS_model/, MS_model/, and DA_model/.

## Model performance evaluation

The PS, MS, and DA evaluation workflows use the same preprocessing:

- Label connected non-background prediction regions and assign each region
  its majority non-background prediction class.
- Reconstruct the evaluation mask using the original, unexpanded regions.
  Expanded labels are used only for supplementary visualisation in these
  prediction notebooks, not for geometric measurements or score calculation.

Accuracy is calculated over pixels with non-background ground-truth labels.
Dice and IoU are calculated separately for vessel, fibre, axial parenchyma,
and ray parenchyma. Background-class scores are omitted, but non-background
predictions on ground-truth background pixels count as false positives.
IoU scores are included for exploratory comparison and are not reported
in the manuscript.

## Network analysis

The network results reported in the manuscript are based on post-processed
ground-truth masks. Network-specific label expansion and border handling
are separate from model performance evaluation. Predicted masks should be
reviewed and corrected before use in quantitative anatomy or network analysis.

## Data availability and citation

Public example image and mask data are limited to Allocasuarina.
The full training dataset and trained model weights are not included.
See the manuscript for further data availability details.

Please cite the accompanying manuscript when using this code.
