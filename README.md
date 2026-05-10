# Xylem instance segmentation

This repository provides image-analysis code for instance segmentation of xylem cells across multiple species, together with downstream evaluation and network-analysis notebooks.

The repository contains:

1. segmentation pipelines for three modelling strategies;
2. notebooks corresponding to the original analysis workflow;
3. cross-pipeline evaluation notebooks; and
4. network-analysis notebooks for analysing xylem cell adjacency, vessel–axial parenchyma connectivity, and vessel–ray graph distances.

## Paper information

This code accompanies the following paper:

- **Title:** "@@@"
- **Authors:** "@@@"
- **Journal / Preprint:** "@@@"
- **Year:** "@@@"
- **DOI / arXiv:** "@@@"

## Overview

The segmentation analysis compares three modelling strategies described in the paper.

### Per-species model

Separate Mask R-CNN models are trained independently for each species using only image patches derived from that species.

This approach serves as a within-domain benchmark because training and evaluation data share the same species-specific image characteristics.

### Mixed-species model

A single Mask R-CNN model is trained using pooled image patches from all species.

This model does not explicitly use species or domain labels and serves as a standard multi-species baseline.

### Domain-adaptive model

A single Mask R-CNN model is trained using pooled multi-species data with an adversarial domain-adaptation framework.

This model incorporates a gradient reversal layer and a domain classifier to encourage domain-invariant feature learning across species.

## Repository structure

```text
xylem-instance-segmentation/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ requirements.txt or environment.yml
│
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ metrics/
│  └─ utils/
│
├─ pipelines/
│  ├─ per_species/
│  │  ├─ README.md
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  └─ inference.py
│  │
│  ├─ mixed_species/
│  │  ├─ README.md
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  └─ inference.py
│  │
│  └─ domain_adaptive/
│     ├─ README.md
│     ├─ preprocess.py
│     ├─ train.py
│     └─ inference.py
│
├─ notebooks/
│  ├─ per_species/
│  ├─ mixed_species/
│  ├─ domain_adaptive/
│  ├─ network_analysis/
│  │  ├─ README.md
│  │  ├─ 00_visualise_partial_graph.ipynb
│  │  ├─ 01_vap_connectivity.ipynb
│  │  └─ 02_vessel_ray_distance_path_composition.ipynb
│  │
│  └─ evaluation/
│     └─ 00_score_comparison.ipynb
│
├─ network_analysis_outputs/
├─ evaluation_outputs/
└─ examples/
```

## Main components

### `src/`

The `src/` directory contains shared source code used by the segmentation pipelines where applicable.

Typical contents include:

- dataset handling;
- model definitions;
- metric calculation;
- post-processing utilities; and
- helper functions.

Some pipeline-specific logic is kept inside the corresponding `pipelines/` subdirectory because the three modelling strategies use slightly different data structures and execution workflows.

### `pipelines/`

The `pipelines/` directory contains executable scripts for the three segmentation workflows.

Each pipeline contains:

- `preprocess.py`: prepares image and mask patches and, where applicable, COCO-style annotation files;
- `train.py`: trains the Mask R-CNN model;
- `inference.py`: runs inference using a trained checkpoint;
- `README.md`: explains the pipeline-specific input structure and execution procedure.

The scripts are designed to be run from the command line. User-specific values such as dataset location, species name, image size, checkpoint file, and output directory are supplied as command-line arguments.

### `notebooks/`

The `notebooks/` directory contains the original or cleaned analysis notebooks.

The segmentation notebooks are kept to document the exploratory and original execution workflow. The corresponding `pipelines/` scripts provide cleaner command-line implementations where available.

The network-analysis notebooks are used for downstream structural analysis and figure generation.

### `notebooks/network_analysis/`

This directory contains notebooks for graph-based analysis of xylem cell adjacency.

The recommended structure is:

```text
notebooks/network_analysis/
├─ README.md
├─ 00_visualise_partial_graph.ipynb
├─ 01_vap_connectivity.ipynb
└─ 02_vessel_ray_distance_path_composition.ipynb
```

These notebooks analyse:

1. partial vessel–parenchyma–ray graph visualisation;
2. vessel–axial parenchyma connectivity; and
3. vessel–ray graph distance and vessel-to-vessel shortest-path composition.

Outputs are saved to:

```text
network_analysis_outputs/
```

Typical outputs include:

```text
network_analysis_outputs/
├─ G_vpr_panels.pdf
├─ df_for_weighted_degree.csv
├─ vessel_level_connectivity_summary.csv
├─ weighted_degree_scatter_with_means.pdf
├─ v_to_nearest_ray_distances_with_hops.csv
├─ hops_hist_vertical.png
├─ hops_hist_vertical.pdf
├─ vessel_ray_distance_summary_by_species.csv
├─ vv_shortest_paths_composition_LITE.csv
├─ vv_avg_composition_by_species_LITE.png
└─ vv_avg_composition_by_species_LITE.pdf
```

### `notebooks/evaluation/`

This directory contains notebooks for comparing segmentation performance across the three modelling strategies.

The main notebook is:

```text
notebooks/evaluation/00_score_comparison.ipynb
```

This notebook aggregates score CSV files generated by the segmentation pipelines and produces comparison figures and summary tables.

Outputs are saved to:

```text
evaluation_outputs/
```

Typical outputs include:

```text
evaluation_outputs/
├─ all_model_scores.csv
├─ mean_dice_scores.csv
├─ mean_iou_scores.csv
├─ accuracy_scores.csv
├─ mean_dice.pdf
├─ mean_iou.pdf
├─ accuracy.pdf
├─ per_class_dice_heatmap.pdf
├─ per_class_dice_heatmap.png
├─ per_class_iou_heatmap.pdf
└─ per_class_iou_heatmap.png
```

## Example usage

### Per-species pipeline

Preprocessing:

```bash
python pipelines/per_species/preprocess.py \
  --dataset-root @@@ \
  --species @@@ \
  --image-size 256
```

Training:

```bash
python pipelines/per_species/train.py \
  --dataset-root @@@ \
  --species @@@ \
  --image-size 256 \
  --epochs @@@ \
  --batch-size @@@ \
  --checkpoint-file @@@
```

Inference:

```bash
python pipelines/per_species/inference.py \
  --dataset-root @@@ \
  --species @@@ \
  --image-size 256 \
  --checkpoint-file @@@ \
  --output-root @@@
```

### Mixed-species pipeline

Preprocessing:

```bash
python pipelines/mixed_species/preprocess.py \
  --dataset-root @@@ \
  --image-size 256
```

Training:

```bash
python pipelines/mixed_species/train.py \
  --dataset-root @@@ \
  --image-size 256 \
  --epochs @@@ \
  --batch-size @@@ \
  --checkpoint-file @@@
```

Inference:

```bash
python pipelines/mixed_species/inference.py \
  --dataset-root @@@ \
  --image-size 256 \
  --checkpoint-file @@@ \
  --output-root @@@
```

### Domain-adaptive pipeline

Preprocessing:

```bash
python pipelines/domain_adaptive/preprocess.py \
  --dataset-root @@@ \
  --image-size 256
```

Training:

```bash
python pipelines/domain_adaptive/train.py \
  --dataset-root @@@ \
  --image-size 256 \
  --epochs @@@ \
  --batch-size @@@ \
  --checkpoint-file @@@
```

Inference:

```bash
python pipelines/domain_adaptive/inference.py \
  --dataset-root @@@ \
  --image-size 256 \
  --checkpoint-file @@@ \
  --output-root @@@
```

## Data and outputs

Large raw microscopy images, generated training patches, trained checkpoints, and full inference outputs are not necessarily tracked in this repository.

The expected input data structure is documented in each pipeline-specific README.

Generated outputs should be written to local output directories such as:

```text
runs/
network_analysis_outputs/
evaluation_outputs/
```

Large files such as model checkpoints and full prediction outputs should not be committed to GitHub unless explicitly intended.

Recommended ignored files include:

```text
runs/
__pycache__/
.ipynb_checkpoints/
*.ckpt
*.pth
*.pt
*.tar
*.log
```

## Data availability

The original microscopy images used in the study are "@@@".

- **Data availability status:** "@@@"
- **Restrictions, if any:** "@@@"
- **Example data:** The `examples/` directory provides minimal example data or format specifications where available.

## Experimental environment

The experiments reported in the paper were conducted using the following environment:

- Python: 3.12
- PyTorch: 2.4.1
- torchvision: 0.19.1
- Operating system: Ubuntu 20.04
- GPU: NVIDIA RTX A6000, 49 GB VRAM
- cuDNN: 9.3.0

Exact dependency versions are provided in:

```text
requirements.txt
```

or:

```text
environment.yml
```

## Requirements

Core requirements include:

- Python >= 3.9
- PyTorch
- torchvision
- CUDA-compatible GPU, optional but recommended for training
- NumPy
- pandas
- scikit-image
- OpenCV
- matplotlib
- seaborn
- networkx
- pycocotools
- albumentations

See `requirements.txt` or `environment.yml` for the exact environment.

## Reproducibility notes

The repository contains both notebooks and command-line scripts.

- The `notebooks/` directory documents the original analysis workflow, visualisation, evaluation, and network-analysis procedures.
- The `pipelines/` directory contains cleaned executable scripts for preprocessing, training, and inference.
- The network-analysis and evaluation notebooks may require intermediate outputs generated by the segmentation pipelines.

For full reproduction, users should check:

1. input data paths;
2. image size;
3. species name, where applicable;
4. checkpoint file path;
5. output directory;
6. generated score CSV paths; and
7. graph files used in network analysis.

## License

This repository is released under the "@@@" License.

## Citation

If you use this code, please cite the corresponding paper:

"@@@"

Alternatively, see `CITATION.cff` for citation metadata.
