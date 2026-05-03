
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
da_inference.py

Derived from:
- 04_NI_single_frame_segmentation.ipynb
- 05_no_clear_border.ipynb

Purpose:
- Load a trained domain-adaptive / multidomain Mask R-CNN checkpoint
- Run single-frame, multi-frame, or batch multi-frame inference
- Save raw prediction images, cleaned prediction images, and regionprops tables
- Optionally export cleaned predictions to COCO JSON

Expected input structure for single-frame inference:
    dataset/png_256/image/test/
    dataset/png_512/image/test/

or, if --data-suffix _thin is used:
    dataset/png_256_thin/image/test/
    dataset/png_512_thin/image/test/

Expected performance-check structure for batch inference:
    dataset/performance_check/image/*.png

Gray value mapping follows the current preprocess / COCO convention:
    vessel      -> 255
    ray         -> 191
    fiber       -> 127
    parenchyma  -> 63
    background  -> 0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.measure import label, regionprops_table
from skimage.segmentation import expand_labels
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Dataset root. Default: dataset",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        choices=[256, 512],
        help="Patch size used for inference. Default: 256",
    )

    parser.add_argument(
        "--data-suffix",
        type=str,
        default="",
        help=(
            "Suffix after png_<image_size>. "
            "Use '' for dataset/png_256 and '_thin' for dataset/png_256_thin. "
            "Default: ''."
        ),
    )

    parser.add_argument("--gpu-id", type=int, default=0)

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to trained domain-adaptive checkpoint file.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "multi", "batch"],
        help="Inference mode: single, multi, or batch.",
    )

    parser.add_argument(
        "--input-image",
        type=str,
        default=None,
        help=(
            "Optional explicit input image path. "
            "Required or strongly recommended for multi-frame inference."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Input directory for batch mode. "
            "Default: dataset/performance_check/image"
        ),
    )

    parser.add_argument(
        "--test-index",
        type=int,
        default=0,
        help="Index of test image for single mode if --input-image is not specified.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="./inference_results_domain_adaptive",
        help="Output directory root.",
    )

    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.7)

    parser.add_argument(
        "--clean-expand-distance",
        type=int,
        default=15,
        help=(
            "Expansion distance used during connected-component cleaning. "
            "Default follows 05_no_clear_border.ipynb: 15"
        ),
    )

    parser.add_argument(
        "--export-coco",
        action="store_true",
        help="Export cleaned prediction as COCO JSON.",
    )

    parser.add_argument(
        "--num-domains",
        type=int,
        default=6,
        help="Number of domains/species used in domain-adaptive training. Default: 6",
    )

    return parser.parse_args()


# ---------------------------
# Domain-adaptive model
# ---------------------------

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output.clone(),


class DomainClassifier(nn.Module):
    def __init__(self, in_features: int = 256, hidden_size: int = 128, num_domains: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_domains),
        )

    def forward(self, x):
        return self.classifier(x)


class CustomMaskRCNN(nn.Module):
    """
    Wrapper used for domain-adaptive training.

    During inference, this forwards directly to the underlying Mask R-CNN model.
    The wrapper is still needed because the checkpoint state_dict contains:
        model.*
        domain_classifier.*
    """

    def __init__(self, num_classes: int = 5, num_domains: int = 6):
        super().__init__()

        self.model = maskrcnn_resnet50_fpn_v2(
            weights="DEFAULT",
            box_detections_per_img=200,
        )

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features,
            num_classes,
        )

        self.domain_classifier = DomainClassifier(
            in_features=256,
            hidden_size=128,
            num_domains=num_domains,
        )

        self.grl = GradientReversalLayer.apply

    def forward(self, images, targets=None):
        if not self.training:
            return self.model(images)

        if isinstance(images, list):
            images_tensor = torch.stack(images)
        elif hasattr(images, "tensors"):
            images_tensor = images.tensors
        else:
            images_tensor = images

        features = self.model.backbone(images_tensor)
        feat_map = features["0"]

        pooled_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(
            feat_map.size(0), -1
        )

        reversed_feat = self.grl(pooled_feat)
        domain_pred = self.domain_classifier(reversed_feat)

        if isinstance(targets, list) and len(targets) > 0 and "domain_id" in targets[0]:
            domain_labels = torch.stack([t["domain_id"] for t in targets]).to(
                domain_pred.device
            )
            domain_loss = F.cross_entropy(domain_pred, domain_labels)
        else:
            domain_loss = torch.tensor(0.0, device=domain_pred.device)

        loss_dict = self.model(images, targets)

        if isinstance(loss_dict, dict):
            loss_dict["domain_loss"] = domain_loss

        return loss_dict


def get_model(device: torch.device, num_domains: int):
    model = CustomMaskRCNN(
        num_classes=5,
        num_domains=num_domains,
    )
    model.to(device)
    return model


# ---------------------------
# Inference core
# ---------------------------

def predict_patch(
    model,
    frame: np.ndarray,
    image_size: tuple[int, int],
    score_threshold: float = 0.7,
    mask_threshold: float = 0.7,
    device: torch.device | str = "cuda",
) -> np.ndarray:
    """
    Run Mask R-CNN prediction on one image patch/frame.

    Label-to-gray mapping:
        1 -> 255  vessel
        2 -> 191  ray
        3 -> 127  fiber
        4 -> 63   parenchyma
    """
    image = cv2.resize(frame, image_size, cv2.INTER_LINEAR) / 255.0
    image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    images = [image[0].to(device)]

    with torch.no_grad():
        pred = model(images)

    label_to_gray = {
        1: 255,
        2: 191,
        3: 127,
        4: 63,
    }

    height, width = images[0].shape[1:]
    im_label = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(pred[0]["masks"])):
        mask_tensor = pred[0]["masks"][i]
        score_val = pred[0]["scores"][i].item()
        label_val = pred[0]["labels"][i].item()

        if score_val > score_threshold and label_val in label_to_gray:
            if mask_tensor.ndim == 3:
                mask = mask_tensor[0].detach().cpu().numpy()
            else:
                mask = mask_tensor.detach().cpu().numpy()

            im_label[mask > mask_threshold] = label_val

    im_label_expanded = expand_labels(im_label, distance=1)

    vectorized = np.vectorize(lambda x: label_to_gray.get(x, 0))
    im_gray = vectorized(im_label_expanded).astype(np.uint8)

    return im_gray


# ---------------------------
# Input path helpers
# ---------------------------

def default_test_dir(dataset_root: str, image_size: int, data_suffix: str) -> str:
    return f"{dataset_root}/png_{image_size}{data_suffix}/image/test"


def default_performance_check_dir(dataset_root: str) -> str:
    return f"{dataset_root}/performance_check/image"


def get_single_input_image(
    dataset_root: str,
    image_size: int,
    data_suffix: str,
    input_image: str | None,
    test_index: int,
) -> str:
    if input_image is not None:
        return input_image

    test_dir = default_test_dir(
        dataset_root=dataset_root,
        image_size=image_size,
        data_suffix=data_suffix,
    )

    test_files = sorted(
        [f for f in os.listdir(test_dir) if f.endswith(".png")]
    )

    if len(test_files) == 0:
        raise RuntimeError(f"No png files found in {test_dir}")

    if test_index >= len(test_files):
        raise IndexError(
            f"--test-index {test_index} is out of range. "
            f"{test_dir} contains {len(test_files)} PNG files."
        )

    return os.path.join(test_dir, test_files[test_index])


def get_batch_input_images(dataset_root: str, input_dir: str | None) -> list[str]:
    if input_dir is None:
        input_dir = default_performance_check_dir(dataset_root)

    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_path}\n"
            "For batch mode, specify --input-dir or prepare dataset/performance_check/image."
        )

    files = sorted(input_path.glob("*.png"))

    if len(files) == 0:
        raise RuntimeError(f"No PNG files found in: {input_path}")

    return [str(f) for f in files]


# ---------------------------
# Single-frame and multi-frame inference
# ---------------------------

def run_single_frame(
    model,
    image_path: str,
    image_size: int,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
) -> np.ndarray:
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    pred = predict_patch(
        model=model,
        frame=img,
        image_size=(image_size, image_size),
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        device=device,
    )

    return pred


def get_tiling_params(image_size: int):
    """
    Notebook-style tiling parameters.

    For 512:
        pad = 64
        inner tile size = 384
        4 x 4 tiles

    For 256:
        pad = 32
        inner tile size = 192
        8 x 8 tiles
    """
    if image_size == 512:
        pad = 64
        sz = 384
        tile_range = list(range(4))
    elif image_size == 256:
        pad = 32
        sz = 192
        tile_range = list(range(8))
    else:
        raise ValueError(f"Unsupported image_size: {image_size}")

    return pad, sz, tile_range


def run_multi_frame(
    model,
    image_path: str,
    image_size: int,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
) -> np.ndarray:
    pad, sz, tile_range = get_tiling_params(image_size)

    img_in = cv2.imread(image_path)

    if img_in is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_padded = cv2.copyMakeBorder(
        img_in,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    img_pred = np.zeros_like(img_in[:, :, 0], dtype=np.uint8)

    start = time.time()

    for i, j in itertools.product(tile_range, tile_range):
        img_patch = img_padded[
            sz * i : sz * i + image_size,
            sz * j : sz * j + image_size,
            :,
        ]

        res = predict_patch(
            model=model,
            frame=img_patch,
            image_size=(image_size, image_size),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            device=device,
        )

        res = res[pad : pad + sz, pad : pad + sz]
        img_pred[sz * i : sz * i + sz, sz * j : sz * j + sz] = res

    end = time.time()

    print(f"{Path(image_path).name}: GPU time = {end - start:.3f} sec")

    return img_pred


# ---------------------------
# Cleaning / post-processing
# ---------------------------

def clean_prediction(
    img_pred: np.ndarray,
    expand_distance: int = 15,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Notebook-style no-clear-border cleaning.

    1. Connected components are created from img_pred > 0.
    2. Labels are expanded by expand_distance for property extraction and gray assignment.
    3. The final cleaned annotation is written back on the unexpanded label map.
    """
    new_labels = label(img_pred > 0)
    expanded_labels = expand_labels(new_labels, distance=expand_distance)

    props = regionprops_table(
        expanded_labels,
        properties=(
            "label",
            "centroid",
            "bbox",
            "orientation",
            "axis_major_length",
            "axis_minor_length",
            "area",
        ),
    )

    df = pd.DataFrame(props)

    valid_gray_values = [255, 191, 127, 63]

    def get_gray(region_label):
        output = img_pred[expanded_labels == region_label]
        out = output[np.isin(output, valid_gray_values)]

        if len(out) == 0:
            return 0

        return int(np.argmax(np.bincount(out)))

    label_list = np.unique(expanded_labels)[1:]
    gray_values = [get_gray(lbl) for lbl in label_list]

    df["gray"] = gray_values

    gray_to_anatomy = {
        0: "stuff",
        255: "vessel",
        191: "ray",
        127: "fiber",
        63: "parenchyma",
    }

    df["anatomy"] = df["gray"].map(gray_to_anatomy)

    new_labels_copy = copy.deepcopy(new_labels).flatten()
    new_annotation = np.zeros_like(new_labels_copy, dtype=np.uint8)

    replace_func = {
        int(label_id): int(gray_val)
        for label_id, gray_val in zip(
            df["label"].values,
            df["gray"].values,
        )
    }

    replace_result = np.asarray(
        list(map(replace_func.get, new_labels_copy.flatten()))
    )

    result_index = np.where(replace_result != None)[0]
    new_annotation[result_index] = replace_result[result_index].astype(np.uint8)

    cleaned_img = new_annotation.reshape(new_labels.shape)

    return cleaned_img, df


# ---------------------------
# Optional COCO export
# ---------------------------

def export_coco_if_requested(
    cleaned_img: np.ndarray,
    data_img: str,
    out_json: str,
) -> None:
    """
    Export cleaned prediction to COCO JSON.

    Requires src.create_annotations.
    """
    try:
        from src.create_annotations import (
            get_coco_json_format,
            info,
            create_category_annotation,
            singleshot_annotations_info,
        )
    except ImportError as e:
        raise ImportError(
            "src.create_annotations could not be imported. "
            "COCO export requires that module to be available."
        ) from e

    category_ids = {
        "stuff": 0,
        "vessel": 1,
        "ray": 2,
        "fiber": 3,
        "parenchyma": 4,
    }

    category_colors = {
        "(255, 255, 255)": 1,
        "(191, 191, 191)": 2,
        "(127, 127, 127)": 3,
        "(63, 63, 63)": 4,
        "(0, 0, 0)": 0,
    }

    multipolygon_ids = []

    coco_format = get_coco_json_format()
    coco_format["info"] = info()
    coco_format["categories"] = create_category_annotation(category_ids)

    (
        coco_format["images"],
        coco_format["annotations"],
        annotation_cnt,
    ) = singleshot_annotations_info(
        cleaned_img,
        data_img,
        category_ids,
        category_colors,
        multipolygon_ids,
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    with open(out_json, "w") as outfile:
        json.dump(coco_format, outfile)

    print(f"Created {annotation_cnt} annotations and saved to: {out_json}")


# ---------------------------
# Output saving
# ---------------------------

def save_outputs(
    image_path: str,
    pred_raw: np.ndarray,
    output_subdir: str,
    export_coco: bool,
    clean_expand_distance: int,
) -> None:
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(os.path.join(output_subdir, "prediction_raw"), exist_ok=True)
    os.makedirs(os.path.join(output_subdir, "prediction_cleaned"), exist_ok=True)
    os.makedirs(os.path.join(output_subdir, "cell_props"), exist_ok=True)

    image_id = Path(image_path).stem

    raw_path = os.path.join(
        output_subdir,
        "prediction_raw",
        f"{image_id}_prediction_raw.png",
    )

    Image.fromarray(pred_raw.astype(np.uint8)).save(raw_path)
    print(f"Saved raw prediction to: {raw_path}")

    cleaned_img, df = clean_prediction(
        pred_raw,
        expand_distance=clean_expand_distance,
    )

    cleaned_path = os.path.join(
        output_subdir,
        "prediction_cleaned",
        f"{image_id}_pred_cleaned.png",
    )

    Image.fromarray(cleaned_img.astype(np.uint8)).save(cleaned_path)
    print(f"Saved cleaned prediction to: {cleaned_path}")

    props_csv = os.path.join(
        output_subdir,
        "cell_props",
        f"{image_id}_cell_props.csv",
    )

    df["image"] = image_id
    df.to_csv(props_csv, index=False)
    print(f"Saved regionprops table to: {props_csv}")

    if export_coco:
        out_json = os.path.join(
            output_subdir,
            "prediction_cleaned_json",
            f"{image_id}_annot_coco.json",
        )

        export_coco_if_requested(
            cleaned_img=cleaned_img,
            data_img=image_path,
            out_json=out_json,
        )


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    model = get_model(
        device=device,
        num_domains=args.num_domains,
    )

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    model.train(False)

    ckpt_stem = Path(args.checkpoint_file).stem

    output_subdir = os.path.join(
        args.output_root,
        f"{args.image_size}{args.data_suffix}_domain_adaptive_{ckpt_stem}",
    )

    if args.mode == "single":
        image_path = get_single_input_image(
            dataset_root=args.dataset_root,
            image_size=args.image_size,
            data_suffix=args.data_suffix,
            input_image=args.input_image,
            test_index=args.test_index,
        )

        pred_raw = run_single_frame(
            model=model,
            image_path=image_path,
            image_size=args.image_size,
            device=device,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
        )

        save_outputs(
            image_path=image_path,
            pred_raw=pred_raw,
            output_subdir=output_subdir,
            export_coco=args.export_coco,
            clean_expand_distance=args.clean_expand_distance,
        )

    elif args.mode == "multi":
        if args.input_image is None:
            raise ValueError("For --mode multi, specify --input-image.")

        pred_raw = run_multi_frame(
            model=model,
            image_path=args.input_image,
            image_size=args.image_size,
            device=device,
            score_threshold=args.score_threshold,
            mask_threshold=max(args.mask_threshold, 0.9),
        )

        save_outputs(
            image_path=args.input_image,
            pred_raw=pred_raw,
            output_subdir=output_subdir,
            export_coco=args.export_coco,
            clean_expand_distance=args.clean_expand_distance,
        )

    elif args.mode == "batch":
        image_paths = get_batch_input_images(
            dataset_root=args.dataset_root,
            input_dir=args.input_dir,
        )

        print(f"Batch inference: {len(image_paths)} images")

        for image_path in image_paths:
            pred_raw = run_multi_frame(
                model=model,
                image_path=image_path,
                image_size=args.image_size,
                device=device,
                score_threshold=args.score_threshold,
                mask_threshold=max(args.mask_threshold, 0.9),
            )

            save_outputs(
                image_path=image_path,
                pred_raw=pred_raw,
                output_subdir=output_subdir,
                export_coco=args.export_coco,
                clean_expand_distance=args.clean_expand_distance,
            )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
