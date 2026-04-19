
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipelines/per_species/inference.py

Derived from:
- 04_single_frame_segmentation_ps.ipynb
- 05_predict_multi_frame_ps.ipynb

Purpose:
- Load a trained per-species Mask R-CNN checkpoint
- Run single-frame or multi-frame inference
- Save raw and cleaned prediction images
- Optionally export cleaned predictions to COCO JSON

CLI arguments:
- --species
- --image-size
- --dataset-root
- --gpu-id
- --checkpoint-file
- --mode                  ("single" or "multi")
- --input-image
- --test-index
- --output-root
- --score-threshold
- --mask-threshold
- --export-coco

Notes:
- This script preserves notebook behavior as closely as practical.
- Single-frame mode is mainly for inspection.
- Multi-frame mode is the main full-image prediction path.
- COCO export requires src.create_annotations to be available.
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
from PIL import Image
from skimage.measure import label, regionprops_table
from skimage.segmentation import expand_labels
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--species", type=str, default="acapod")
    p.add_argument("--image-size", type=int, default=256, choices=[256, 512])
    p.add_argument("--dataset-root", type=str, default="dataset")
    p.add_argument("--gpu-id", type=int, default=0)

    p.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="Path to trained checkpoint tar file.",
    )

    p.add_argument("--mode", type=str, default="single", choices=["single", "multi"])
    p.add_argument(
        "--input-image",
        type=str,
        default=None,
        help="Optional explicit input image path. If omitted, script uses notebook-style defaults.",
    )
    p.add_argument(
        "--test-index",
        type=int,
        default=1,
        help="Index of test image for single mode if --input-image is not specified.",
    )

    p.add_argument(
        "--output-root",
        type=str,
        default="./inference_results_without_clearborder",
    )

    p.add_argument("--score-threshold", type=float, default=0.7)
    p.add_argument("--mask-threshold", type=float, default=0.7)
    p.add_argument("--export-coco", action="store_true")

    return p.parse_args()


# ---------------------------
# Model
# ---------------------------

def get_model(device: torch.device):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_detections_per_img=200,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=1 + 4
    )
    model.to(device)
    return model


def load_checkpoint(checkpoint, model, optimizer, lr: float) -> None:
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
    Equivalent to notebook predict_multi_frame() behavior:
    - resize patch/frame to image_size
    - run model
    - convert labels to gray values
    - expand labels slightly
    """
    images = cv2.resize(frame, image_size, cv2.INTER_LINEAR) / 255.0
    images = torch.as_tensor(images, dtype=torch.float32).unsqueeze(0)  # [1, H, W, C]
    images = images.permute(0, 3, 1, 2)  # [1, C, H, W]
    images = [image.to(device) for image in images]

    with torch.no_grad():
        pred = model(images)

    label_to_gray = {1: 64, 2: 128, 3: 192, 4: 255}

    height, width = images[0].shape[1:]
    im_label = np.zeros((height, width), dtype=np.uint8)

    for i in range(len(pred[0]["masks"])):
        mask_tensor = pred[0]["masks"][i]
        score_val = pred[0]["scores"][i].item()
        label_val = pred[0]["labels"][i].item()

        if score_val > score_threshold and label_val in label_to_gray:
            if mask_tensor.ndim == 3:
                msk = mask_tensor[0].detach().cpu().numpy()
            else:
                msk = mask_tensor.detach().cpu().numpy()

            im_label[msk > mask_threshold] = label_val

    im_label_expanded = expand_labels(im_label, distance=1)

    vectorized = np.vectorize(lambda x: label_to_gray.get(x, 0))
    im_gray = vectorized(im_label_expanded).astype(np.uint8)

    return im_gray


# ---------------------------
# Paths / defaults
# ---------------------------

def default_test_dir(dataset_root: str, species: str, image_size: int) -> str:
    return f"{dataset_root}/{species}/png_{image_size}_thin/image/test"


def default_performance_check_image(dataset_root: str, species: str) -> str:
    sp_to_filename = {
        "acapod": "acapod2_01_img.png",
        "alllit": "alllit2_01_img.png",
        "burspi": "burspi1_01_img.png",
        "euclon": "euclon1_01_img.png",
        "exostr": "exostr8_01_img.png",
        "hakdac": "hakdac2_01_img.png",
    }
    filename = sp_to_filename[species]
    return os.path.join(dataset_root, "performance_check", "image", filename)


# ---------------------------
# Single-frame inference
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


# ---------------------------
# Multi-frame inference
# ---------------------------

def get_tiling_params(image_size: int):
    """
    Notebook logic:
    - 512 -> pad=64, sz=384, range(4)
    - 256 -> pad=32, sz=192, range(8)
    """
    if image_size == 512:
        pad = 64
        sz = 384
        l = list(range(4))
    elif image_size == 256:
        pad = 32
        sz = 192
        l = list(range(8))
    else:
        raise ValueError(f"Unsupported image_size: {image_size}")
    return pad, sz, l


def run_multi_frame(
    model,
    image_path: str,
    image_size: int,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
) -> np.ndarray:
    pad, sz, l = get_tiling_params(image_size)

    img_in = cv2.imread(image_path)
    if img_in is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_padded = cv2.copyMakeBorder(
        img_in, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    img_pred = np.zeros_like(img_in[:, :, 0], dtype=np.uint8)

    start = time.time()

    for i, j in itertools.product(l, l):
        img_patch = img_padded[
            sz * i : sz * i + image_size,
            sz * j : sz * j + image_size,
            :
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
    print(f"GPU time = {end - start:.3f} sec")

    return img_pred


# ---------------------------
# Cleaning / post-processing
# ---------------------------

def clean_prediction(img_pred: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Notebook logic:
    - label connected components
    - expand labels
    - assign dominant gray level per connected component
    - reconstruct cleaned annotation image
    """
    new_labels = label(img_pred > 0)

    props = regionprops_table(
        new_labels,
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

    def get_gray(region_label):
        output = img_pred[new_labels == region_label]
        out = output[np.isin(output, [64, 128, 192, 255])]
        if len(out) == 0:
            return 0
        gl = np.argmax(np.bincount(out))
        return gl

    label_list = np.unique(new_labels)[1:]
    lbl_gray = [get_gray(x) for x in label_list]
    df["gray"] = lbl_gray

    new_labels_copy = copy.deepcopy(new_labels).flatten()
    new_annotation = np.zeros((len(new_labels_copy)), dtype=np.uint8)

    replace_func = {
        label_id: gray_val
        for gray_val, label_id in zip(list(df["gray"].values), list(df["label"].values))
    }

    replace_result = np.asarray(list(map(replace_func.get, new_labels_copy.flatten())))
    result_index = np.where(replace_result != None)[0]
    new_annotation[result_index] = replace_result[result_index].astype(np.uint8)

    newly_annotated_img = new_annotation.reshape((new_labels.shape))

    return newly_annotated_img, df


# ---------------------------
# Optional COCO export
# ---------------------------

def export_coco_if_requested(cleaned_img: np.ndarray, data_img: str, out_json: str) -> None:
    """
    Uses src.create_annotations if available.
    """
    try:
        from src.create_annotations import (
            get_coco_json_format,
            info,
            create_category_annotation,
            singleshot_annotations_info,
        )
    except ImportError:
        raise ImportError(
            "src.create_annotations could not be imported. "
            "COCO export requires that module to be available."
        )

    category_ids = {
        "stuff": 0,
        "vessel": 1,
        "ray": 2,
        "fiber": 3,
        "parenchyma": 4,
    }

    category_colors = {
        "(0, 0, 0)": 0,
        "(64, 64, 64)": 1,
        "(128, 128, 128)": 2,
        "(192, 192, 192)": 3,
        "(255, 255, 255)": 4,
    }

    multipolygon_ids = []

    coco_format = get_coco_json_format()
    coco_format["info"] = info()
    coco_format["categories"] = create_category_annotation(category_ids)

    coco_format["images"], coco_format["annotations"], annotation_cnt = singleshot_annotations_info(
        cleaned_img, data_img, category_ids, category_colors, multipolygon_ids
    )

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as outfile:
        json.dump(coco_format, outfile)

    print(f"Created {annotation_cnt} annotations and saved to: {out_json}")


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    learning_rate = 1e-5
    weight_decay = 5e-4

    model = get_model(device=device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    load_checkpoint(checkpoint, model, optimizer, learning_rate)
    model.eval()
    model.train(False)

    if args.mode == "single":
        if args.input_image is not None:
            image_path = args.input_image
        else:
            test_dir = default_test_dir(args.dataset_root, args.species, args.image_size)
            test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".png")])
            if len(test_files) == 0:
                raise RuntimeError(f"No png files found in {test_dir}")
            image_path = os.path.join(test_dir, test_files[args.test_index])

        pred_raw = run_single_frame(
            model=model,
            image_path=image_path,
            image_size=args.image_size,
            device=device,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
        )

    else:  # multi
        if args.input_image is not None:
            image_path = args.input_image
        else:
            image_path = default_performance_check_image(args.dataset_root, args.species)

        pred_raw = run_multi_frame(
            model=model,
            image_path=image_path,
            image_size=args.image_size,
            device=device,
            score_threshold=args.score_threshold,
            mask_threshold=max(args.mask_threshold, 0.9),  # preserves notebook tendency in multi-frame
        )

    # output structure
    ckpt_stem = Path(args.checkpoint_file).stem
    sub_dir = os.path.join(
        args.output_root,
        f"{args.image_size}_per_species_{args.species}_{ckpt_stem}",
    )
    os.makedirs(sub_dir, exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "prediction_raw"), exist_ok=True)
    os.makedirs(os.path.join(sub_dir, "prediction_cleaned"), exist_ok=True)

    fnam = Path(image_path).stem

    raw_path = os.path.join(sub_dir, "prediction_raw", f"{fnam}_prediction_raw.png")
    Image.fromarray(pred_raw.astype(np.uint8)).save(raw_path)
    print(f"Saved raw prediction to: {raw_path}")

    cleaned_img, df = clean_prediction(pred_raw)

    cleaned_path = os.path.join(sub_dir, "prediction_cleaned", f"{fnam}_pred_cleaned.png")
    Image.fromarray(cleaned_img.astype(np.uint8)).save(cleaned_path)
    print(f"Saved cleaned prediction to: {cleaned_path}")

    props_csv = os.path.join(sub_dir, f"{fnam}_regionprops.csv")
    df.to_csv(props_csv, index=False)
    print(f"Saved regionprops table to: {props_csv}")

    if args.export_coco:
        out_json = os.path.join(sub_dir, "prediction_cleaned_json", f"{fnam}_annot_coco.json")
        export_coco_if_requested(cleaned_img, image_path, out_json)


if __name__ == "__main__":
    main()
