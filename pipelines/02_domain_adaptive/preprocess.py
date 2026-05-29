
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
preprocess.py

Derived from:
- 00_multisp_img_msk_adjust.ipynb
- 01_NI_create_custom_coco_dataset.ipynb

Purpose:
- Read pooled multi-species image/mask pairs from dataset/original/
- Resize each original image and mask to 1536 x 1536
- Apply notebook-style mask preprocessing
- Patchify into 256 and 512 patches
- Split patches into train / valid / test within each original image
- Save image and mask patches with notebook-style filenames
- Optionally create COCO JSON annotation files

This preprocessing script is used for the multidomain / domain-adaptive pipeline.

Notebook-style design:
- input image: dataset/original/image/*.png
- input mask : dataset/original/mask/*.png
- output:
    dataset/png_256/image/{train,valid,test}/
    dataset/png_256/mask/{train,valid,test}/
    dataset/png_512/image/{train,valid,test}/
    dataset/png_512/mask/{train,valid,test}/
- output filename:
    {patch_number:04}_{original_image_basename}.png

Example:
    original image:
        dataset/original/image/acapod3_01_img.png

    output image patch:
        dataset/png_256/image/train/0000_acapod3_01_img.png

    output mask patch:
        dataset/png_256/mask/train/0000_acapod3_01_img.png

Notes:
- The output mask uses the same basename as the image, following the notebook.
- The split is performed within each original image, not across all images.
- By default, no random seed is fixed, following the original notebook behaviour.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps
from patchify import patchify


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------
# Mask processing
# ---------------------------

def check_allowed_values(mask: np.ndarray, allowed_values=(0, 64, 128, 192, 255)) -> None:
    unique_values = set(np.unique(mask))
    allowed_values = set(allowed_values)

    if not unique_values.issubset(allowed_values):
        raise ValueError(
            f"Unexpected mask values found: {sorted(unique_values - allowed_values)}"
        )


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Notebook logic:
    pixel > 230 -> 255
    otherwise   -> 0
    """
    return np.where(mask > 230, 255, 0).astype(np.uint8)


def thicken_binary(mask_binary: np.ndarray) -> np.ndarray:
    """
    Notebook logic:
    dilation: 2 iterations
    erosion : 2 iterations
    """
    kernel = np.ones((3, 3), np.uint8)
    thick_binary = cv2.dilate(mask_binary, kernel, iterations=2)
    thick_binary = cv2.erode(thick_binary, kernel, iterations=2)
    thick_binary[thick_binary > 0] = 255
    return thick_binary


def overlay_thick_on_mask(original_mask: np.ndarray, thick_binary: np.ndarray) -> np.ndarray:
    """
    Notebook logic:
    where thick_binary == 255, set mask value to 255;
    otherwise keep the original mask value.
    """
    return np.where(thick_binary == 255, 255, original_mask).astype(np.uint8)


def make_processed_mask(mask_pil: Image.Image) -> Image.Image:
    """
    Full notebook-style mask processing:
    1. binarize mask boundary
    2. thicken binary boundary
    3. overlay thick boundary on original multi-class mask
    4. invert final mask
    """
    mask_arr = np.array(mask_pil.convert("L"))

    check_allowed_values(mask_arr)

    binary = binarize_mask(mask_arr)
    thick_binary = thicken_binary(binary)
    thick_mask = overlay_thick_on_mask(mask_arr, thick_binary)

    thick_mask_pil = Image.fromarray(thick_mask).convert("L")
    inverted_mask = ImageOps.invert(thick_mask_pil)

    return inverted_mask


# ---------------------------
# Pairing and split
# ---------------------------

def list_pairs_by_sorted_order(input_dir: Path):
    """
    Notebook-style pairing:
    - image paths and mask paths are separately sorted
    - img_paths[i] is paired with msk_paths[i]
    - no explicit ID matching is performed
    """
    image_dir = input_dir / "image"
    mask_dir = input_dir / "mask"

    img_paths = sorted(glob.glob(str(image_dir / "*.png")))
    msk_paths = sorted(glob.glob(str(mask_dir / "*.png")))

    if len(img_paths) == 0:
        raise RuntimeError(
            f"No image files found in: {image_dir}\n"
            "Check --dataset-root. It should point to the folder containing "
            "original/image and original/mask."
        )

    if len(msk_paths) == 0:
        raise RuntimeError(
            f"No mask files found in: {mask_dir}\n"
            "Check --dataset-root. It should point to the folder containing "
            "original/image and original/mask."
        )

    if len(img_paths) != len(msk_paths):
        raise RuntimeError(
            f"Image/mask count mismatch: "
            f"{len(img_paths)} images vs {len(msk_paths)} masks"
        )

    return [(Path(img_paths[i]), Path(msk_paths[i])) for i in range(len(img_paths))]


def rand_order(num: int):
    """
    Notebook-style random split.

    For each original image:
    - randomly select int(num * 0.6) patches for train
    - randomly select int(remaining * 0.6) patches for valid
    - use the rest for test

    This approximates:
    train: 60%
    valid: 24%
    test : 16%
    """
    all_ids = set(range(num))
    train_ids = set(random.sample(range(0, num), k=int(num * 0.6)))
    valid_ids = set(
        random.sample(
            list(all_ids - train_ids),
            k=int((num - len(train_ids)) * 0.6),
        )
    )
    test_ids = list(all_ids - train_ids - valid_ids)

    return [list(train_ids), list(valid_ids), test_ids]


# ---------------------------
# Output helpers
# ---------------------------

def ensure_output_dirs(dataset_root: Path, patch_sizes: list[int]) -> None:
    for patch_size in patch_sizes:
        for split in ["train", "valid", "test"]:
            img_dir = dataset_root / f"png_{patch_size}" / "image" / split
            msk_dir = dataset_root / f"png_{patch_size}" / "mask" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            msk_dir.mkdir(parents=True, exist_ok=True)


def save_patch(
    dataset_root: Path,
    patch_size: int,
    split: str,
    filename: str,
    image_patch: np.ndarray,
    mask_patch: np.ndarray,
) -> None:
    img_dir = dataset_root / f"png_{patch_size}" / "image" / split
    msk_dir = dataset_root / f"png_{patch_size}" / "mask" / split

    Image.fromarray(image_patch).convert("RGB").save(img_dir / filename)
    Image.fromarray(mask_patch).convert("L").save(msk_dir / filename)


# ---------------------------
# Main preprocessing
# ---------------------------

def process_dataset(
    dataset_root: Path,
    image_size: int,
    patch_sizes: list[int],
    seed: int | None,
    dry_run: bool,
    create_coco_json: bool,
) -> None:
    if seed is not None:
        random.seed(seed)

    input_dir = dataset_root / "original"
    pairs = list_pairs_by_sorted_order(input_dir)

    if not dry_run:
        ensure_output_dirs(dataset_root, patch_sizes)

    for img_path, msk_path in pairs:
        original_image_basename = img_path.stem
        save_index = f"_{original_image_basename}"

        img = Image.open(img_path).resize((image_size, image_size))
        msk = Image.open(msk_path).resize((image_size, image_size))

        processed_mask = make_processed_mask(msk)

        image_arr = np.array(img)[:, :, :3]
        mask_arr = np.array(processed_mask)

        for patch_size in patch_sizes:
            image_patches = patchify(
                image_arr,
                (patch_size, patch_size, 3),
                step=patch_size,
            )
            image_patches = image_patches.reshape(
                (-1, patch_size, patch_size, 3)
            )

            mask_patches = patchify(
                mask_arr,
                (patch_size, patch_size),
                step=patch_size,
            )
            mask_patches = mask_patches.reshape(
                (-1, patch_size, patch_size)
            )

            num_patches = int(image_size / patch_size) * int(image_size / patch_size)
            split_lists = rand_order(num_patches)

            for split_idx, split in enumerate(["train", "valid", "test"]):
                for patch_number in split_lists[split_idx]:
                    filename = f"{patch_number:04}{save_index}.png"

                    if not dry_run:
                        save_patch(
                            dataset_root=dataset_root,
                            patch_size=patch_size,
                            split=split,
                            filename=filename,
                            image_patch=image_patches[patch_number],
                            mask_patch=mask_patches[patch_number],
                        )

        print(f"[OK] processed: image={img_path.name}, mask={msk_path.name}")

    if create_coco_json and not dry_run:
        for patch_size in patch_sizes:
            create_coco_json_files(
                dataset_root=dataset_root,
                image_size=patch_size,
            )

    print(f"[DONE] processed {len(pairs)} image-mask pairs.")


# ---------------------------
# COCO JSON creation
# ---------------------------

def create_coco_json_files(dataset_root: Path, image_size: int) -> None:
    """
    Reproduces 01_NI_create_custom_coco_dataset.ipynb.

    Creates:
        dataset/png_<image_size>/image/train/train_annot_coco_<image_size>.json
        dataset/png_<image_size>/image/valid/valid_annot_coco_<image_size>.json
        dataset/png_<image_size>/image/test/test_annot_coco_<image_size>.json
    """
    try:
        from src.create_annotations import (
            get_coco_json_format,
            info,
            create_category_annotation,
            create_image_annotation,
            create_sub_masks,
            create_sub_mask_annotation,
            create_annotation_format,
        )
        from shapely.geometry import MultiPolygon
    except ImportError as e:
        raise ImportError(
            "COCO JSON creation requires src.create_annotations and shapely. "
            "Either make sure these are available, or run with --no-create-coco-json."
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

    def images_annotations_info(mask_dir: Path):
        annotation_id = 0
        image_id = 0
        annotations = []
        images = []

        mask_paths = sorted(mask_dir.glob("*.png"))

        for mask_image in mask_paths:
            original_file_name = mask_image.stem + ".png"

            mask_image_open = Image.open(mask_image).convert("RGB")
            w, h = mask_image_open.size

            image = create_image_annotation(
                original_file_name,
                w,
                h,
                image_id,
            )
            images.append(image)

            sub_masks = create_sub_masks(mask_image_open, w, h)

            for color, sub_mask in sub_masks.items():
                category_id = category_colors[color]

                polygons, segmentations = create_sub_mask_annotation(sub_mask)

                if category_id in multipolygon_ids:
                    multi_poly = MultiPolygon(polygons)

                    annotation = create_annotation_format(
                        multi_poly,
                        segmentations,
                        image_id,
                        category_id,
                        annotation_id,
                    )

                    annotations.append(annotation)
                    annotation_id += 1

                else:
                    for i in range(len(polygons)):
                        segmentation = [
                            np.array(polygons[i].exterior.coords).ravel().tolist()
                        ]

                        annotation = create_annotation_format(
                            polygons[i],
                            segmentation,
                            image_id,
                            category_id,
                            annotation_id,
                        )

                        annotations.append(annotation)
                        annotation_id += 1

            image_id += 1

        return images, annotations, annotation_id

    for split in ["test", "valid", "train"]:
        mask_dir = dataset_root / f"png_{image_size}" / "mask" / split
        out_json = (
            dataset_root
            / f"png_{image_size}"
            / "image"
            / split
            / f"{split}_annot_coco_{image_size}.json"
        )

        coco_format = get_coco_json_format()
        coco_format["info"] = info()
        coco_format["categories"] = create_category_annotation(category_ids)
        coco_format["images"], coco_format["annotations"], annotation_cnt = (
            images_annotations_info(mask_dir)
        )

        with open(out_json, "w") as outfile:
            json.dump(coco_format, outfile)

        print(f"Created {annotation_cnt} annotations for images in folder: {mask_dir}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset",
        help="Dataset root. Default follows notebook: dataset",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=1536,
        help="Resize size for original images and masks. Default: 1536",
    )

    parser.add_argument(
        "--patch-sizes",
        nargs="+",
        type=int,
        default=[256, 512],
        help="Patch sizes to generate. Default: 256 512",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional random seed. "
            "If omitted, the split is not fixed, following the original notebook."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run checks and processing logic without writing files.",
    )

    parser.add_argument(
        "--create-coco-json",
        dest="create_coco_json",
        action="store_true",
        default=True,
        help="Create COCO JSON annotation files after patch generation. Default: True",
    )

    parser.add_argument(
        "--no-create-coco-json",
        dest="create_coco_json",
        action="store_false",
        help="Skip COCO JSON creation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    process_dataset(
        dataset_root=Path(args.dataset_root),
        image_size=args.image_size,
        patch_sizes=args.patch_sizes,
        seed=args.seed,
        dry_run=args.dry_run,
        create_coco_json=args.create_coco_json,
    )


if __name__ == "__main__":
    main()
