
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipelines/mixed_species/preprocess.py

Derived from:
- 00_multisp_img_msk_adjust_ms.ipynb
- 01_create_custom_coco_dataset_ms.ipynb

Purpose:
- Read pooled multi-species image/mask pairs
- Resize images and masks
- Apply mask preprocessing
- Split data into train / valid / test
- Patchify and save outputs into png_<patch_size> directories

CLI arguments:
- --dataset-root
- --input-image-dir
- --input-mask-dir
- --image-size
- --patch-sizes
- --seed
- --dry-run

Notes:
- This script preserves the mixed-species notebook logic as closely as practical.
- Viewer logic is intentionally not included here.
- This version assumes:
    image: <ID>_img.png
    mask : <ID>_msk.png
  and matches them by the shared <ID>.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from patchify import patchify


# ---------------------------
# Mask processing
# ---------------------------

def check_allowed_values(mask: np.ndarray, allowed_values=(0, 64, 128, 192, 255)) -> None:
    unique = np.unique(mask)
    if not set(unique).issubset(set(allowed_values)):
        raise ValueError(f"Unexpected mask values found: {unique}")


def binarize_mask(mask: np.ndarray, threshold: int = 230) -> np.ndarray:
    return np.where(mask > threshold, 255, 0).astype(np.uint8)


def thicken_binary(
    mask_binary: np.ndarray,
    dilate_iter: int = 2,
    erode_iter: int = 2,
    kernel_size: int = 3,
) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    thick = cv2.dilate(mask_binary, kernel, iterations=dilate_iter)
    thick = cv2.erode(thick, kernel, iterations=erode_iter)
    return thick


def invert_mask(mask: np.ndarray) -> np.ndarray:
    return 255 - mask


# ---------------------------
# Data IO helpers
# ---------------------------

def load_and_resize(img_path: Path, msk_path: Path, image_size: int):
    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
    msk = Image.open(msk_path).convert("L").resize((image_size, image_size))
    return np.array(img), np.array(msk)


def list_pairs(image_dir: Path, mask_dir: Path):
    """
    Mixed-species raw pooled data.

    Expected structure by default:
    <dataset-root>/original/image/*_img.png
    <dataset-root>/original/mask/*_msk.png

    Matching rule:
    image: <ID>_img.png
    mask : <ID>_msk.png
    """
    imgs = sorted(image_dir.glob("*_img.png"))
    msks = sorted(mask_dir.glob("*_msk.png"))

    img_map = {p.name.replace("_img.png", ""): p for p in imgs}
    msk_map = {p.name.replace("_msk.png", ""): p for p in msks}

    common_ids = sorted(set(img_map) & set(msk_map))

    if len(common_ids) == 0:
        raise RuntimeError(
            f"No paired image/mask files found.\n"
            f"image_dir={image_dir}\n"
            f"mask_dir={mask_dir}\n"
            f"Expected image files like *_img.png and mask files like *_msk.png."
        )

    # Warn if there are unpaired files
    only_img = sorted(set(img_map) - set(msk_map))
    only_msk = sorted(set(msk_map) - set(img_map))
    if only_img or only_msk:
        print(
            f"[WARN] Unpaired files found: "
            f"only_img={len(only_img)}, only_msk={len(only_msk)}"
        )

    return [(img_map[i], msk_map[i], i) for i in common_ids]


def split_ids(ids, seed=0, train_ratio=0.6, valid_ratio=0.24, test_ratio=0.16):
    random.seed(seed)
    ids = ids.copy()
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_ids = ids[:n_train]
    valid_ids = ids[n_train:n_train + n_valid]
    test_ids = ids[n_train + n_valid:]

    return {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }


def ensure_out_dirs(dataset_root: Path, patch_size: int):
    """
    Output structure:
    <dataset-root>/png_<patch_size>/image/{train,valid,test}
    <dataset-root>/png_<patch_size>/mask/{train,valid,test}
    """
    for split in ("train", "valid", "test"):
        (dataset_root / f"png_{patch_size}" / "image" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / f"png_{patch_size}" / "mask" / split).mkdir(parents=True, exist_ok=True)


def save_patch(
    dataset_root: Path,
    patch_size: int,
    split: str,
    name: str,
    img_patch: np.ndarray,
    msk_patch: np.ndarray,
):
    Image.fromarray(img_patch).save(
        dataset_root / f"png_{patch_size}" / "image" / split / f"{name}.png"
    )
    Image.fromarray(msk_patch).save(
        dataset_root / f"png_{patch_size}" / "mask" / split / f"{name}.png"
    )


# ---------------------------
# Main processing
# ---------------------------

def process_dataset(
    dataset_root: Path,
    input_image_dir: Path,
    input_mask_dir: Path,
    image_size: int,
    patch_sizes: list[int],
    seed: int,
    dry_run: bool,
):
    pairs = list_pairs(input_image_dir, input_mask_dir)

    ids = [x[2] for x in pairs]
    split_map = split_ids(ids, seed=seed)

    for ps in patch_sizes:
        ensure_out_dirs(dataset_root, ps)

    img_map = {id_: img for img, _, id_ in pairs}
    msk_map = {id_: msk for _, msk, id_ in pairs}

    for split_name, id_list in split_map.items():
        for id_ in id_list:
            img_path = img_map[id_]
            msk_path = msk_map[id_]

            img, msk = load_and_resize(img_path, msk_path, image_size)

            check_allowed_values(msk)
            binary = binarize_mask(msk, threshold=230)
            thick = thicken_binary(binary, dilate_iter=2, erode_iter=2, kernel_size=3)
            out_mask = invert_mask(thick)

            for ps in patch_sizes:
                img_patches = patchify(img, (ps, ps, 3), step=ps)
                msk_patches = patchify(out_mask, (ps, ps), step=ps)

                h, w = img_patches.shape[:2]
                count = 0
                for r in range(h):
                    for c in range(w):
                        name = f"{id_}_{count}"
                        if not dry_run:
                            save_patch(
                                dataset_root,
                                ps,
                                split_name,
                                name,
                                img_patches[r, c, 0],
                                msk_patches[r, c],
                            )
                        count += 1

    print(f"[OK] {len(pairs)} image-mask pairs processed.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument(
        "--input-image-dir",
        type=str,
        default=None,
        help="Optional explicit image directory. If omitted, uses <dataset-root>/original/image",
    )
    parser.add_argument(
        "--input-mask-dir",
        type=str,
        default=None,
        help="Optional explicit mask directory. If omitted, uses <dataset-root>/original/mask",
    )
    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if args.input_image_dir is None:
        input_image_dir = dataset_root / "original" / "image"
    else:
        input_image_dir = Path(args.input_image_dir)

    if args.input_mask_dir is None:
        input_mask_dir = dataset_root / "original" / "mask"
    else:
        input_mask_dir = Path(args.input_mask_dir)

    process_dataset(
        dataset_root=dataset_root,
        input_image_dir=input_image_dir,
        input_mask_dir=input_mask_dir,
        image_size=args.image_size,
        patch_sizes=args.patch_sizes,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
