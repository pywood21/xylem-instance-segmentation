
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipelines/per_species/preprocess.py

Derived from:
- 00_bysp_img_msk_adjust_ps.ipynb
- 01_create_custom_coco_dataset_ps.ipynb

Purpose:
- For each species, read image/mask pairs from the original directories
- Resize images and masks
- Apply mask preprocessing
- Split data into train / valid / test
- Patchify and save outputs into png_<patch_size> directories

CLI arguments:
- --dataset-root
- --species
- --species-list
- --image-size
- --patch-sizes
- --seed
- --dry-run

Notes:
- This script preserves the per-species notebook logic as closely as practical.
- Viewer logic (notebook 02) is intentionally not included here.
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

def round_mask_levels(mask: np.ndarray, valid_levels=(0, 64, 128, 192, 255)) -> np.ndarray:
    """Round every pixel to the nearest valid gray level."""
    levels = np.array(valid_levels)
    idx = np.abs(mask[..., None] - levels).argmin(axis=-1)
    return levels[idx].astype(np.uint8)


def check_allowed_values(mask: np.ndarray, allowed_values=(0, 64, 128, 192, 255)) -> None:
    """Raise an error if unexpected pixel values are present."""
    unique = np.unique(mask)
    if not set(unique).issubset(set(allowed_values)):
        raise ValueError(f"Unexpected mask values found: {unique}")


def binarize_mask(mask: np.ndarray, threshold: int = 230) -> np.ndarray:
    """Convert mask to binary mask."""
    return np.where(mask > threshold, 255, 0).astype(np.uint8)


def thicken_binary(
    mask_binary: np.ndarray,
    dilate_iter: int = 2,
    erode_iter: int = 1,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply dilation then erosion."""
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


def list_pairs_for_species(species_root: Path):
    """
    Expected structure:
    <dataset-root>/<species>/original/image/*_img.png
    <dataset-root>/<species>/original/mask/*_msk.png
    """
    img_dir = species_root / "original" / "image"
    msk_dir = species_root / "original" / "mask"

    imgs = sorted(img_dir.glob("*_img.png"))
    msks = sorted(msk_dir.glob("*_msk.png"))

    img_map = {p.name.replace("_img.png", ""): p for p in imgs}
    msk_map = {p.name.replace("_msk.png", ""): p for p in msks}

    common_ids = sorted(set(img_map) & set(msk_map))

    if len(common_ids) == 0:
        raise RuntimeError(
            f"No paired image/mask files found under {species_root}.\n"
            f"Expected image files like *_img.png and mask files like *_msk.png."
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


def ensure_out_dirs(species_root: Path, patch_size: int):
    """
    Output structure:
    <species_root>/png_<patch_size>_thin/image/{train,valid,test}
    <species_root>/png_<patch_size>_thin/mask/{train,valid,test}
    """
    for split in ("train", "valid", "test"):
        (species_root / f"png_{patch_size}_thin" / "image" / split).mkdir(parents=True, exist_ok=True)
        (species_root / f"png_{patch_size}_thin" / "mask" / split).mkdir(parents=True, exist_ok=True)


def save_patch(
    species_root: Path,
    patch_size: int,
    split: str,
    name: str,
    img_patch: np.ndarray,
    msk_patch: np.ndarray,
):
    Image.fromarray(img_patch).save(
        species_root / f"png_{patch_size}_thin" / "image" / split / f"{name}.png"
    )
    Image.fromarray(msk_patch).save(
        species_root / f"png_{patch_size}_thin" / "mask" / split / f"{name}.png"
    )


# ---------------------------
# Main processing
# ---------------------------

def process_species(
    species: str,
    dataset_root: Path,
    image_size: int,
    patch_sizes: list[int],
    seed: int,
    dry_run: bool,
):
    species_root = dataset_root / species
    pairs = list_pairs_for_species(species_root)

    ids = [x[2] for x in pairs]
    split_map = split_ids(ids, seed=seed)

    for ps in patch_sizes:
        ensure_out_dirs(species_root, ps)

    img_map = {id_: img for img, _, id_ in pairs}
    msk_map = {id_: msk for _, msk, id_ in pairs}

    for split_name, id_list in split_map.items():
        for id_ in id_list:
            img_path = img_map[id_]
            msk_path = msk_map[id_]

            img, msk = load_and_resize(img_path, msk_path, image_size)

            # per-species notebook behaviour
            msk = round_mask_levels(msk)
            check_allowed_values(msk)

            binary = binarize_mask(msk, threshold=230)
            thick = thicken_binary(binary, dilate_iter=2, erode_iter=1, kernel_size=3)
            out_mask = invert_mask(thick)

            for ps in patch_sizes:
                img_patches = patchify(img, (ps, ps, 3), step=ps)
                msk_patches = patchify(out_mask, (ps, ps), step=ps)

                h, w = img_patches.shape[:2]
                count = 0
                for r in range(h):
                    for c in range(w):
                        name = f"{species}_{id_}_{count}"
                        if not dry_run:
                            save_patch(
                                species_root,
                                ps,
                                split_name,
                                name,
                                img_patches[r, c, 0],
                                msk_patches[r, c],
                            )
                        count += 1

    print(f"[OK] {species}: {len(pairs)} image-mask pairs processed.")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--species", type=str, default="all")
    parser.add_argument("--species-list", nargs="*", default=None)
    parser.add_argument("--image-size", type=int, default=1536)
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=[256, 512])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)

    if args.species != "all":
        targets = [args.species]
    else:
        if args.species_list:
            targets = args.species_list
        else:
            targets = [p.name for p in dataset_root.iterdir() if p.is_dir()]

    for sp in targets:
        process_species(
            species=sp,
            dataset_root=dataset_root,
            image_size=args.image_size,
            patch_sizes=args.patch_sizes,
            seed=args.seed,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
