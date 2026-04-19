
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipelines/per_species/train.py

Derived from:
- 03_cell_instance_recognition_ps.ipynb

Purpose:
- Train a per-species Mask R-CNN model using COCO-formatted annotations
- Save the best checkpoint based on validation loss

CLI arguments:
- --species
- --image-size
- --dataset-root
- --gpu-id
- --batch-size
- --epochs
- --lr
- --weight-decay
- --num-workers
- --load-model
- --checkpoint-file

Notes:
- This script intentionally preserves the notebook logic as closely as possible.
- Validation uses the same model loss path as training (as in the notebook).
- Plotting, GIF generation, and test-time visualization were intentionally left in the notebook.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
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

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=700)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=2)

    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    p.add_argument("--save-model", action="store_true", default=True)
    p.add_argument("--no-save-model", dest="save_model", action="store_false")

    p.add_argument("--load-model", action="store_true", default=False)
    p.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Optional explicit checkpoint file path. If omitted, uses the notebook-style default name.",
    )

    return p.parse_args()


# ---------------------------
# Dataset
# ---------------------------

class CCDataset(Dataset):
    """
    COCO dataset reader for the per-species training data.

    Expected structure:
    dataset/<species>/png_<image_size>_thin/image/train/
    dataset/<species>/png_<image_size>_thin/image/valid/
    dataset/<species>/png_<image_size>_thin/image/test/

    And inside each split directory:
    - images (*.png)
    - COCO annotation json:
      train_annot_coco_<image_size>.json
      valid_annot_coco_<image_size>.json
      test_annot_coco_<image_size>.json
    """

    def __init__(
        self,
        mode: str,
        image_size: int,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        augmentation=None,
    ) -> None:
        if mode == "train":
            self.dataset_path = train_dir
            ann_path = os.path.join(train_dir, f"train_annot_coco_{image_size}.json")
        elif mode == "valid":
            self.dataset_path = valid_dir
            ann_path = os.path.join(valid_dir, f"valid_annot_coco_{image_size}.json")
        elif mode == "test":
            self.dataset_path = test_dir
            ann_path = os.path.join(test_dir, f"test_annot_coco_{image_size}.json")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.coco = COCO(ann_path)
        self.cat_ids = self.coco.getCatIds()
        self.augmentation = augmentation

        # NOTE:
        # The notebook uses index directly in loadImgs/getAnnIds.
        # To preserve behavior while being slightly safer, we keep a stable list of image ids.
        self.image_ids = sorted(list(self.coco.imgs.keys()))

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_masks(self, image_id: int):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ann_ids)

        masks = []
        cell_type = []
        for ann in anns:
            cell_type.append(ann["category_id"])
            mask = self.coco.annToMask(ann)
            masks.append(mask)

        return masks, cell_type

    def get_boxes(self, masks, cell_type):
        valid_masks = []
        valid_types = []
        boxes = []

        for i in range(len(masks)):
            x, y, w, h = cv2.boundingRect(masks[i].astype(np.uint8))
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                valid_masks.append(masks[i])
                valid_types.append(cell_type[i])

        return np.array(boxes), np.array(valid_masks), np.array(valid_types)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        img_info = self.coco.loadImgs([image_id])[0]

        image = cv2.imread(os.path.join(self.dataset_path, img_info["file_name"]))
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {os.path.join(self.dataset_path, img_info['file_name'])}"
            )

        masks, cell_type = self.get_masks(image_id)

        if self.augmentation:
            augmented = self.augmentation(image=image, masks=masks)
            image, masks = augmented["image"], augmented["masks"]

        image = image.transpose(2, 0, 1) / 255.0
        boxes, masks, cell_type = self.get_boxes(masks, cell_type)

        if len(boxes) == 0:
            # preserve notebook intent, but avoid hard failure in collate
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(cell_type, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image = torch.as_tensor(image, dtype=torch.float32)

        data = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        return image, data


def collate_fn(batch):
    """
    Notebook version stacked images and left targets as a list.
    Here we also filter out None entries to avoid crashing if an image has no valid boxes.
    """
    batch = [b for b in batch if b is not None]

    images = []
    targets = []
    for b in batch:
        images.append(b[0])
        targets.append(b[1])

    images = torch.stack(images, dim=0)
    return images, targets


# ---------------------------
# Augmentation
# ---------------------------

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(
        contrast_limit=0.2, brightness_limit=0.3, p=0.5
    ),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
        A.Equalize(mode="cv", p=0.8),
    ], p=1.0),
    A.OneOf([
        A.ImageCompression(p=0.8),
        A.RandomGamma(p=0.8),
        A.Blur(p=0.8),
        A.Equalize(mode="cv", p=0.8),
    ], p=1.0),
])


# ---------------------------
# Model
# ---------------------------

def get_model(device: torch.device):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=1 + 4
    )
    model.to(device)
    return model


# ---------------------------
# Checkpoint utils
# ---------------------------

def save_checkpoint(state, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    print("=> Model improved, saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr: float) -> None:
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # preserve notebook behavior
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------
# Train / validation
# ---------------------------

def train_one_epoch(loader, model, optimizer, device: torch.device) -> float:
    model.train()
    last_loss = 0.0

    for images, targets in loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        last_loss = losses.item()

    return last_loss


best_vloss = np.inf

def validate(loader, model, optimizer, device: torch.device, epoch: int,
             save_model: bool, checkpoint_path: str) -> torch.Tensor:
    """
    Preserves the notebook logic:
    - validation still uses model(images, targets) to compute losses
    - best checkpoint is determined by avg validation loss
    """
    global best_vloss

    # Detection models typically need train mode for loss computation.
    model.train()

    running_vloss = 0.0
    last_batch_idx = -1

    for batch_idx, (images, targets) in enumerate(loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        running_vloss += losses
        last_batch_idx = batch_idx

    if last_batch_idx < 0:
        raise RuntimeError("Validation loader produced no batches.")

    avg_vloss = running_vloss / (last_batch_idx + 1)

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=checkpoint_path)

    return avg_vloss


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    species = args.species
    image_size = args.image_size
    dataset_root = args.dataset_root

    train_dir = f"{dataset_root}/{species}/png_{image_size}_thin/image/train"
    valid_dir = f"{dataset_root}/{species}/png_{image_size}_thin/image/valid"
    test_dir = f"{dataset_root}/{species}/png_{image_size}_thin/image/test"

    if args.checkpoint_file is None:
        checkpoint_path = f"./checkpoint/{image_size}_thin_{species}_{args.epochs}_epoch.tar"
    else:
        checkpoint_path = args.checkpoint_file

    print("=== Training configuration ===")
    print(f"device          : {device}")
    print(f"species         : {species}")
    print(f"image_size      : {image_size}")
    print(f"train_dir       : {train_dir}")
    print(f"valid_dir       : {valid_dir}")
    print(f"checkpoint_path : {checkpoint_path}")
    print(f"batch_size      : {args.batch_size}")
    print(f"epochs          : {args.epochs}")
    print(f"lr              : {args.lr}")
    print("==============================")

    train_dataset = CCDataset(
        mode="train",
        image_size=image_size,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        augmentation=transform,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )

    valid_dataset = CCDataset(
        mode="valid",
        image_size=image_size,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        augmentation=None,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )

    model = get_model(device=device)
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.load_model and os.path.exists(checkpoint_path):
        print("Loading checkpoint")
        load_checkpoint(
            torch.load(checkpoint_path, map_location=device),
            model,
            optimizer,
            args.lr,
        )

    total_loss = []
    valid_loss = []

    for epoch in range(args.epochs):
        tloss = train_one_epoch(train_loader, model, optimizer, device)
        vloss = validate(
            valid_loader,
            model,
            optimizer,
            device,
            epoch,
            save_model=args.save_model,
            checkpoint_path=checkpoint_path,
        )

        total_loss.append(tloss)
        valid_loss.append(vloss.cpu().item())

        comment = (
            f"Epoch: {epoch}  "
            f"Total loss: {tloss:.3f}  "
            f"Avg Valid Loss: {vloss:.3f}"
        )
        print(comment)

    print("Training finished!")


if __name__ == "__main__":
    main()
