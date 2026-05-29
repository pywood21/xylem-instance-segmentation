
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipelines/domain_adaptive/train.py

Derived from:
- 03_multisp_cell_instance_recognition_thin.ipynb

Purpose:
- Train a domain-adaptive Mask R-CNN model using pooled multi-species COCO-formatted annotations
- Add a gradient reversal layer (GRL) and domain classifier to encourage domain-invariant features
- Save the best checkpoint based on validation loss

Expected input structure by default:
    dataset/png_256/image/train/
    dataset/png_256/image/valid/
    dataset/png_256/image/test/

or, if --data-suffix _thin is used:
    dataset/png_256_thin/image/train/
    dataset/png_256_thin/image/valid/
    dataset/png_256_thin/image/test/

Expected COCO JSON files:
    train_annot_coco_256.json
    valid_annot_coco_256.json
    test_annot_coco_256.json

Example:
    python train.py \
        --dataset-root /path/to/dataset \
        --image-size 256 \
        --epochs 700 \
        --batch-size 8 \
        --checkpoint-file ./model/256_multidomain_700_epoch.tar
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        help="Patch size used for training. Default: 256",
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")

    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--no-save-model", dest="save_model", action="store_false")

    parser.add_argument("--load-model", action="store_true", default=False)

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help=(
            "Optional checkpoint file path. "
            "If omitted, uses ./model/{image_size}{data_suffix}_multidomain_{epochs}_epoch.tar"
        ),
    )

    parser.add_argument(
        "--domain-names",
        nargs="+",
        default=["alllit", "acapod", "burspi", "euclon", "exostr", "hakdac"],
        help=(
            "Domain/species names in the order used for domain IDs. "
            "Default follows the notebook: alllit acapod burspi euclon exostr hakdac."
        ),
    )

    parser.add_argument(
        "--grl-lambda",
        type=float,
        default=1.0,
        help="Gradient reversal coefficient. Default: 1.0",
    )

    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=1.0,
        help="Weight applied to domain loss before adding it to Mask R-CNN losses. Default: 1.0",
    )

    return parser.parse_args()


# ---------------------------
# Dataset
# ---------------------------

class CCDataset(Dataset):
    """
    COCO dataset reader for pooled multi-species domain-adaptive training.

    Expected split directory:
        dataset/png_<image_size><data_suffix>/image/{split}/

    Expected annotation file:
        {split}_annot_coco_<image_size>.json

    Domain labels are extracted from the patch filename.
    Example:
        0000_burspi2_01_img.png -> burspi -> domain id
    """

    def __init__(
        self,
        mode: str,
        image_size: int,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        domain_to_id: dict[str, int],
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

        if not os.path.exists(ann_path):
            raise FileNotFoundError(
                f"COCO annotation file not found: {ann_path}\n"
                "Run preprocess.py first, or check --dataset-root, --image-size, and --data-suffix."
            )

        self.coco = COCO(ann_path)
        self.cat_ids = self.coco.getCatIds()
        self.augmentation = augmentation
        self.image_ids = sorted(list(self.coco.imgs.keys()))
        self.domain_to_id = domain_to_id

    def __len__(self) -> int:
        return len(self.image_ids)

    def extract_domain_id(self, file_name: str) -> torch.Tensor:
        parts = file_name.split("_")
        if len(parts) < 2:
            raise ValueError(f"Unexpected filename format: {file_name}")

        species_with_number = parts[1]
        match = re.match(r"[a-zA-Z]+", species_with_number)

        if match is None:
            raise ValueError(f"Could not extract species/domain name from: {file_name}")

        species_name = match.group(0)

        if species_name not in self.domain_to_id:
            raise ValueError(
                f"Species/domain '{species_name}' is not in domain_to_id. "
                f"Available domains: {list(self.domain_to_id.keys())}"
            )

        return torch.tensor(self.domain_to_id[species_name], dtype=torch.long)

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
        file_name = img_info["file_name"]

        domain_id = self.extract_domain_id(file_name)

        image_path = os.path.join(self.dataset_path, file_name)
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        masks, cell_type = self.get_masks(image_id)

        if self.augmentation:
            augmented = self.augmentation(image=image, masks=masks)
            image = augmented["image"]
            masks = augmented["masks"]

        image = image.transpose(2, 0, 1) / 255.0
        boxes, masks, cell_type = self.get_boxes(masks, cell_type)

        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(cell_type, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image = torch.as_tensor(image, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "domain_id": domain_id,
        }

        return image, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

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
        contrast_limit=0.2,
        brightness_limit=0.3,
        p=0.5,
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
# Domain-adaptive Mask R-CNN
# ---------------------------

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    def __init__(self, in_features: int = 256, hidden: int = 128, num_domains: int = 6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, x):
        return self.classifier(x)


class CustomMaskRCNN(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        domain_classifier: nn.Module,
        grl_lambda: float = 1.0,
        domain_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.model = base_model
        self.domain_classifier = domain_classifier
        self.grl = GradientReversalLayer(lambda_=grl_lambda)
        self.domain_loss_weight = domain_loss_weight

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images_tensor = torch.stack(images)
        else:
            images_tensor = images

        backbone_features = self.model.backbone(images_tensor)
        feat_map = backbone_features["0"]

        pooled_feat = F.adaptive_avg_pool2d(feat_map, (1, 1)).view(
            feat_map.size(0), -1
        )
        reversed_feat = self.grl(pooled_feat)
        domain_pred = self.domain_classifier(reversed_feat)

        if self.training and targets is not None:
            domain_labels = torch.stack([t["domain_id"] for t in targets]).to(
                domain_pred.device
            )
            domain_loss = F.cross_entropy(domain_pred, domain_labels)
        else:
            domain_loss = torch.tensor(0.0, device=domain_pred.device)

        loss_dict = self.model(images, targets)

        if isinstance(loss_dict, dict):
            loss_dict["domain_loss"] = self.domain_loss_weight * domain_loss
            loss_dict["domain_loss_unweighted"] = domain_loss.detach()

        return loss_dict


def get_model(
    device: torch.device,
    num_domains: int,
    grl_lambda: float,
    domain_loss_weight: float,
):
    base_model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )

    in_features = base_model.roi_heads.box_predictor.cls_score.in_features
    base_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes=1 + 4,
    )

    domain_classifier = DomainClassifier(
        in_features=256,
        hidden=128,
        num_domains=num_domains,
    )

    model = CustomMaskRCNN(
        base_model=base_model,
        domain_classifier=domain_classifier,
        grl_lambda=grl_lambda,
        domain_loss_weight=domain_loss_weight,
    )
    model.to(device)

    return model


# ---------------------------
# Checkpoint utils
# ---------------------------

def save_checkpoint(state, filename: str) -> None:
    checkpoint_path = Path(filename)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print("=> Model improved, saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer, lr: float) -> None:
    print("=> Loading checkpoint")

    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


# ---------------------------
# Train / validation
# ---------------------------

def loss_sum_for_backprop(loss_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(
        loss for name, loss in loss_dict.items()
        if name != "domain_loss_unweighted"
    )


def train_one_epoch(loader, model, optimizer, device: torch.device) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_domain_loss = 0.0
    valid_batches = 0

    for batch in loader:
        if batch is None:
            continue

        images, targets = batch

        images = [image.to(device) for image in images]
        targets = [
            {k: v.to(device) for k, v in target.items()}
            for target in targets
        ]

        loss_dict = model(images, targets)
        losses = loss_sum_for_backprop(loss_dict)
        domain_loss = loss_dict.get(
            "domain_loss_unweighted",
            torch.tensor(0.0, device=device),
        )

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        running_domain_loss += domain_loss.item()
        valid_batches += 1

    if valid_batches == 0:
        raise RuntimeError("Training loader produced no valid batches.")

    avg_loss = running_loss / valid_batches
    avg_domain_loss = running_domain_loss / valid_batches

    return avg_loss, avg_domain_loss


best_vloss = np.inf


def validate(
    loader,
    model,
    device: torch.device,
    save_model: bool,
    checkpoint_path: str,
    optimizer,
) -> tuple[torch.Tensor, float]:
    """
    Validation uses model(images, targets) to compute validation loss.

    Detection models need train mode for loss computation.
    This follows the original notebook logic.
    """
    global best_vloss

    model.train()

    running_vloss = 0.0
    running_domain_loss = 0.0
    valid_batches = 0

    for batch in loader:
        if batch is None:
            continue

        images, targets = batch

        images = [image.to(device) for image in images]
        targets = [
            {k: v.to(device) for k, v in target.items()}
            for target in targets
        ]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = loss_sum_for_backprop(loss_dict)
        domain_loss = loss_dict.get(
            "domain_loss_unweighted",
            torch.tensor(0.0, device=device),
        )

        running_vloss += losses
        running_domain_loss += domain_loss.item()
        valid_batches += 1

    if valid_batches == 0:
        raise RuntimeError("Validation loader produced no valid batches.")

    avg_vloss = running_vloss / valid_batches
    avg_domain_loss = running_domain_loss / valid_batches

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=checkpoint_path)

    return avg_vloss, avg_domain_loss


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    domain_to_id = {
        domain_name: idx
        for idx, domain_name in enumerate(args.domain_names)
    }

    data_dir_name = f"png_{args.image_size}{args.data_suffix}"

    train_dir = f"{args.dataset_root}/{data_dir_name}/image/train"
    valid_dir = f"{args.dataset_root}/{data_dir_name}/image/valid"
    test_dir = f"{args.dataset_root}/{data_dir_name}/image/test"

    if args.checkpoint_file is None:
        checkpoint_path = (
            f"./model/{args.image_size}{args.data_suffix}_multidomain_"
            f"{args.epochs}_epoch.tar"
        )
    else:
        checkpoint_path = args.checkpoint_file

    print("=== Training configuration ===")
    print(f"device              : {device}")
    print(f"image_size          : {args.image_size}")
    print(f"data_dir_name       : {data_dir_name}")
    print(f"train_dir           : {train_dir}")
    print(f"valid_dir           : {valid_dir}")
    print(f"test_dir            : {test_dir}")
    print(f"checkpoint_path     : {checkpoint_path}")
    print(f"batch_size          : {args.batch_size}")
    print(f"epochs              : {args.epochs}")
    print(f"lr                  : {args.lr}")
    print(f"domains             : {domain_to_id}")
    print(f"grl_lambda          : {args.grl_lambda}")
    print(f"domain_loss_weight  : {args.domain_loss_weight}")
    print("================================")

    train_dataset = CCDataset(
        mode="train",
        image_size=args.image_size,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        domain_to_id=domain_to_id,
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
        image_size=args.image_size,
        train_dir=train_dir,
        valid_dir=valid_dir,
        test_dir=test_dir,
        domain_to_id=domain_to_id,
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

    model = get_model(
        device=device,
        num_domains=len(domain_to_id),
        grl_lambda=args.grl_lambda,
        domain_loss_weight=args.domain_loss_weight,
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.load_model:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"--load-model was specified, but checkpoint was not found: {checkpoint_path}"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            lr=args.lr,
        )

    for epoch in range(args.epochs):
        tloss, t_domain_loss = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            device=device,
        )

        vloss, v_domain_loss = validate(
            loader=valid_loader,
            model=model,
            device=device,
            save_model=args.save_model,
            checkpoint_path=checkpoint_path,
            optimizer=optimizer,
        )

        print(
            f"Epoch: {epoch}  "
            f"Total loss: {tloss:.3f}  "
            f"Train domain loss: {t_domain_loss:.3f}  "
            f"Avg Valid Loss: {vloss:.3f}  "
            f"Valid domain loss: {v_domain_loss:.3f}"
        )

    print("Training finished!")


if __name__ == "__main__":
    main()
