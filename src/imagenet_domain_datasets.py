# src/imagenet_domain_datasets.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageNetDomainDataset(Dataset):
    """
    Domain classification dataset (IAM vs Emuru) for ImageNet-style backbones
    such as ConvNeXt and ViT.

    Expects CSVs like:
      - filepath      (relative to project root)
      - source        ("iam" or "emuru")
      - label         ("genuine" or "fake")
      - text          (line text)
      - idx           (IAM index / line id)
      - hf_split      ("train", "validation", "test")
      - domain_label  (0 = IAM/genuine, 1 = Emuru/fake)

    Options:
      - augment_geo    (A1: small geometric jitter, train only)
      - augment_photo  (A2: photometric jitter + blur, train only)
      - normalize_all  (A3: contrast normalization for ALL splits)
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        project_root: Union[str, Path],
        split: Optional[str] = None,
        image_size: int = 224,
        augment_geo: bool = False,
        augment_photo: bool = False,
        normalize_all: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)

        df = pd.read_csv(self.csv_path)
        if split is not None:
            df = df[df["hf_split"] == split].copy()
            self.split = split
        else:
            self.split = None

        self.df = df.reset_index(drop=True)

        # flags
        self.augment_geo = augment_geo and (self.split == "train")
        self.augment_photo = augment_photo and (self.split == "train")
        self.normalize_all = normalize_all

        # --- transforms ---

        # A1: small geometric jitter (before resize/normalize)
        self.geo_transform = transforms.RandomAffine(
            degrees=3.0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=255,  # white background
        )

        # A2: photometric jitter + blur (before resize/normalize)
        self.photo_transform = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ),
            transforms.GaussianBlur(
                kernel_size=3,
                sigma=(0.1, 1.0),
            ),
        ])

        # Base ImageNet preprocessing
        self.base_transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"ImageNetDomainDataset: {len(self.df)} samples "
            f"(csv='{self.csv_path.name}', split='{self.split}', "
            f"augment_geo={self.augment_geo}, augment_photo={self.augment_photo}, "
            f"normalize_all={self.normalize_all})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_rgb(self, img_path: Path) -> Image.Image:
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _normalize_contrast(self, img_rgb: Image.Image) -> Image.Image:
        """
        A3-style contrast normalization: per-image min/max stretch to [0, 255].
        """
        img_gray = img_rgb.convert("L")
        arr = np.array(img_gray).astype("float32")

        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin) * 255.0
        else:
            arr[:] = 255.0

        arr = arr.clip(0, 255).astype("uint8")
        img_norm_gray = Image.fromarray(arr, mode="L")
        return img_norm_gray.convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_rel_path = Path(row["filepath"])
        img_path = self.project_root / img_rel_path

        img = self._load_image_rgb(img_path)

        # A3: normalize both IAM & Emuru for all splits if enabled
        if self.normalize_all:
            img = self._normalize_contrast(img)

        # A1: geometric aug on train only
        if self.augment_geo:
            img = self.geo_transform(img)

        # A2: photometric aug on train only
        if self.augment_photo:
            img = self.photo_transform(img)

        img_t = self.base_transform(img)  # (3, H, W), ImageNet normalized

        label = int(row["domain_label"])

        return {
            "image": img_t,
            "label": label,
            "source": row["source"],
            "idx": int(row["idx"]),
            "text": row["text"],
            "hf_split": row["hf_split"],
        }


def imagenet_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    batch_images = torch.stack(images, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)

    sources = [item["source"] for item in batch]
    idxs = [item["idx"] for item in batch]
    texts = [item["text"] for item in batch]
    hf_splits = [item["hf_split"] for item in batch]

    return {
        "images": batch_images,
        "labels": batch_labels,
        "sources": sources,
        "idxs": idxs,
        "texts": texts,
        "hf_splits": hf_splits,
    }
