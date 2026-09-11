from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ClipDomainDataset(Dataset):
    """
    Dataset for IAM vs Emuru domain classification using CLIP vision features.

    Expects the same CSV format as DomainClassificationDataset, e.g.:

        - filepath      (relative to project root)
        - source        ("iam" or "emuru")
        - label         ("genuine" or "fake")
        - text          (line text)
        - idx           (IAM index / line id)
        - hf_split      ("train", "validation", "test")
        - domain_label  (0 = IAM/genuine, 1 = Emuru/fake)

    Options:
        - augment_geo      (A1-style geometric aug, train only)
        - augment_photo    (A2-style photometric aug, train only)
        - normalize_all    (A3-style contrast normalization, ALL splits)
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        project_root: Union[str, Path],
        clip_processor,
        split: Optional[str] = None,
        augment: bool = False,          # geometric augmentation (A1)
        photo_augment: bool = False,    # photometric augmentation (A2)
        normalize_all: bool = False,    # contrast normalization for ALL splits (A3)
    ) -> None:
        """
        Args:
            csv_path: path to domain_classification_sentences.csv (or A4 .csv)
            project_root: root of the project (for resolving filepaths)
            clip_processor: a CLIP image processor
            split: if not None, filter rows with hf_split == split
                   (e.g. "train", "validation", "test").
            augment: if True and split == "train", apply small geometric aug
                     (A1-style) before CLIP preprocessing.
            photo_augment: if True and split == "train", apply photometric aug
                     (A2-style) before CLIP preprocessing.
            normalize_all: if True, apply per-image contrast normalization
                     (A3-style) for ALL splits (train/val/test).
        """
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.clip_processor = clip_processor

        if self.clip_processor is None:
            raise ValueError("clip_processor must be provided to ClipDomainDataset")

        df = pd.read_csv(self.csv_path)

        if split is not None:
            df = df[df["hf_split"] == split].copy()
            self.split = split
        else:
            self.split = None

        self.df = df.reset_index(drop=True)

        # Only augment train split; normalization can apply to all splits.
        self.augment_geo = augment and (self.split == "train")
        self.augment_photo = photo_augment and (self.split == "train")
        self.normalize_all = normalize_all

        # Geometric aug (A1): small affine jitter
        self.geo_transform = transforms.RandomAffine(
            degrees=3.0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=255,          # white background
        )

        # Photometric aug (A2): brightness/contrast jitter + blur
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

        print(
            f"ClipDomainDataset: {len(self.df)} samples "
            f"(csv='{self.csv_path.name}', split='{self.split}', "
            f"augment_geo={self.augment_geo}, augment_photo={self.augment_photo}, "
            f"normalize_all={self.normalize_all})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image_rgb(self, img_path: Path) -> Image.Image:
        """
        Load an image from disk and convert to RGB for CLIP.
        """
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _normalize_contrast(self, img_rgb: Image.Image) -> Image.Image:
        """
        A3-style contrast normalization: stretch grayscale intensities
        to [0, 255] per image, then convert back to RGB.
        """
        # Work in grayscale
        img_gray = img_rgb.convert("L")
        arr = np.array(img_gray).astype("float32")

        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin) * 255.0
        else:
            # flat image -> leave as white
            arr[:] = 255.0

        arr = arr.clip(0, 255).astype("uint8")
        img_norm_gray = Image.fromarray(arr, mode="L")
        img_norm_rgb = img_norm_gray.convert("RGB")
        return img_norm_rgb

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_rel_path = Path(row["filepath"])
        img_path = self.project_root / img_rel_path

        # Load and convert to RGB
        img = self._load_image_rgb(img_path)

        # A3: contrast normalization for all splits if enabled
        if self.normalize_all:
            img = self._normalize_contrast(img)

        # Optional geometric augmentation (A1, train only)
        if self.augment_geo:
            img = self.geo_transform(img)

        # Optional photometric augmentation (A2, train only)
        if self.augment_photo:
            img = self.photo_transform(img)

        # Use CLIP's processor to get pixel_values: (1, 3, H, W)
        inputs = self.clip_processor(
            images=img,
            return_tensors="pt",
        )
        pixel_values = inputs["pixel_values"].squeeze(0)  # (3, H, W)

        label = int(row["domain_label"])  # 0 = IAM/genuine, 1 = Emuru/fake

        sample: Dict[str, Any] = {
            "pixel_values": pixel_values,
            "label": label,
            "source": row["source"],
            "idx": int(row["idx"]),
            "text": row["text"],
            "hf_split": row["hf_split"],
        }
        return sample


def clip_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    pixel_values_list = [item["pixel_values"] for item in batch]
    labels = [item["label"] for item in batch]

    batch_pixel_values = torch.stack(pixel_values_list, dim=0)  # (B, 3, H, W)
    batch_labels = torch.tensor(labels, dtype=torch.long)

    sources = [item["source"] for item in batch]
    idxs = [item["idx"] for item in batch]
    texts = [item["text"] for item in batch]
    hf_splits = [item["hf_split"] for item in batch]

    return {
        "pixel_values": batch_pixel_values,
        "labels": batch_labels,
        "sources": sources,
        "idxs": idxs,
        "texts": texts,
        "hf_splits": hf_splits,
    }
