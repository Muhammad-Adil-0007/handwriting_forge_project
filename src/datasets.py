from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision import transforms
import torch.nn.functional as F


# All line images (IAM + Emuru) will be resized to this height.
TARGET_HEIGHT: int = 64


def _geo_augment_pil(img: Image.Image) -> Image.Image:
    """
    Geometric augmentation for A1_geo_only:
    - small rotation
    - small scale
    - small translation
    """
    aug = transforms.RandomAffine(
        degrees=3,               # ±3°
        translate=(0.02, 0.02),  # up to 2% shift
        scale=(0.9, 1.1),        # 0.9x–1.1x
        shear=(-2, 2),           # small shear
        fill=255,                # white background for grayscale
    )
    return aug(img)


def _photo_augment_pil(img: Image.Image) -> Image.Image:
    """
    Photometric augmentation for A2_photo_both:
    - brightness / contrast jitter
    - occasional slight blur
    """
    aug = transforms.Compose([
        transforms.ColorJitter(
            brightness=0.2,   # up to ±20% brightness
            contrast=0.2,     # up to ±20% contrast
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
            p=0.3,
        ),
    ])
    return aug(img)


def _normalize_contrast_pil(img: Image.Image) -> Image.Image:
    """
    Deterministic contrast normalization for A3_domain_norm.

    - Convert to float32
    - Subtract mean, divide by std
    - Rescale to ~[0, 255] with fixed gain

    This is applied identically to ALL images (IAM + Emuru)
    and on ALL splits (train/val/test) when normalize=True.
    """
    arr = np.array(img).astype("float32")  # (H, W)

    m = arr.mean()
    s = arr.std()

    if s < 1e-6:
        # avoid exploding a near-constant image
        return img

    arr = (arr - m) / s
    # Gain & bias chosen heuristically; you can tweak
    arr = arr * 20.0 + 128.0
    arr = np.clip(arr, 0, 255)

    return Image.fromarray(arr.astype("uint8"), mode="L")


def load_line_image(
    img_path: Union[str, Path],
    target_height: int = TARGET_HEIGHT,
    augment_geom: bool = False,
    augment_photo: bool = False,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Load a line image (IAM or Emuru), convert to grayscale, optionally apply
    geometric and/or photometric augmentation, optionally apply deterministic
    contrast normalization, and resize so:

      - height = target_height
      - width is scaled proportionally

    Returns:
        Tensor of shape (1, H, W) with values in [0, 1].
    """
    img_path = Path(img_path)
    img = Image.open(img_path).convert("L")  # grayscale

    if augment_geom:
        img = _geo_augment_pil(img)

    if augment_photo:
        img = _photo_augment_pil(img)

    if normalize:
        img = _normalize_contrast_pil(img)

    w, h = img.size
    if h == 0:
        raise ValueError(f"Invalid image height 0 for {img_path}")

    scale = target_height / h
    new_w = max(1, int(round(w * scale)))

    img = img.resize((new_w, target_height), Image.BILINEAR)

    t = TF.to_tensor(img)  # (1, H, W), [0, 1]
    return t


class DomainClassificationDataset(Dataset):
    """
    Dataset for IAM vs Emuru domain classification.

    Expects a CSV with columns:
        - filepath      (relative to project root)
        - source        ("iam" or "emuru")
        - label         ("genuine" or "fake")  [string]
        - text          (line text, for debugging)
        - idx           (IAM index / line id)
        - hf_split      (e.g. "train", "test")
        - domain_label  (0 = IAM/genuine, 1 = Emuru/fake)
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        project_root: Union[str, Path],
        split: Optional[str] = None,
        target_height: int = TARGET_HEIGHT,
        augment_train: bool = False,
        photo_aug_train: bool = False,
        normalize_all: bool = False,
    ) -> None:
        """
        Args:
            csv_path: path to domain_classification_sentences.csv
            project_root: root of the project (for resolving filepaths)
            split: if not None, filter rows with hf_split == split
            target_height: height to which all images are resized
            augment_train: if True, apply geometric augmentation (A1)
                           ONLY when split == "train"
            photo_aug_train: if True, apply photometric augmentation (A2)
                             ONLY when split == "train"
            normalize_all: if True, apply contrast normalization (A3)
                           on ALL splits (train/val/test).
        """
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root)
        self.target_height = target_height
        self.augment_train = augment_train           # geometric (A1)
        self.photo_aug_train = photo_aug_train       # photometric (A2)
        self.normalize_all = normalize_all           # deterministic (A3)

        df = pd.read_csv(self.csv_path)

        if split is not None:
            df = df[df["hf_split"] == split].copy()
            self.split = split
        else:
            self.split = None

        self.df = df.reset_index(drop=True)

        print(
            f"DomainClassificationDataset: {len(self.df)} samples "
            f"(csv='{self.csv_path.name}', split='{self.split}', "
            f"augment_train={self.augment_train}, "
            f"photo_aug_train={self.photo_aug_train}, "
            f"normalize_all={self.normalize_all})"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        img_rel_path = Path(row["filepath"])
        img_path = self.project_root / img_rel_path

        is_train_split = (self.split == "train")

        do_geom = bool(self.augment_train and is_train_split)
        do_photo = bool(self.photo_aug_train and is_train_split)
        do_norm = bool(self.normalize_all)  # A3: always on if requested

        img_tensor = load_line_image(
            img_path,
            target_height=self.target_height,
            augment_geom=do_geom,
            augment_photo=do_photo,
            normalize=do_norm,
        )

        label = int(row["domain_label"])  # 0 or 1

        sample: Dict[str, Any] = {
            "image": img_tensor,            # (1, H, W)
            "label": label,                 # 0 = IAM/genuine, 1 = Emuru/fake
            "source": row["source"],
            "idx": int(row["idx"]),
            "text": row["text"],
            "hf_split": row["hf_split"],
        }
        return sample


def pad_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate_fn for variable-width line images.
    """
    images = [item["image"] for item in batch]  # list of (1, H, W)
    labels = [item["label"] for item in batch]

    heights = [img.shape[1] for img in images]
    widths = [img.shape[2] for img in images]

    H = heights[0]
    max_W = max(widths)

    padded_images = []
    for img in images:
        _, h, w = img.shape
        pad_right = max_W - w
        padded = F.pad(img, (0, pad_right, 0, 0), value=0.0)
        padded_images.append(padded)

    batch_images = torch.stack(padded_images, dim=0)  # (B, 1, H, max_W)
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
