#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import gc
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import transforms
import torchvision.models as tv_models

from transformers import CLIPVisionModel, CLIPImageProcessor


# ----------------------------
# Models
# ----------------------------
class ConvNeXtDomainClassifier(nn.Module):
    def __init__(self, backbone_name="convnext_base", num_classes=2):
        super().__init__()
        if backbone_name == "convnext_base":
            self.backbone = tv_models.convnext_base(weights=None)
        elif backbone_name == "convnext_large":
            self.backbone = tv_models.convnext_large(weights=None)
        elif backbone_name == "convnext_small":
            self.backbone = tv_models.convnext_small(weights=None)
        elif backbone_name == "convnext_tiny":
            self.backbone = tv_models.convnext_tiny(weights=None)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {backbone_name}")

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ViTDomainClassifier(nn.Module):
    def __init__(self, backbone_name="vit_b_16", num_classes=2):
        super().__init__()
        if backbone_name == "vit_b_16":
            self.backbone = tv_models.vit_b_16(weights=None)
        elif backbone_name == "vit_l_16":
            self.backbone = tv_models.vit_l_16(weights=None)
        else:
            raise ValueError(f"Unsupported ViT backbone: {backbone_name}")

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ClipDomainClassifier(nn.Module):
    def __init__(self, clip_vision: CLIPVisionModel, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.clip_vision = clip_vision
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip_vision(pixel_values=pixel_values)
        feats = out.pooler_output
        return self.classifier(feats)


# ----------------------------
# Datasets (base preprocess only)
# ----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class BaseImageNetDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, project_root: Path):
        self.df = df_subset.reset_index(drop=True)
        self.project_root = project_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = self.project_root / row["filepath"]
        img = Image.open(img_path).convert("RGB")
        x = imagenet_transform(img)
        return {
            "images": x,
            "labels": int(row["domain_label"]),
        }


class BaseClipDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, project_root: Path, clip_processor: CLIPImageProcessor):
        self.df = df_subset.reset_index(drop=True)
        self.project_root = project_root
        self.clip_processor = clip_processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = self.project_root / row["filepath"]
        img = Image.open(img_path).convert("RGB")
        pv = self.clip_processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pv,
            "labels": int(row["domain_label"]),
        }


def imagenet_collate(batch):
    return {
        "images": torch.stack([b["images"] for b in batch], dim=0),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


def clip_collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


# ----------------------------
# GPU degradation (noise + blur)
# ----------------------------
def gaussian_kernel_2d(kernel_size: int, sigma: float, device):
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_blur_gpu(x: torch.Tensor, sigma: float):
    if sigma <= 0:
        return x
    k = int(max(3, 2 * math.ceil(3 * sigma) + 1))  # odd
    kernel = gaussian_kernel_2d(k, sigma, x.device)
    kernel = kernel.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)  # depthwise
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])


def degrade_batch_gpu(x: torch.Tensor, level: int, noise_step: float, blur_step: float):
    if level <= 0:
        return x
    x = x + torch.randn_like(x) * (level * noise_step)
    x = gaussian_blur_gpu(x, level * blur_step)
    return x


# ----------------------------
# Streaming sweep runner
# ----------------------------
@torch.no_grad()
def run_sweep_streaming(
    model_name: str,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    levels: list[int],
    noise_step: float,
    blur_step: float,
    use_amp: bool,
):
    rows = []
    model.eval()

    for level in levels:
        total = 0
        correct = 0

        stats = {
            0: {"n": 0, "sum_p": 0.0, "sumsq_p": 0.0, "sum_m": 0.0, "sumsq_m": 0.0},
            1: {"n": 0, "sum_p": 0.0, "sumsq_p": 0.0, "sum_m": 0.0, "sumsq_m": 0.0},
        }

        for batch in loader:
            if model_name == "clip":
                x = batch["pixel_values"].to(device, non_blocking=True)
            else:
                x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            x = degrade_batch_gpu(x, level, noise_step=noise_step, blur_step=blur_step)

            if device.type == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", enabled=True):
                    logits = model(x)
            else:
                logits = model(x)

            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            bs = y.numel()
            total += bs
            correct += (pred == y).sum().item()

            p_fake = probs[:, 1].detach().float()
            margin = (logits[:, 1] - logits[:, 0]).detach().float()

            for cls in (0, 1):
                mask = (y == cls)
                if mask.any():
                    p = p_fake[mask]
                    m = margin[mask]
                    n = int(mask.sum().item())

                    stats[cls]["n"] += n
                    stats[cls]["sum_p"] += float(p.sum().item())
                    stats[cls]["sumsq_p"] += float((p * p).sum().item())
                    stats[cls]["sum_m"] += float(m.sum().item())
                    stats[cls]["sumsq_m"] += float((m * m).sum().item())

        acc = correct / total if total > 0 else 0.0

        for cls in (0, 1):
            n = stats[cls]["n"]
            if n > 0:
                mean_p = stats[cls]["sum_p"] / n
                var_p = max(0.0, stats[cls]["sumsq_p"] / n - mean_p**2)
                std_p = var_p**0.5

                mean_m = stats[cls]["sum_m"] / n
                var_m = max(0.0, stats[cls]["sumsq_m"] / n - mean_m**2)
                std_m = var_m**0.5
            else:
                mean_p = std_p = mean_m = std_m = float("nan")

            rows.append({
                "model": model_name,
                "level": level,
                "true_label": cls,
                "mean_prob_fake": mean_p,
                "std_prob_fake": std_p,
                "mean_margin": mean_m,
                "std_margin": std_m,
                "accuracy_overall": float(acc),
                "n": int(n),
            })

        print(f"[{model_name}] level={level:02d} acc={acc*100:.2f}%")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def parse_levels(s: str) -> list[int]:
    # e.g. "0-10" or "0,1,2,3"
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--csv-path", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="analysis/noise_sweep_scores_gpu.csv")

    ap.add_argument("--levels", type=str, default="0-10")
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--noise-step", type=float, default=0.005)
    ap.add_argument("--blur-step", type=float, default=0.10)

    ap.add_argument("--run-clip", action="store_true")
    ap.add_argument("--run-convnext", action="store_true")
    ap.add_argument("--run-vit", action="store_true")

    ap.add_argument("--clip-ckpt", type=str, default="")
    ap.add_argument("--convnext-ckpt", type=str, default="")
    ap.add_argument("--vit-ckpt", type=str, default="")

    ap.add_argument("--convnext-backbone", type=str, default="convnext_base")
    ap.add_argument("--vit-backbone", type=str, default="vit_b_16")

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()
    out_csv = (project_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    levels = parse_levels(args.levels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CSV:", csv_path)
    print("Levels:", levels)

    df = pd.read_csv(csv_path)
    df_test = df[df["hf_split"] == "test"].copy()

    df0 = df_test[df_test["domain_label"] == 0]
    df1 = df_test[df_test["domain_label"] == 1]
    n0 = min(len(df0), args.max_per_class)
    n1 = min(len(df1), args.max_per_class)

    df0s = df0.sample(n=n0, random_state=42) if len(df0) > n0 else df0
    df1s = df1.sample(n=n1, random_state=42) if len(df1) > n1 else df1
    df_sub = pd.concat([df0s, df1s], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("Subset size:", df_sub.shape)

    parts = []

    # CLIP
    if args.run_clip:
        ckpt = Path(args.clip_ckpt) if args.clip_ckpt else None
        if ckpt is None:
            raise ValueError("Please provide --clip-ckpt when using --run-clip")
        if not ckpt.is_absolute():
            ckpt = (project_root / ckpt).resolve()
        print("CLIP ckpt:", ckpt)

        clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True,)
        clip_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True,).to(device)
        for p in clip_vision.parameters():
            p.requires_grad = False

        clip_model = ClipDomainClassifier(clip_vision, clip_vision.config.hidden_size).to(device)
        clip_model.load_state_dict(torch.load(ckpt, map_location=device))
        clip_model.eval()

        ds = BaseClipDataset(df_sub, project_root, clip_processor)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=clip_collate,
        )

        # Often safest to run CLIP without AMP
        df_clip = run_sweep_streaming(
            "clip", clip_model, loader, device, levels,
            noise_step=args.noise_step, blur_step=args.blur_step,
            use_amp=False,
        )
        parts.append(df_clip)

    # ConvNeXt
    if args.run_convnext:
        ckpt = Path(args.convnext_ckpt) if args.convnext_ckpt else None
        if ckpt is None:
            raise ValueError("Please provide --convnext-ckpt when using --run-convnext")
        if not ckpt.is_absolute():
            ckpt = (project_root / ckpt).resolve()
        print("ConvNeXt ckpt:", ckpt)

        model = ConvNeXtDomainClassifier(args.convnext_backbone).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        ds = BaseImageNetDataset(df_sub, project_root)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=imagenet_collate,
        )

        df_conv = run_sweep_streaming(
            "convnext", model, loader, device, levels,
            noise_step=args.noise_step, blur_step=args.blur_step,
            use_amp=True,
        )
        parts.append(df_conv)

    # ViT
    if args.run_vit:
        ckpt = Path(args.vit_ckpt) if args.vit_ckpt else None
        if ckpt is None:
            raise ValueError("Please provide --vit-ckpt when using --run-vit")
        if not ckpt.is_absolute():
            ckpt = (project_root / ckpt).resolve()
        print("ViT ckpt:", ckpt)

        model = ViTDomainClassifier(args.vit_backbone).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        ds = BaseImageNetDataset(df_sub, project_root)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=imagenet_collate,
        )

        df_vit = run_sweep_streaming(
            "vit", model, loader, device, levels,
            noise_step=args.noise_step, blur_step=args.blur_step,
            use_amp=True,
        )
        parts.append(df_vit)

    if not parts:
        raise RuntimeError("Nothing to run. Use --run-clip / --run-convnext / --run-vit.")

    out = pd.concat(parts, axis=0).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
