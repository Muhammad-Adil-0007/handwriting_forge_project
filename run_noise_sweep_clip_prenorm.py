#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
from pathlib import Path
import math
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFilter
from transformers import CLIPVisionModel, CLIPImageProcessor


# ----------------------------
# Model wrapper (same as your training)
# ----------------------------
class ClipDomainClassifier(nn.Module):
    def __init__(self, clip_vision: CLIPVisionModel, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.clip_vision = clip_vision
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.clip_vision(pixel_values=pixel_values)
        feats = outputs.pooler_output
        return self.classifier(feats)


# ----------------------------
# Corruption in image space (before normalization)
# ----------------------------
def corrupt_pil_pre_norm(img: Image.Image, level: int, rng: np.random.Generator,
                         noise_step: float, blur_step: float) -> Image.Image:
    """
    Apply corruption in image space:
      - Additive Gaussian noise in [0,1] space
      - Gaussian blur in pixel space
    """
    if level <= 0:
        return img

    # 1) Gaussian noise on float image
    arr = np.asarray(img).astype(np.float32) / 255.0
    sigma = level * noise_step
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    img = Image.fromarray((arr * 255).astype(np.uint8))

    # 2) Blur
    blur_radius = level * blur_step
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))

    return img


# ----------------------------
# Dataset: loads PIL, corrupts PIL, then applies CLIP processor
# ----------------------------
class ClipPreNormSweepDataset(Dataset):
    def __init__(self, df_subset: pd.DataFrame, project_root: Path,
                 clip_processor: CLIPImageProcessor, level: int,
                 noise_step: float, blur_step: float, seed: int):
        self.df = df_subset.reset_index(drop=True)
        self.project_root = project_root
        self.clip_processor = clip_processor
        self.level = level
        self.noise_step = noise_step
        self.blur_step = blur_step
        self.seed = seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = self.project_root / row["filepath"]
        img = Image.open(img_path).convert("RGB")

        # deterministic per (level, sample index) so the curve is reproducible
        rng = np.random.default_rng(self.seed + 1000 * self.level + i)
        img = corrupt_pil_pre_norm(img, self.level, rng, self.noise_step, self.blur_step)

        pv = self.clip_processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pv,
            "labels": int(row["domain_label"]),
        }


def clip_collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


# ----------------------------
# Streaming sweep
# ----------------------------
@torch.no_grad()
def run_sweep_streaming(model: nn.Module, device: torch.device, loader: DataLoader, level: int):
    """
    Returns:
      acc, mean/std P(fake) and margin per true class.
    """
    total = 0
    correct = 0

    # per-class aggregates
    stats = {
        0: {"n": 0, "sum_p": 0.0, "sumsq_p": 0.0, "sum_m": 0.0, "sumsq_m": 0.0},
        1: {"n": 0, "sum_p": 0.0, "sumsq_p": 0.0, "sum_m": 0.0, "sumsq_m": 0.0},
    }

    model.eval()
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        # AMP is usually fine on GPU; if unstable, disable it
        if device.type == "cuda":
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

    rows = []
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
            "level": level,
            "true_label": cls,
            "mean_prob_fake": mean_p,
            "std_prob_fake": std_p,
            "mean_margin": mean_m,
            "std_margin": std_m,
            "accuracy_overall": float(acc),
            "n": int(n),
        })

    return rows


def parse_levels(s: str) -> list[int]:
    s = s.strip()
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="CLIP noise sweep with corruption BEFORE normalization (image space).")
    ap.add_argument("--project-root", type=str, default=".")
    ap.add_argument("--csv-path", type=str, required=True)
    ap.add_argument("--clip-ckpt", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default="analysis/noise_sweep_A5_clip_pre_norm.csv")

    ap.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32",
                    help="HF model name or local directory path")
    ap.add_argument("--local-files-only", action="store_true",
                    help="Force offline load of clip model/processor")

    ap.add_argument("--levels", type=str, default="0-10")
    ap.add_argument("--max-per-class", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--noise-step", type=float, default=0.01,
                    help="Std of Gaussian noise in [0,1] image space per level (e.g., 0.01).")
    ap.add_argument("--blur-step", type=float, default=0.25,
                    help="Gaussian blur radius per level in pixels (e.g., 0.25).")
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()
    ckpt_path = (project_root / args.clip_ckpt).resolve()
    out_csv = (project_root / args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CSV:", csv_path)
    print("CLIP ckpt:", ckpt_path)
    levels = parse_levels(args.levels)
    print("Levels:", levels)

    # Load CSV and balanced subset from test
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

    # Load CLIP model + processor (supports offline/local)
    model_id = args.clip_model
    clip_processor = CLIPImageProcessor.from_pretrained(
        model_id, local_files_only=args.local_files_only
    )
    clip_vision = CLIPVisionModel.from_pretrained(
        model_id, local_files_only=args.local_files_only
    ).to(device)

    for p in clip_vision.parameters():
        p.requires_grad = False

    clip = ClipDomainClassifier(clip_vision, clip_vision.config.hidden_size).to(device)
    clip.load_state_dict(torch.load(ckpt_path, map_location=device))
    clip.eval()

    # Sweep
    all_rows = []
    for level in levels:
        ds = ClipPreNormSweepDataset(
            df_subset=df_sub,
            project_root=project_root,
            clip_processor=clip_processor,
            level=level,
            noise_step=args.noise_step,
            blur_step=args.blur_step,
            seed=args.seed,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=clip_collate,
        )

        rows = run_sweep_streaming(clip, device, loader, level)
        for r in rows:
            r["model"] = "clip_pre_norm"
        all_rows.extend(rows)

        acc = rows[0]["accuracy_overall"]
        print(f"[clip_pre_norm] level={level:02d} acc={acc*100:.2f}%")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = pd.DataFrame(all_rows)
    out.to_csv(out_csv, index=False)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()