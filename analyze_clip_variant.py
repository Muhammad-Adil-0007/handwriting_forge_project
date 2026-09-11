#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPVisionModel, CLIPImageProcessor

from src.clip_domain_datasets import ClipDomainDataset, clip_collate_fn


class ClipDomainClassifier(nn.Module):
    def __init__(self, clip_vision: CLIPVisionModel, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.clip_vision = clip_vision
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.clip_vision(pixel_values=pixel_values)
        feats = out.pooler_output
        return self.classifier(feats)


def create_test_loader(
    csv_path: Path,
    project_root: Path,
    clip_processor: CLIPImageProcessor,
    batch_size: int,
    num_workers: int,
    normalize_all: bool,
) -> Tuple[DataLoader, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df_test = df[df["hf_split"] == "test"].copy()
    print("Full CSV shape:", df.shape)
    print("Test split shape:", df_test.shape)

    test_ds = ClipDomainDataset(
        csv_path=csv_path,
        project_root=project_root,
        split="test",
        clip_processor=clip_processor,
        augment=False,
        photo_augment=False,
        normalize_all=normalize_all,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=clip_collate_fn,
    )
    return test_loader, df_test


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    criterion = nn.CrossEntropyLoss()

    all_true, all_pred, all_prob_fake = [], [], []
    all_sources, all_idxs, all_texts = [], [], []

    total_loss = 0.0
    total_samples = 0
    model.eval()

    for batch in loader:
        x = batch["pixel_values"].to(device)
        y = batch["labels"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        all_true.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob_fake.extend(probs[:, 1].cpu().numpy().tolist())

        all_sources.extend(batch["sources"])
        all_idxs.extend(batch["idxs"])
        all_texts.extend(batch["texts"])

    test_loss = total_loss / total_samples
    test_acc = (np.array(all_true) == np.array(all_pred)).mean()

    df_res = pd.DataFrame({
        "true_label": all_true,
        "pred_label": all_pred,
        "prob_fake": all_prob_fake,
        "source": all_sources,
        "idx": all_idxs,
        "text": all_texts,
    })

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc*100:.2f}%")

    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(df_res["true_label"], df_res["pred_label"]):
        cm[t, p] += 1

    print("\nConfusion matrix (rows=true, cols=pred):")
    print("       pred=0    pred=1")
    print(f"true=0   {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"true=1   {cm[1,0]:6d}   {cm[1,1]:6d}")

    acc_iam = cm[0,0] / cm[0].sum() if cm[0].sum() > 0 else 0.0
    acc_emu = cm[1,1] / cm[1].sum() if cm[1].sum() > 0 else 0.0
    print("\nPer-class accuracy:")
    print(f"  Class 0 (IAM / genuine): {acc_iam*100:.2f}%")
    print(f"  Class 1 (Emuru / fake) : {acc_emu*100:.2f}%")

    return df_res


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CLIP-head checkpoint (test metrics).")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="Used for output file naming.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to classifier checkpoint (.pt).")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--normalize-all", action="store_true")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()
    ckpt_path = (project_root / args.ckpt_path).resolve()

    print("Project root:", project_root)
    print("CSV path    :", csv_path)
    print("Checkpoint  :", ckpt_path)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    analysis_dir = project_root / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    print(f"Loading CLIP backbone: {args.clip_model} (local_files_only={args.local_files_only})")
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=args.local_files_only)
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model, local_files_only=args.local_files_only).to(device)
    for p in clip_vision.parameters():
        p.requires_grad = False

    model = ClipDomainClassifier(clip_vision, clip_vision.config.hidden_size, num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print("Loaded classifier checkpoint.")

    test_loader, _ = create_test_loader(
        csv_path=csv_path,
        project_root=project_root,
        clip_processor=clip_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_all=args.normalize_all,
    )

    results_df = evaluate(model, test_loader, device)
    out_csv = analysis_dir / f"{args.tag}_test_results.csv"
    results_df.to_csv(out_csv, index=False)
    print("Saved per-sample test results to:", out_csv)


if __name__ == "__main__":
    main()