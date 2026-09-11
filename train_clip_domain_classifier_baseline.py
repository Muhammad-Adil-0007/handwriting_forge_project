#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import argparse
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import CLIPVisionModel, CLIPImageProcessor

from src.clip_domain_datasets import ClipDomainDataset, clip_collate_fn


class ClipDomainClassifier(nn.Module):
    def __init__(self, clip_vision: CLIPVisionModel, hidden_size: int, num_classes: int = 2):
        super().__init__()
        self.clip_vision = clip_vision
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.clip_vision(pixel_values=pixel_values)
        feats = outputs.pooler_output  # (B, hidden_size)
        logits = self.classifier(feats)
        return logits


def create_dataloaders(
    csv_path: Path,
    project_root: Path,
    clip_processor: CLIPImageProcessor,
    batch_size: int,
    num_workers: int = 4,
    augment_train: bool = False,
    photo_augment_train: bool = False,
    normalize_all: bool = False,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train / val / test DataLoaders from the domain CSV.

    augment_train:       geometric augmentation for train split (A1-style)
    photo_augment_train: photometric augmentation for train split (A2-style)
    normalize_all:       contrast normalization for ALL splits (A3-style)
    """
    df = pd.read_csv(csv_path)
    splits = set(df["hf_split"].unique())
    print("Found splits:", splits)

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    if "train" in splits:
        train_ds = ClipDomainDataset(
            csv_path=csv_path,
            project_root=project_root,
            split="train",
            clip_processor=clip_processor,
            augment=augment_train,
            photo_augment=photo_augment_train,
            normalize_all=normalize_all,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=clip_collate_fn,
        )

    # handle validation naming
    val_split_name = None
    for candidate in ["validation", "valid", "val"]:
        if candidate in splits:
            val_split_name = candidate
            break

    if val_split_name is not None:
        val_ds = ClipDomainDataset(
            csv_path=csv_path,
            project_root=project_root,
            split=val_split_name,
            clip_processor=clip_processor,
            augment=False,
            photo_augment=False,
            normalize_all=normalize_all,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=clip_collate_fn,
        )

    if "test" in splits:
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

    return train_loader, val_loader, test_loader


def run_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[float, float]:
    if loader is None:
        return 0.0, 0.0

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(pixel_values)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * pixel_values.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += pixel_values.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IAM vs Emuru domain classifier using a frozen CLIP vision encoder (Baseline / A1–A5)."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Path to project root (default: script's parent directory)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/processed/metadata/domain_classification_sentences_A5_emuru_realistic_noisy.csv",
        help="Path to domain classification CSV (relative to project root). Default: A5 CSV.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="A5_clip",
        help="Tag for this run (checkpoint will be checkpoints/clip_domain_<tag>.pt). Default: A5_clip",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cpu', or 'cuda'",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HuggingFace model name (or local path) for CLIP vision encoder",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="If set, load CLIP model/processor from local cache only (no internet).",
    )
    parser.add_argument(
        "--augment-train",
        action="store_true",
        help="If set, apply geometric augmentation to train split (A1-style).",
    )
    parser.add_argument(
        "--photo-augment-train",
        action="store_true",
        help="If set, apply photometric augmentation to train split (A2-style).",
    )
    parser.add_argument(
        "--normalize-all",
        action="store_true",
        help="If set, apply contrast normalization to ALL splits (A3-style).",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()

    print("Project root:", project_root)
    print("CSV path    :", csv_path)

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:  # auto
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Checkpoint path
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / f"clip_domain_{args.tag}.pt"
    print("Checkpoint will be saved to:", ckpt_path)

    # Load CLIP (offline-friendly)
    print(f"\nLoading CLIP vision model: {args.clip_model} (local_files_only={args.local_files_only})")
    clip_processor = CLIPImageProcessor.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    )
    clip_vision = CLIPVisionModel.from_pretrained(
        args.clip_model, local_files_only=args.local_files_only
    ).to(device)

    for p in clip_vision.parameters():
        p.requires_grad = False

    hidden_size = clip_vision.config.hidden_size
    print("CLIP hidden_size:", hidden_size)

    model = ClipDomainClassifier(
        clip_vision=clip_vision,
        hidden_size=hidden_size,
        num_classes=2,
    ).to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        project_root=project_root,
        clip_processor=clip_processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_train=args.augment_train,
        photo_augment_train=args.photo_augment_train,
        normalize_all=args.normalize_all,
    )

    if train_loader is None:
        raise RuntimeError("No 'train' split found in CSV. Cannot train.")

    print("\nTrain geo augmentation enabled   :", args.augment_train)
    print("Train photo augmentation enabled :", args.photo_augment_train)
    print("Normalize_all (A3) enabled       :", args.normalize_all)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_one_epoch(
            train_loader, model, device, criterion, optimizer
        )
        print(f"  Train - loss: {train_loss:.4f} | acc: {train_acc*100:.2f}%")

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = run_one_epoch(
                    val_loader, model, device, criterion, optimizer=None
                )
            print(f"  Val   - loss: {val_loss:.4f} | acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  New best val acc! Saved checkpoint to: {ckpt_path}")

    if ckpt_path.exists():
        print(f"\nLoading best model from checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print("\nNo best model checkpoint found; using last epoch model.")

    if test_loader is not None:
        with torch.no_grad():
            test_loss, test_acc = run_one_epoch(
                test_loader, model, device, criterion, optimizer=None
            )
        print(f"\nTest - loss: {test_loss:.4f} | acc: {test_acc*100:.2f}%")
    else:
        print("\nNo 'test' split found; skipping final test evaluation.")


if __name__ == "__main__":
    main()