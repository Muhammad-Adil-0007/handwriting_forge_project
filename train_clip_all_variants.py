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
        feats = outputs.pooler_output
        return self.classifier(feats)


def create_dataloaders(
    csv_path: Path,
    project_root: Path,
    clip_processor: CLIPImageProcessor,
    batch_size: int,
    num_workers: int,
    augment_train: bool,
    photo_augment_train: bool,
    normalize_all: bool,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    df = pd.read_csv(csv_path)
    splits = set(df["hf_split"].unique())
    print("Found splits:", splits)

    train_loader = val_loader = test_loader = None

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
            pin_memory=torch.cuda.is_available(),
        )

    val_split_name = None
    for cand in ["validation", "valid", "val"]:
        if cand in splits:
            val_split_name = cand
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
            pin_memory=torch.cuda.is_available(),
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
            pin_memory=torch.cuda.is_available(),
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
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

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
    parser = argparse.ArgumentParser(description="Train CLIP-head (frozen CLIP vision encoder) for Baseline/A1–A5.")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--csv-path", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="Checkpoint name: clip_domain_<tag>.pt")
    parser.add_argument("--ckpt-dir", type=str, default="newCheckpoints", help="Directory to save checkpoints")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")

    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--augment-train", action="store_true", help="A1 geo (train only)")
    parser.add_argument("--photo-augment-train", action="store_true", help="A2 photo (train only)")
    parser.add_argument("--normalize-all", action="store_true", help="A3 norm (all splits)")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = (project_root / args.ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"clip_domain_{args.tag}.pt"

    print("Project root:", project_root)
    print("CSV path    :", csv_path)
    print("Using device:", device)
    print("Checkpoint  :", ckpt_path)

    print(f"\nLoading CLIP: {args.clip_model} (local_files_only={args.local_files_only})")
    clip_processor = CLIPImageProcessor.from_pretrained(args.clip_model, local_files_only=args.local_files_only)
    clip_vision = CLIPVisionModel.from_pretrained(args.clip_model, local_files_only=args.local_files_only).to(device)

    for p in clip_vision.parameters():
        p.requires_grad = False

    model = ClipDomainClassifier(clip_vision, clip_vision.config.hidden_size, num_classes=2).to(device)

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

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

    print("\nFlags:")
    print("  augment_train      :", args.augment_train)
    print("  photo_augment_train:", args.photo_augment_train)
    print("  normalize_all      :", args.normalize_all)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = run_one_epoch(train_loader, model, device, criterion, optimizer)
        print(f"  Train - loss: {train_loss:.4f} | acc: {train_acc*100:.2f}%")

        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = run_one_epoch(val_loader, model, device, criterion, optimizer=None)
            print(f"  Val   - loss: {val_loss:.4f} | acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                print(f"  New best val acc! Saved checkpoint to: {ckpt_path}")

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"\nLoaded best checkpoint: {ckpt_path}")

    if test_loader is not None:
        with torch.no_grad():
            test_loss, test_acc = run_one_epoch(test_loader, model, device, criterion, optimizer=None)
        print(f"\nTest - loss: {test_loss:.4f} | acc: {test_acc*100:.2f}%")
    else:
        print("\nNo test split found; skipping test.")


if __name__ == "__main__":
    main()