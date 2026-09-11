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

from src.datasets import DomainClassificationDataset, pad_collate_fn
from src.models import DomainCNN


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / "domain_cnn_best_A1_geo.pt"


def create_dataloaders(
    csv_path: Path,
    project_root: Path,
    batch_size: int,
    num_workers: int = 4,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train / val / test DataLoaders from the domain_classification CSV,
    based on the 'hf_split' column.

    For A1_geo_only:
    - train_ds uses augment_train=True (geometric augmentation).
    - val/test use augment_train=False (no augmentation).
    """
    df = pd.read_csv(csv_path)
    splits = set(df["hf_split"].unique())
    print("Found splits:", splits)

    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None

    if "train" in splits:
        train_ds = DomainClassificationDataset(
            csv_path=csv_path,
            project_root=project_root,
            split="train",
            augment_train=True,   # <--- geometric aug enabled here
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
        )

    # handle potential validation naming
    val_split_name = None
    for candidate in ["validation", "valid", "val"]:
        if candidate in splits:
            val_split_name = candidate
            break

    if val_split_name is not None:
        val_ds = DomainClassificationDataset(
            csv_path=csv_path,
            project_root=project_root,
            split=val_split_name,
            augment_train=False,  # no aug at eval time
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
        )

    if "test" in splits:
        test_ds = DomainClassificationDataset(
            csv_path=csv_path,
            project_root=project_root,
            split="test",
            augment_train=False,  # no aug at eval time
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pad_collate_fn,
        )

    return train_loader, val_loader, test_loader


def run_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> Tuple[float, float]:
    """
    If optimizer is provided -> training mode
    If optimizer is None    -> eval mode

    Returns:
        avg_loss, accuracy
    """
    if loader is None:
        return 0.0, 0.0

    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["images"].to(device)   # (B, 1, 64, W_max)
        labels = batch["labels"].to(device)   # (B,)

        if is_train:
            optimizer.zero_grad()

        logits = model(images)               # (B, 2)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    acc = total_correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, acc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train IAM vs Emuru domain classifier with A1_geo_only augmentation"
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
        default="data/processed/metadata/domain_classification_sentences.csv",
        help="Path to domain classification CSV (relative to project root)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cpu', or 'cuda'",
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

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=csv_path,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if train_loader is None:
        raise RuntimeError("No 'train' split found in CSV. Cannot train.")

    # Model, loss, optimizer
    model = DomainCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # ---- Train ----
        train_loss, train_acc = run_one_epoch(
            train_loader, model, device, criterion, optimizer
        )
        print(f"  Train - loss: {train_loss:.4f} | acc: {train_acc*100:.2f}%")

        # ---- Validation ----
        val_acc = None
        if val_loader is not None:
            with torch.no_grad():
                val_loss, val_acc = run_one_epoch(
                    val_loader, model, device, criterion, optimizer=None
                )
            print(f"  Val   - loss: {val_loss:.4f} | acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"  New best val acc! Saved checkpoint to: {BEST_MODEL_PATH}")

    # Load best model (if exists) before testing
    if BEST_MODEL_PATH.exists():
        print(f"\nLoading best model from checkpoint: {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print("\nNo best model checkpoint found; using last epoch model.")

    # Final test evaluation
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
