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

from src.models import DomainCNN
from src.datasets import DomainClassificationDataset, pad_collate_fn


def create_test_loader(
    csv_path: Path,
    project_root: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df_test = df[df["hf_split"] == "test"].copy()
    print("Full CSV shape:", df.shape)
    print("Test split shape:", df_test.shape)

    test_ds = DomainClassificationDataset(
        csv_path=csv_path,
        project_root=project_root,
        split="test",
        augment_train=False,
        photo_aug_train=False,
        normalize_all=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=pad_collate_fn,
    )
    return test_loader, df_test


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    criterion = nn.CrossEntropyLoss()

    all_true, all_pred, all_prob_fake = [], [], []
    all_sources, all_idxs, all_texts = [], [], []

    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["images"].to(device)      # (B, 1, H, Wmax)
        labels = batch["labels"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        all_true.extend(labels.cpu().numpy().tolist())
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


@torch.no_grad()
def compute_tsne_features(
    model: DomainCNN,
    loader: DataLoader,
    device: torch.device,
    max_points: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Extract DomainCNN features (after conv blocks + GAP) and run t-SNE.
    """
    from sklearn.manifold import TSNE

    all_feats, all_true, all_pred, all_prob_fake = [], [], [], []
    all_sources, all_idxs, all_texts = [], [], []

    model.eval()
    for batch in loader:
        images = batch["images"].to(device)
        labels = batch["labels"]

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)

        # Feature extraction: use the model's conv_block and GAP as in forward()
        feat_map = model.conv_block(images)         # (B, C, H', W')
        feats = feat_map.mean(dim=(2, 3))           # GAP -> (B, C)

        all_feats.append(feats.cpu().numpy())
        all_true.extend(labels.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob_fake.extend(probs[:, 1].cpu().numpy().tolist())
        all_sources.extend(batch["sources"])
        all_idxs.extend(batch["idxs"])
        all_texts.extend(batch["texts"])

        if len(all_true) >= max_points:
            break

    X = np.concatenate(all_feats, axis=0)
    N = min(len(all_true), max_points)
    X = X[:N]

    y_true = np.array(all_true[:N])
    y_pred = np.array(all_pred[:N])
    prob_fake = np.array(all_prob_fake[:N])
    sources_sub = all_sources[:N]
    idxs_sub = all_idxs[:N]
    texts_sub = all_texts[:N]

    print(f"\nComputing t-SNE on N={N} points, feature dim={X.shape[1]}")
    tsne = TSNE(n_components=2, init="random", learning_rate="auto", random_state=42, perplexity=30)
    X_emb = tsne.fit_transform(X)

    tsne_df = pd.DataFrame({
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_fake": prob_fake,
        "source": sources_sub,
        "idx": idxs_sub,
        "text": texts_sub,
        "tsne_x": X_emb[:, 0],
        "tsne_y": X_emb[:, 1],
    })
    return X_emb, tsne_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DomainCNN checkpoint (test metrics + optional t-SNE).")
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--csv-path", type=str, required=True, help="CSV path relative to project root.")
    parser.add_argument("--tag", type=str, required=True, help="Used for checkpoint name and output files.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tsne", action="store_true")
    parser.add_argument("--tsne-max-points", type=int, default=800)

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    csv_path = (project_root / args.csv_path).resolve()
    ckpt_path = project_root / "checkpoints" / f"{args.tag}.pt"

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

    test_loader, _ = create_test_loader(
        csv_path=csv_path,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DomainCNN().to(device)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded model checkpoint.")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    results_df = evaluate(model, test_loader, device)
    out_csv = analysis_dir / f"{args.tag}_test_results.csv"
    results_df.to_csv(out_csv, index=False)
    print("Saved per-sample test results to:", out_csv)

    if args.tsne:
        X_emb, tsne_df = compute_tsne_features(model, test_loader, device, max_points=args.tsne_max_points)
        out_tsne_csv = analysis_dir / f"{args.tag}_tsne.csv"
        tsne_df.to_csv(out_tsne_csv, index=False)
        print("Saved t-SNE CSV to:", out_tsne_csv)

        out_tsne_npy = analysis_dir / f"{args.tag}_tsne_embedding.npy"
        np.save(out_tsne_npy, X_emb)
        print("Saved t-SNE embedding to:", out_tsne_npy)


if __name__ == "__main__":
    main()