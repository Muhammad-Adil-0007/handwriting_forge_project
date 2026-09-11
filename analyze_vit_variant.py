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

import torchvision.models as tv_models

from src.imagenet_domain_datasets import ImageNetDomainDataset, imagenet_collate_fn


class ViTDomainClassifier(nn.Module):
    """
    Matches train_vit_domain_classifier.py:
    - torchvision ViT backbone
    - heads.head replaced with Linear(..., 2)
    """
    def __init__(self, backbone_name: str = "vit_b_16", num_classes: int = 2):
        super().__init__()

        if backbone_name == "vit_b_16":
            self.backbone = tv_models.vit_b_16(weights=None)
        elif backbone_name == "vit_l_16":
            self.backbone = tv_models.vit_l_16(weights=None)
        else:
            raise ValueError(f"Unsupported ViT backbone: {backbone_name}")

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_test_loader(
    csv_path: Path,
    project_root: Path,
    batch_size: int,
    num_workers: int,
    normalize_all: bool,
) -> Tuple[DataLoader, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df_test = df[df["hf_split"] == "test"].copy()
    print("Full CSV shape:", df.shape)
    print("Test split shape:", df_test.shape)

    test_ds = ImageNetDomainDataset(
        csv_path=csv_path,
        project_root=project_root,
        split="test",
        image_size=224,
        augment_geo=False,
        augment_photo=False,
        normalize_all=normalize_all,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=imagenet_collate_fn,
    )

    return test_loader, df_test


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    """
    Run inference on the test loader and return a DataFrame with:
    true_label, pred_label, prob_fake, source, idx, text
    """
    criterion = nn.CrossEntropyLoss()

    all_true = []
    all_pred = []
    all_prob_fake = []
    all_sources = []
    all_idxs = []
    all_texts = []

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
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

    test_loss = total_loss / total_samples if total_samples > 0 else 0.0
    test_acc = (np.array(all_true) == np.array(all_pred)).mean() if total_samples > 0 else 0.0

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

    # Confusion matrix
    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=int)
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
def extract_vit_features(model: ViTDomainClassifier, images: torch.Tensor) -> torch.Tensor:
    """
    Extract ViT features (CLS embedding before the classification head).

    torchvision ViT forward does:
      x = _process_input(x)
      n = x.shape[0]
      batch_class_token = cls_token.expand(n, -1, -1)
      x = torch.cat([batch_class_token, x], dim=1)
      x = encoder(x)
      x = x[:, 0]   # CLS
      x = heads(x)

    We'll replicate up to CLS.
    """
    vit = model.backbone

    x = vit._process_input(images)  # (B, num_patches, hidden_dim)
    n = x.shape[0]
    cls = vit.class_token.expand(n, -1, -1)  # (B, 1, hidden_dim)
    x = torch.cat([cls, x], dim=1)           # (B, 1+num_patches, hidden_dim)
    x = vit.encoder(x)                       # (B, 1+num_patches, hidden_dim)
    feats = x[:, 0]                          # (B, hidden_dim)
    return feats


def compute_tsne_features(
    model: ViTDomainClassifier,
    loader: DataLoader,
    device: torch.device,
    max_points: int,
) -> Tuple[np.ndarray, pd.DataFrame]:
    from sklearn.manifold import TSNE

    all_feats = []
    all_true = []
    all_pred = []
    all_prob_fake = []
    all_sources = []
    all_idxs = []
    all_texts = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            labels = batch["labels"]
            sources = batch["sources"]
            idxs = batch["idxs"]
            texts = batch["texts"]

            logits = model(images)
            probs = F.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)

            feats = extract_vit_features(model, images)  # (B, hidden_dim)

            all_feats.append(feats.cpu().numpy())
            all_true.extend(labels.cpu().numpy().tolist())
            all_pred.extend(pred.cpu().numpy().tolist())
            all_prob_fake.extend(probs[:, 1].cpu().numpy().tolist())
            all_sources.extend(sources)
            all_idxs.extend(idxs)
            all_texts.extend(texts)

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
    tsne = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        random_state=42,
        perplexity=30,
    )
    X_emb = tsne.fit_transform(X)
    print("t-SNE embedding shape:", X_emb.shape)

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
    parser = argparse.ArgumentParser(
        description="Analyze a ViT domain classifier checkpoint (test metrics + optional t-SNE)."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Project root (default: script's parent).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to domain CSV (relative to project root).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Run tag, e.g. vit_baseline, vit_A4_realistic, etc. Used to find checkpoint and name outputs.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="vit_b_16",
        help="ViT backbone name (must match training): vit_b_16 or vit_l_16",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--normalize-all",
        action="store_true",
        help="A3-style contrast normalization for ALL splits (must match training).",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="If set, also compute t-SNE on ViT CLS features.",
    )
    parser.add_argument(
        "--tsne-max-points",
        type=int,
        default=800,
        help="Max number of points to use for t-SNE (default: 800).",
    )

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

    # Data
    test_loader, _ = create_test_loader(
        csv_path=csv_path,
        project_root=project_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize_all=args.normalize_all,
    )

    # Model
    model = ViTDomainClassifier(backbone_name=args.backbone, num_classes=2).to(device)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        print("Loaded model checkpoint.")
    else:
        print("WARNING: checkpoint not found; using random weights!")

    # ---- Evaluation ----
    results_df = evaluate(model, test_loader, device)

    out_csv = analysis_dir / f"{args.tag}_test_results.csv"
    results_df.to_csv(out_csv, index=False)
    print("Saved per-sample test results to:", out_csv)

    # ---- t-SNE (optional) ----
    if args.tsne:
        X_emb, tsne_df = compute_tsne_features(
            model, test_loader, device, max_points=args.tsne_max_points
        )
        out_tsne_csv = analysis_dir / f"{args.tag}_tsne.csv"
        tsne_df.to_csv(out_tsne_csv, index=False)
        print("Saved t-SNE CSV to:", out_tsne_csv)

        out_tsne_npy = analysis_dir / f"{args.tag}_tsne_embedding.npy"
        np.save(out_tsne_npy, X_emb)
        print("Saved t-SNE embedding (numpy) to:", out_tsne_npy)


if __name__ == "__main__":
    main()
