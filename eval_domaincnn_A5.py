from pathlib import Path
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- find project root (folder that contains src/ and data/) ---
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "src").is_dir() and (p / "data").is_dir():
            return p
    raise RuntimeError(f"Could not find project root starting from {start}")

cwd = Path.cwd()
project_root = find_project_root(cwd)
print("CWD         :", cwd)
print("PROJECT_ROOT:", project_root)

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
print("project_root in sys.path:", str(project_root) in sys.path)

# --- imports from your project ---
from src.models import DomainCNN
from src.datasets import DomainClassificationDataset, pad_collate_fn

# --- paths ---
CSV = project_root / "data/processed/metadata/domain_classification_sentences_A5_emuru_realistic_noisy.csv"
CKPT = project_root / "checkpoints/domain_cnn_best_A5_emuru_realistic_noisy.pt"

print("CSV :", CSV)
print("CKPT:", CKPT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --- dataset/loader ---
ds = DomainClassificationDataset(
    csv_path=CSV,
    project_root=project_root,
    split="test",
    augment_train=False,
    photo_aug_train=False,
    normalize_all=False,
)
loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=pad_collate_fn)

# --- model ---
model = DomainCNN().to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

crit = nn.CrossEntropyLoss()
all_true, all_pred = [], []
total_loss, total = 0.0, 0

with torch.no_grad():
    for batch in loader:
        x = batch["images"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        loss = crit(logits, y)
        pred = logits.argmax(1)

        total_loss += loss.item() * y.size(0)
        total += y.size(0)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

y_true = np.concatenate(all_true)
y_pred = np.concatenate(all_pred)

acc = (y_true == y_pred).mean()

cm = np.zeros((2,2), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

acc_iam = cm[0,0] / cm[0].sum() if cm[0].sum() else 0.0
acc_emu = cm[1,1] / cm[1].sum() if cm[1].sum() else 0.0

print("\nTest loss:", total_loss/total)
print("Test acc :", acc*100)
print("Confusion matrix (rows=true, cols=pred):\n", cm)
print(f"Per-class acc: IAM={acc_iam*100:.2f}% | Emuru={acc_emu*100:.2f}%")