#!/usr/bin/env python
"""
Generate more IAM-like Emuru images (A4_emuru_realistic).

- Reads:  data/processed/metadata/domain_classification_sentences.csv
- For rows with source == "emuru":
    - loads the original Emuru image
    - applies a "realism" post-processing (noise, blur, contrast)
    - saves to: data/raw/emuru_A4/<same_filename>
- Writes a new CSV:
    data/processed/metadata/domain_classification_sentences_A4_emuru_realistic.csv

IAM rows are unchanged. Emuru rows get updated filepaths.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageFilter
import pandas as pd


# --- Helper: find project root (directory that has src/ and data/) ---

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "src").is_dir() and (p / "data").is_dir():
            return p
    raise RuntimeError(f"Could not find project root starting from {start}")


# --- Emuru realism transform (offline) ---

def make_emuru_more_realistic(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Take a grayscale Emuru image and make it look a bit more like scanned IAM:
    - slight blur
    - low-frequency background noise
    - mild brightness / contrast variations

    This runs OFFLINE and is deterministic given rng state.
    """
    # Ensure grayscale
    img = img.convert("L")

    # 1) Slight blur (scanner / paper softness)
    if rng.random() < 0.7:
        sigma = rng.uniform(0.3, 0.9)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # 2) Add smooth-ish noise to background
    arr = np.array(img).astype("float32")  # (H, W)
    H, W = arr.shape

    # create low-res noise and upsample
    noise_small_h = max(8, H // 16)
    noise_small_w = max(8, W // 16)
    noise_small = rng.normal(loc=0.0, scale=8.0, size=(noise_small_h, noise_small_w)).astype("float32")

    noise = Image.fromarray(noise_small).resize((W, H), Image.BILINEAR)
    noise = np.array(noise).astype("float32")

    arr = arr + noise  # add subtle paper noise

    # 3) Slight global brightness/contrast jitter
    #    arr' = (arr - 128)*c + 128 + b
    c = rng.uniform(0.9, 1.1)      # contrast
    b = rng.uniform(-10.0, 10.0)   # brightness offset

    arr = (arr - 128.0) * c + 128.0 + b

    # Clip + convert back
    arr = np.clip(arr, 0, 255).astype("uint8")
    out = Image.fromarray(arr, mode="L")
    return out


def main() -> None:
    cwd = Path.cwd()
    project_root = find_project_root(cwd)
    print("Project root:", project_root)

    meta_dir = project_root / "data" / "processed" / "metadata"
    csv_in = meta_dir / "domain_classification_sentences.csv"
    csv_out = meta_dir / "domain_classification_sentences_A4_emuru_realistic.csv"

    if not csv_in.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_in}")

    print("Reading CSV:", csv_in)
    df = pd.read_csv(csv_in)
    print("Original CSV shape:", df.shape)
    print("Sources:\n", df["source"].value_counts())

    # Output directory for Emuru_A4 images
    emuru_out_dir = project_root / "data" / "raw" / "emuru_A4"
    emuru_out_dir.mkdir(parents=True, exist_ok=True)
    print("Emuru A4 output dir:", emuru_out_dir)

    rng = np.random.default_rng(seed=42)

    # We'll create a copy of the dataframe and adjust only the Emuru rows
    df_out = df.copy()

    # Process Emuru rows
    emuru_mask = (df["source"] == "emuru")
    emuru_rows = df[emuru_mask]
    print("Emuru rows:", len(emuru_rows))

    for i, row in emuru_rows.iterrows():
        rel_path_str = row["filepath"]
        rel_path = Path(rel_path_str)
        in_path = project_root / rel_path

        if not in_path.exists():
            print(f"[WARN] Emuru image not found: {in_path} (skipping row idx={i})")
            continue

        try:
            img = Image.open(in_path).convert("L")
        except Exception as e:
            print(f"[WARN] Could not open {in_path}: {e}")
            continue

        img_out = make_emuru_more_realistic(img, rng)

        # Save under same filename but different directory
        out_filename = rel_path.name  # e.g. emuru_00001.png
        out_path = emuru_out_dir / out_filename

        img_out.save(out_path)

        # Update filepath in df_out (relative path from project root)
        new_rel = Path("data") / "raw" / "emuru_A4" / out_filename
        df_out.at[i, "filepath"] = str(new_rel)

        if (i % 500) == 0:
            print(f"Processed Emuru row idx={i} -> {new_rel}")

    print("Done processing Emuru images.")

    print("Writing new CSV:", csv_out)
    df_out.to_csv(csv_out, index=False)
    print("New CSV shape:", df_out.shape)
    print("Example filepaths after update:")
    print(df_out[["source", "filepath"]].head(10))


if __name__ == "__main__":
    main()
