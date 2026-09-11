#!/usr/bin/env python
"""
Batch Emuru image generation from IAM metadata.

This is a scriptified version of `02_generate_emuru_data.ipynb`
intended to run on the HPC cluster (TinyGPU/TinyFat).

It:
- reads IAM line metadata (filepath + text)
- finds lines that do NOT yet have Emuru sentence images
- generates:
  - word-level Emuru images for each 2-word chunk
  - a merged sentence-level image per IAM line
- appends metadata to:
  - emuru_words_metadata.csv
  - emuru_sentences_metadata.csv
  - emuru_failures_metadata.csv
"""

from __future__ import annotations

import argparse
import random
import traceback
from pathlib import Path
from typing import List, Dict, Any

import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F
from transformers import AutoModel


# ---------------------------
# Helper: paths & image utils
# ---------------------------

def load_image(img_path: Path) -> torch.Tensor:
    """
    Load an image for Emuru:
    - Convert to RGB
    - Resize to fixed height 64 (keep aspect ratio)
    - Convert to tensor (C, H, W)
    - Normalize to [-1, 1] with mean=0.5, std=0.5 per channel
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img.width * 64 // img.height, 64))
    t = F.to_tensor(img)
    t = F.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    return t  # (C, H, W)


def append_records_to_csv(records: List[Dict[str, Any]], csv_path: Path) -> None:
    """
    Append one or more records to a CSV file immediately.

    - Creates the file with a header if it doesn't exist.
    - Appends rows without reloading the entire CSV into memory.
    """
    if not records:
        return
    df = pd.DataFrame(records)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", index=False, header=write_header)


def pick_device(device_arg: str = "auto") -> torch.device:
    """
    Decide which torch.device to use.
    - "auto": prefer CUDA, then MPS, then CPU
    - "cpu" / "cuda" / "mps": force that if available, else fall back to CPU
    """
    device_arg = device_arg.lower()

    if device_arg == "cpu":
        print("Forcing CPU.")
        return torch.device("cpu")

    if device_arg == "cuda":
        if torch.cuda.is_available():
            print("Using CUDA GPU.")
            return torch.device("cuda")
        print("Requested CUDA but no GPU available. Falling back to CPU.")
        return torch.device("cpu")

    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Using MPS (Apple silicon).")
            return torch.device("mps")
        print("Requested MPS but it is not available. Falling back to CPU.")
        return torch.device("cpu")

    # auto
    if torch.cuda.is_available():
        print("Auto device: using CUDA GPU.")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Auto device: using MPS.")
        return torch.device("mps")
    print("Auto device: using CPU.")
    return torch.device("cpu")


# ---------------------------
# Core logic
# ---------------------------

def run_generation(
    project_root: Path,
    batch_size: int,
    device_str: str = "auto",
    retry_failed_only: bool = False,
) -> None:
    """Main generation routine (one batch)."""

    project_root = project_root.resolve()
    print(f"PROJECT_ROOT: {project_root}")

    # --- Paths (adapted from your notebook) ---
    iam_metadata_path = project_root / "data" / "processed" / "metadata" / "iam_metadata.csv"

    emuru_base_dir = project_root / "data" / "raw" / "emuru"
    emuru_words_dir = emuru_base_dir / "words"
    emuru_sentences_dir = emuru_base_dir / "sentences"

    emuru_words_dir.mkdir(parents=True, exist_ok=True)
    emuru_sentences_dir.mkdir(parents=True, exist_ok=True)

    meta_base_dir = project_root / "data" / "processed" / "metadata"
    meta_base_dir.mkdir(parents=True, exist_ok=True)

    emuru_words_metadata_path = meta_base_dir / "emuru_words_metadata.csv"
    emuru_sentences_metadata_path = meta_base_dir / "emuru_sentences_metadata.csv"
    emuru_failures_metadata_path = meta_base_dir / "emuru_failures_metadata.csv"

    print("IAM metadata:", iam_metadata_path)
    print("Words images dir:", emuru_words_dir)
    print("Sentences images dir:", emuru_sentences_dir)
    print("Words metadata CSV:", emuru_words_metadata_path)
    print("Sentences metadata CSV:", emuru_sentences_metadata_path)
    print("Failures metadata CSV:", emuru_failures_metadata_path)

    if not iam_metadata_path.exists():
        raise FileNotFoundError(f"IAM metadata CSV not found: {iam_metadata_path}")

    iam_df = pd.read_csv(iam_metadata_path)
    print("IAM df shape:", iam_df.shape)

    # --- Which IAM lines are already done? (from sentences metadata) ---
    if emuru_sentences_metadata_path.exists():
        sent_meta = pd.read_csv(emuru_sentences_metadata_path)
        done_iam_paths = set(sent_meta["iam_filepath"].astype(str))
        print(f"Loaded sentences metadata with {len(sent_meta)} rows.")
    else:
        done_iam_paths = set()
        print("No existing sentences metadata found. Starting from scratch.")

    # --- Which IAM lines have already failed? (from failures metadata) ---
    if emuru_failures_metadata_path.exists():
        fail_meta = pd.read_csv(emuru_failures_metadata_path)
        failed_iam_paths = set(fail_meta["iam_filepath"].astype(str))
        print(f"Loaded failures metadata with {len(fail_meta)} rows.")
    else:
        failed_iam_paths = set()
        print("No existing failures metadata found.")

    if retry_failed_only:
        # We only want IAM lines that have failed but are not yet successful
        target_paths = failed_iam_paths - done_iam_paths
        remaining_mask = iam_df["filepath"].astype(str).isin(target_paths)
        mode_desc = "FAILED-ONLY RETRY MODE"
    else:
        # Normal mode: skip both successful and failed
        skip_iam_paths = done_iam_paths.union(failed_iam_paths)
        remaining_mask = ~iam_df["filepath"].astype(str).isin(skip_iam_paths)
        mode_desc = "NORMAL MODE"

    remaining_df = iam_df[remaining_mask].copy()

    print(f"\n=== {mode_desc} ===")
    print(f"Total IAM lines: {len(iam_df)}")
    print(f"Already processed (sentences): {len(done_iam_paths)}")
    print(f"Marked as failed: {len(failed_iam_paths)}")
    print(f"Remaining to process in this mode: {len(remaining_df)}")

    if remaining_df.empty:
        print("Nothing left to process for this mode.")
        return

    # Pick batch
    batch_df = remaining_df.head(batch_size)
    print(f"Processing {len(batch_df)} IAM lines in this batch.")

    # --- Device + model (cluster-friendly version of your cell) ---
    device = pick_device(device_str)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = AutoModel.from_pretrained(
        "blowing-up-groundhogs/emuru",
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    print("Emuru model loaded.")

    # --- Counters just for logging ---
    num_word_records = 0
    num_sentence_records = 0
    num_failure_records = 0

    # --- Main loop (direct port of your notebook logic) ---
    with torch.no_grad():
        for row_idx, row in batch_df.iterrows():
            iam_rel_path = Path(row["filepath"])
            iam_abs_path = project_root / iam_rel_path
            full_text = str(row["text"]).strip()

            if not full_text:
                print(f"\nSkipping IAM row {row_idx}: empty text.")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "empty_text",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": "EmptyText",
                    "error_message": "IAM text is empty.",
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                continue

            print("\n" + "=" * 60)
            print(f"IAM row index: {row_idx}")
            print("IAM full text:", full_text)
            print("IAM image path:", iam_abs_path)

            # --- Chunk into 2-word phrases ---
            words = full_text.split()
            chunks: List[str] = []
            j = 0
            while j < len(words) - 1:
                chunk = f"{words[j]} {words[j + 1]}"
                chunks.append(chunk)
                j += 2
            if j < len(words):
                chunks.append(words[j])

            if not chunks:
                print(f"Skipping IAM row {row_idx}: could not build chunks.")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "chunking",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": "NoChunks",
                    "error_message": "Could not build any chunks from text.",
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                continue

            # Emuru style text & style image
            style_text = full_text[:80]
            try:
                style_img = load_image(iam_abs_path).to(device)
            except Exception as e_load:
                print(f"⚠️  Error loading IAM image for row {row_idx}: {e_load}")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "load_iam",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": type(e_load).__name__,
                    "error_message": str(e_load),
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                continue

            iam_stem = iam_rel_path.stem

            # Per-sample buffers (only commit if the whole line succeeds)
            sample_word_images: List[Image.Image] = []
            sample_word_records: List[Dict[str, Any]] = []
            sample_saved_paths: List[Path] = []
            sample_failed = False

            # --- generate word-level images ---
            for chunk_index, gen_text in enumerate(chunks):
                print(f"Generating chunk {chunk_index}: {repr(gen_text)}")

                try:
                    emuru_word_img = model.generate(
                        style_text=style_text,
                        gen_text=gen_text,
                        style_img=style_img,
                        max_new_tokens=256,
                    )

                    # Ensure it's fully loaded & RGB
                    emuru_word_img.load()
                    safe_word_img = emuru_word_img.convert("RGB")

                    word_filename = f"{iam_stem}_chunk_{chunk_index:02d}.png"
                    word_abs_path = emuru_words_dir / word_filename

                    try:
                        # First attempt: normal save
                        safe_word_img.save(word_abs_path)
                    except Exception as e_save:
                        print(f"⚠️  Save error on chunk {chunk_index} of IAM row {row_idx}: {e_save}")

                        if retry_failed_only:
                            # Fallback: rebuild image via numpy -> Image.fromarray
                            print("   Attempting numpy->Image fallback for this chunk...")
                            try:
                                arr = np.asarray(emuru_word_img)

                                # Debug info (helpful when things still fail)
                                print(f"   Fallback debug: arr.shape={arr.shape}, dtype={arr.dtype}")

                                # Ensure uint8 [0, 255]
                                if np.issubdtype(arr.dtype, np.floating):
                                    arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
                                else:
                                    arr = np.clip(arr, 0, 255).astype("uint8")

                                # Handle shape / mode
                                if arr.ndim == 2:
                                    mode = "L"
                                elif arr.ndim == 3:
                                    if arr.shape[2] == 1:
                                        arr = arr[:, :, 0]
                                        mode = "L"
                                    elif arr.shape[2] == 3:
                                        mode = "RGB"
                                    elif arr.shape[2] == 4:
                                        mode = "RGBA"
                                    else:
                                        raise ValueError(f"Unexpected channel count in array: {arr.shape}")
                                else:
                                    raise ValueError(f"Unexpected array shape from Emuru: {arr.shape}")

                                h, w = arr.shape[:2]
                                if w <= 0 or h <= 0:
                                    raise ValueError(f"Invalid image size from Emuru: {w}x{h}")

                                rebuilt_img = Image.fromarray(arr, mode=mode).convert("RGB")
                                rebuilt_img.save(word_abs_path)
                                safe_word_img = rebuilt_img  # use rebuilt for further processing
                                print("   Fallback save via numpy->Image succeeded.")
                            except Exception as e_fix:
                                print(f"   Fallback save via numpy->Image FAILED: {e_fix}")
                                rec = {
                                    "iam_index": row_idx,
                                    "iam_filepath": str(iam_rel_path),
                                    "iam_text": full_text,
                                    "stage": "word_save",
                                    "chunk_index": chunk_index,
                                    "chunk_text": gen_text,
                                    "error_type": type(e_fix).__name__,
                                    "error_message": f"Original save error: {e_save}; Fallback error: {e_fix}",
                                }
                                append_records_to_csv([rec], emuru_failures_metadata_path)
                                num_failure_records += 1
                                sample_failed = True
                                break
                        else:
                            # Normal mode: just record failure and abort this line
                            rec = {
                                "iam_index": row_idx,
                                "iam_filepath": str(iam_rel_path),
                                "iam_text": full_text,
                                "stage": "word_save",
                                "chunk_index": chunk_index,
                                "chunk_text": gen_text,
                                "error_type": type(e_save).__name__,
                                "error_message": str(e_save),
                            }
                            append_records_to_csv([rec], emuru_failures_metadata_path)
                            num_failure_records += 1
                            sample_failed = True
                            break

                    word_rel_path = word_abs_path.relative_to(project_root)

                    sample_word_images.append(safe_word_img)
                    sample_saved_paths.append(word_abs_path)
                    sample_word_records.append({
                        "iam_index": row_idx,
                        "iam_filepath": str(iam_rel_path),
                        "iam_text": full_text,
                        "chunk_index": chunk_index,
                        "chunk_text": gen_text,
                        "emuru_word_filepath": str(word_rel_path),
                        "style_text": style_text,
                    })

                except Exception as e_gen:
                    print(f"⚠️  Generation error on chunk {chunk_index} of IAM row {row_idx}: {e_gen}")
                    rec = {
                        "iam_index": row_idx,
                        "iam_filepath": str(iam_rel_path),
                        "iam_text": full_text,
                        "stage": "word_generate",
                        "chunk_index": chunk_index,
                        "chunk_text": gen_text,
                        "error_type": type(e_gen).__name__,
                        "error_message": str(e_gen),
                    }
                    append_records_to_csv([rec], emuru_failures_metadata_path)
                    num_failure_records += 1
                    sample_failed = True
                    break

            # If any chunk failed, delete partial images and skip this line
            if sample_failed:
                for p in sample_saved_paths:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                print(f"Skipping IAM row {row_idx} due to errors in word-level generation.")
                continue

            # --- Merge word images horizontally into a sentence image ---
            try:
                widths = [im.width for im in sample_word_images]
                heights = [im.height for im in sample_word_images]
                max_height = max(heights)

                # random gaps ONLY between images
                if len(sample_word_images) > 1:
                    gaps = [random.randint(5, 15) for _ in range(len(sample_word_images) - 1)]
                else:
                    gaps = []

                total_width = sum(widths) + sum(gaps)
                merged = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

                x_offset = 0
                for i, im in enumerate(sample_word_images):
                    y_offset = (max_height - im.height) // 2
                    merged.paste(im, (x_offset, y_offset))
                    x_offset += im.width
                    if i < len(sample_word_images) - 1:
                        x_offset += gaps[i]

                sentence_filename = f"{iam_stem}_sentence.png"
                sentence_abs_path = emuru_sentences_dir / sentence_filename

                try:
                    merged.save(sentence_abs_path)
                except Exception as e_sent_save:
                    print(f"⚠️  Save error on merged sentence for IAM row {row_idx}: {e_sent_save}")
                    rec = {
                        "iam_index": row_idx,
                        "iam_filepath": str(iam_rel_path),
                        "iam_text": full_text,
                        "stage": "sentence_save",
                        "chunk_index": None,
                        "chunk_text": None,
                        "error_type": type(e_sent_save).__name__,
                        "error_message": str(e_sent_save),
                    }
                    append_records_to_csv([rec], emuru_failures_metadata_path)
                    num_failure_records += 1
                    # Clean up word images for this line
                    for p in sample_saved_paths:
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    continue

                sentence_rel_path = sentence_abs_path.relative_to(project_root)
                print("Saved merged sentence image to:", sentence_abs_path)

                # Immediately persist word-level metadata for this line
                append_records_to_csv(sample_word_records, emuru_words_metadata_path)
                num_word_records += len(sample_word_records)

                # Immediately persist sentence-level metadata for this line
                sentence_record = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "num_chunks": len(chunks),
                    "chunks_text": " || ".join(chunks),
                    "emuru_sentence_filepath": str(sentence_rel_path),
                    "style_text": style_text,
                }
                append_records_to_csv([sentence_record], emuru_sentences_metadata_path)
                num_sentence_records += 1

            except Exception as e_merge:
                print(f"⚠️  Merge error for IAM row {row_idx}: {e_merge}")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "sentence_merge",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": type(e_merge).__name__,
                    "error_message": str(e_merge),
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                # Clean up word images for this line
                for p in sample_saved_paths:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                continue

    print("\nBatch generation finished.")
    print(f"New word records this run: {num_word_records}")
    print(f"New sentence records this run: {num_sentence_records}")
    print(f"New failure records this run: {num_failure_records}")
    print("\nMetadata updated incrementally (per line).")
    print("Done.")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Emuru handwriting data from IAM metadata.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to project root (contains data/, notebooks/, src/). "
             "Default: directory of this script.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many IAM lines to process in this run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Which device to use for the model (default: auto).",
    )
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help="If set, only process IAM lines that previously failed and "
             "do NOT yet have a successful Emuru sentence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_generation(
            project_root=args.project_root,
            batch_size=args.batch_size,
            device_str=args.device,
            retry_failed_only=args.retry_failed_only,
        )
    except Exception:
        print("\n❌ Unhandled error in generation:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
