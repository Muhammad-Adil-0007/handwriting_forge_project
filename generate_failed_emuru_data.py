#!/usr/bin/env python
"""
Retry Emuru image generation for FAILED IAM lines only.

This script:
- reads IAM line metadata (filepath + text)
- finds IAM lines that appear in emuru_failures_metadata.csv
  but do NOT yet have a successful Emuru sentence
- for those IAM lines:
  - generates word-level Emuru images for each 2-word chunk
  - merges them into a sentence-level image
- appends metadata to:
  - emuru_words_metadata.csv
  - emuru_sentences_metadata.csv
  - emuru_failures_metadata.csv  (for new failures only)

This does NOT touch or re-generate already successful lines.
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

def run_retry_generation(
    project_root: Path,
    batch_size: int,
    device_str: str = "auto",
    max_attempts_per_chunk: int = 3,
) -> None:
    """Retry generation for failed IAM lines only (one batch)."""

    project_root = project_root.resolve()
    print(f"PROJECT_ROOT: {project_root}")

    # --- Paths ---
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
    if not emuru_failures_metadata_path.exists():
        print("No failures metadata found; nothing to retry.")
        return

    iam_df = pd.read_csv(iam_metadata_path)
    fail_meta = pd.read_csv(emuru_failures_metadata_path)
    print("IAM df shape:", iam_df.shape)
    print("Failures df shape:", fail_meta.shape)

    # Already successful lines
    if emuru_sentences_metadata_path.exists():
        sent_meta = pd.read_csv(emuru_sentences_metadata_path)
        done_iam_paths = set(sent_meta["iam_filepath"].astype(str))
        print(f"Loaded sentences metadata with {len(sent_meta)} rows.")
    else:
        sent_meta = None
        done_iam_paths = set()
        print("No existing sentences metadata found.")

    # IAM paths that have failed at least once
    failed_iam_paths = set(fail_meta["iam_filepath"].astype(str))
    print(f"Unique IAM filepaths with failures: {len(failed_iam_paths)}")

    # Retry targets: failed but not yet successful
    target_paths = failed_iam_paths - done_iam_paths
    print(f"Retry targets (failed but not successful yet): {len(target_paths)}")

    if not target_paths:
        print("Nothing left to retry; all failed IAM lines now have sentences.")
        return

    # Filter IAM df down to retry targets
    retry_mask = iam_df["filepath"].astype(str).isin(target_paths)
    retry_df = iam_df[retry_mask].copy()

    print(f"Total IAM lines: {len(iam_df)}")
    print(f"Remaining to retry: {len(retry_df)}")

    if retry_df.empty:
        print("No IAM rows found for retry; exiting.")
        return

    # Pick batch
    batch_df = retry_df.head(batch_size)
    print(f"Processing {len(batch_df)} IAM lines in this RETRY batch.")

    # --- Device + model ---
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

    # --- Main loop (retry) ---
    with torch.no_grad():
        for row_idx, row in batch_df.iterrows():
            iam_rel_path = Path(row["filepath"])
            iam_abs_path = project_root / iam_rel_path
            full_text = str(row["text"]).strip()

            if not full_text:
                print(f"\n[RETRY] Skipping IAM row {row_idx}: empty text.")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "empty_text_retry",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": "EmptyText",
                    "error_message": "IAM text is empty (retry).",
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                continue

            print("\n" + "=" * 60)
            print(f"[RETRY] IAM row index: {row_idx}")
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
                print(f"[RETRY] Skipping IAM row {row_idx}: could not build chunks.")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "chunking_retry",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": "NoChunks",
                    "error_message": "Could not build any chunks from text (retry).",
                }
                append_records_to_csv([rec], emuru_failures_metadata_path)
                num_failure_records += 1
                continue

            # Emuru style text & style image
            style_text = full_text[:80]
            try:
                style_img = load_image(iam_abs_path).to(device)
            except Exception as e_load:
                print(f"⚠️  [RETRY] Error loading IAM image for row {row_idx}: {e_load}")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "load_iam_retry",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": type(e_load).__name__,
                    "error_message": f"(retry) {e_load}",
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

            # --- generate word-level images (RETRY) ---
            for chunk_index, gen_text in enumerate(chunks):
                print(f"[RETRY] Generating chunk {chunk_index}: {repr(gen_text)}")

                success_for_chunk = False
                last_error_msg = None

                for attempt in range(1, max_attempts_per_chunk + 1):
                    print(f"   [RETRY] Attempt {attempt}/{max_attempts_per_chunk} for chunk {chunk_index}")
                    try:
                        emuru_word_img = model.generate(
                            style_text=style_text,
                            gen_text=gen_text,
                            style_img=style_img,
                            max_new_tokens=256,
                        )

                        # Ensure it's fully loaded
                        emuru_word_img.load()
                        safe_word_img = emuru_word_img.convert("RGB")

                        # Check dimensions early
                        w, h = safe_word_img.size
                        if w <= 0 or h <= 0:
                            raise ValueError(f"Invalid image size from Emuru (retry): {w}x{h}")

                        word_filename = f"{iam_stem}_chunk_{chunk_index:02d}.png"
                        word_abs_path = emuru_words_dir / word_filename

                        try:
                            # First attempt: normal save
                            safe_word_img.save(word_abs_path)
                            success_for_chunk = True
                            break
                        except Exception as e_save:
                            print(f"   ⚠️  [RETRY] Save error on chunk {chunk_index} (attempt {attempt}): {e_save}")
                            last_error_msg = str(e_save)

                            # Fallback: numpy -> Image
                            print("       Attempting numpy->Image fallback for this chunk (retry)...")
                            try:
                                arr = np.asarray(emuru_word_img)

                                print(f"       Fallback debug (retry): arr.shape={arr.shape}, dtype={arr.dtype}")

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
                                        raise ValueError(f"Unexpected channel count in array (retry): {arr.shape}")
                                else:
                                    raise ValueError(f"Unexpected array shape from Emuru (retry): {arr.shape}")

                                h2, w2 = arr.shape[:2]
                                if w2 <= 0 or h2 <= 0:
                                    raise ValueError(f"Invalid image size after fallback (retry): {w2}x{h2}")

                                rebuilt_img = Image.fromarray(arr, mode=mode).convert("RGB")
                                rebuilt_img.save(word_abs_path)
                                safe_word_img = rebuilt_img
                                print("       Fallback save via numpy->Image (retry) SUCCEEDED.")
                                success_for_chunk = True
                                break
                            except Exception as e_fix:
                                print(f"       Fallback save via numpy->Image (retry) FAILED: {e_fix}")
                                last_error_msg = f"Original save error: {e_save}; Fallback error: {e_fix}"
                                # loop will continue to next attempt

                    except Exception as e_gen:
                        print(f"   ⚠️  [RETRY] Generation error for chunk {chunk_index} (attempt {attempt}): {e_gen}")
                        last_error_msg = f"Generation error: {e_gen}"
                        # try another attempt

                if not success_for_chunk:
                    # All attempts for this chunk failed -> record and abort line
                    rec = {
                        "iam_index": row_idx,
                        "iam_filepath": str(iam_rel_path),
                        "iam_text": full_text,
                        "stage": "word_retry_exhausted",
                        "chunk_index": chunk_index,
                        "chunk_text": gen_text,
                        "error_type": "RetryExhausted",
                        "error_message": f"(retry) {last_error_msg}",
                    }
                    append_records_to_csv([rec], emuru_failures_metadata_path)
                    num_failure_records += 1
                    sample_failed = True
                    break

                # Success: record this chunk
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

            # If any chunk failed, delete partial images and skip this line
            if sample_failed:
                for p in sample_saved_paths:
                    try:
                        p.unlink()
                    except Exception:
                        pass
                print(f"[RETRY] Skipping IAM row {row_idx} due to errors in word-level generation.")
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
                    print(f"⚠️  [RETRY] Save error on merged sentence for IAM row {row_idx}: {e_sent_save}")
                    rec = {
                        "iam_index": row_idx,
                        "iam_filepath": str(iam_rel_path),
                        "iam_text": full_text,
                        "stage": "sentence_save_retry",
                        "chunk_index": None,
                        "chunk_text": None,
                        "error_type": type(e_sent_save).__name__,
                        "error_message": f"(retry) {e_sent_save}",
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
                print("[RETRY] Saved merged sentence image to:", sentence_abs_path)

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
                print(f"⚠️  [RETRY] Merge error for IAM row {row_idx}: {e_merge}")
                rec = {
                    "iam_index": row_idx,
                    "iam_filepath": str(iam_rel_path),
                    "iam_text": full_text,
                    "stage": "sentence_merge_retry",
                    "chunk_index": None,
                    "chunk_text": None,
                    "error_type": type(e_merge).__name__,
                    "error_message": f"(retry) {e_merge}",
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

    print("\n[RETRY] Batch generation finished.")
    print(f"[RETRY] New word records this run: {num_word_records}")
    print(f"[RETRY] New sentence records this run: {num_sentence_records}")
    print(f"[RETRY] New failure records this run: {num_failure_records}")
    print("\n[RETRY] Metadata updated incrementally (per line).")
    print("[RETRY] Done.")


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry Emuru handwriting data generation for failed IAM lines.")
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
        help="How many IAM lines to process in this retry run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Which device to use for the model (default: auto).",
    )
    parser.add_argument(
        "--max-attempts-per-chunk",
        type=int,
        default=3,
        help="How many times to re-generate a chunk before giving up.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_retry_generation(
            project_root=args.project_root,
            batch_size=args.batch_size,
            device_str=args.device,
            max_attempts_per_chunk=args.max_attempts_per_chunk,
        )
    except Exception:
        print("\n❌ Unhandled error in retry generation:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
