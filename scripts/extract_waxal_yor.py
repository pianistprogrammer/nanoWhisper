#!/usr/bin/env python
"""Extract google/WaxalNLP yor_tts from HuggingFace cache to audio files + manifest CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

CACHE_DIR = "C:/Users/jerem/Documents/Datasets/NgLanguages/huggingface"
AUDIO_OUT = Path("C:/Users/jerem/Documents/Datasets/NgLanguages/waxal_yor_tts/audio")
MANIFEST_OUT = Path("C:/Users/jerem/Documents/Datasets/NgLanguages/waxal_yor_tts")


def extract_split(split_name: str, split_data) -> list[dict[str, str]]:
    out_dir = AUDIO_OUT / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, example in enumerate(split_data):
        audio = example["audio"]
        text = example["text"].strip()
        if not text:
            continue
        sample_id = example["id"] or f"{split_name}_{i:05d}"
        audio_path = out_dir / f"{sample_id}.wav"
        if not audio_path.exists():
            array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            sf.write(str(audio_path), array, sr)
        rows.append({"audio_path": str(audio_path), "text": text})
        if (i + 1) % 200 == 0:
            print(f"  {split_name}: {i+1}/{len(split_data)}")
    return rows


def main() -> None:
    print("Loading dataset from cache...")
    ds = load_dataset("google/WaxalNLP", "yor_tts", cache_dir=CACHE_DIR)

    MANIFEST_OUT.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "validation", "test"):
        print(f"\nExtracting {split_name} ({len(ds[split_name])} examples)...")
        rows = extract_split(split_name, ds[split_name])
        out_name = "val" if split_name == "validation" else split_name
        manifest_path = MANIFEST_OUT / f"{out_name}.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["audio_path", "text"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Wrote {len(rows)} rows to {manifest_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
