#!/usr/bin/env python
"""Merge existing manifests with TTS datasets from the local drive into new train/val/test splits."""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


TTS_ROOT = Path("C:/Users/jerem/Documents/Datasets/NgLanguages/cmo1nlaah0071mk077mw0qhpv/Yoruba-TTS-Dataset")
WAXAL_ROOT = Path("C:/Users/jerem/Documents/Datasets/NgLanguages/waxal_yor_tts")


def load_existing_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_tts_datasets(root: Path) -> list[dict[str, str]]:
    rows = []
    for dataset_dir in sorted(root.iterdir()):
        mapping = dataset_dir / "mapping.tsv"
        if not mapping.exists():
            continue
        with mapping.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                audio_path = dataset_dir / row["audio_filename"]
                text = row["sentence"].strip()
                if not audio_path.exists():
                    continue
                if not text:
                    continue
                rows.append({"audio_path": str(audio_path), "text": text})
    return rows


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all Yoruba data into new manifests.")
    parser.add_argument("--manifest-dir", default="data/manifests", help="Directory with existing manifests")
    parser.add_argument("--out-dir", default="data/manifests", help="Output directory for merged manifests")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)

    # Load existing splits
    existing = []
    for split in ("train.csv", "val.csv", "test.csv"):
        p = manifest_dir / split
        if p.exists():
            existing.extend(load_existing_manifest(p))
    print(f"Existing examples: {len(existing)}")

    # Load TTS datasets
    tts = load_tts_datasets(TTS_ROOT)
    print(f"TTS examples found: {len(tts)}")

    # Load WaxalNLP Yoruba
    waxal = []
    for split in ("train.csv", "val.csv", "test.csv"):
        p = WAXAL_ROOT / split
        if p.exists():
            waxal.extend(load_existing_manifest(p))
    print(f"WaxalNLP examples found: {len(waxal)}")

    # Deduplicate by audio_path
    seen = {r["audio_path"] for r in existing}
    new_tts = [r for r in tts if r["audio_path"] not in seen]
    seen.update(r["audio_path"] for r in new_tts)
    new_waxal = [r for r in waxal if r["audio_path"] not in seen]
    print(f"New TTS examples (after dedup): {len(new_tts)}")
    print(f"New WaxalNLP examples (after dedup): {len(new_waxal)}")

    all_rows = existing + new_tts + new_waxal
    print(f"Total combined: {len(all_rows)}")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_rows)
    n = len(all_rows)
    n_val = int(n * args.val_ratio)
    n_test = int(n * args.test_ratio)
    n_train = n - n_val - n_test

    train = all_rows[:n_train]
    val = all_rows[n_train:n_train + n_val]
    test = all_rows[n_train + n_val:]

    out_dir = Path(args.out_dir)
    write_manifest(out_dir / "train.csv", train)
    write_manifest(out_dir / "val.csv", val)
    write_manifest(out_dir / "test.csv", test)

    print(f"\nWrote to {out_dir}/")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")


if __name__ == "__main__":
    main()
