#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create nanoWhisper manifests from Common Voice TSV files.")
    parser.add_argument("--common-voice-dir", required=True, help="Language directory, e.g. .../cv-corpus/yo")
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--min-up-votes", type=int, default=0)
    parser.add_argument("--max-down-votes", type=int)
    return parser.parse_args()


def convert_split(
    common_voice_dir: Path,
    split: str,
    output_dir: Path,
    min_up_votes: int,
    max_down_votes: int | None,
) -> int:
    input_path = common_voice_dir / f"{split}.tsv"
    clips_dir = common_voice_dir / "clips"
    output_name = "val.csv" if split == "dev" else f"{split}.csv"
    output_path = output_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with input_path.open("r", encoding="utf-8", newline="") as input_handle:
        reader = csv.DictReader(input_handle, delimiter="\t")
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(output_handle, fieldnames=["audio_path", "text"])
            writer.writeheader()
            for row in reader:
                sentence = (row.get("sentence") or "").strip()
                clip_name = (row.get("path") or "").strip()
                up_votes = int(row.get("up_votes") or 0)
                down_votes = int(row.get("down_votes") or 0)
                audio_path = clips_dir / clip_name
                if not sentence or not clip_name or not audio_path.exists():
                    skipped += 1
                    continue
                if up_votes < min_up_votes:
                    skipped += 1
                    continue
                if max_down_votes is not None and down_votes > max_down_votes:
                    skipped += 1
                    continue
                writer.writerow({"audio_path": str(audio_path), "text": sentence})
                written += 1

    print(f"{split}: wrote {written} rows to {output_path} skipped={skipped}")
    return written


def main() -> None:
    args = parse_args()
    common_voice_dir = Path(args.common_voice_dir).expanduser().resolve()
    output_dir = Path(args.output_dir)
    if not common_voice_dir.exists():
        raise FileNotFoundError(common_voice_dir)
    for split in args.splits:
        convert_split(common_voice_dir, split, output_dir, args.min_up_votes, args.max_down_votes)


if __name__ == "__main__":
    main()

