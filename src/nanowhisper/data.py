from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from nanowhisper.audio import LogMelExtractor, augment_waveform, load_audio, pad_or_trim_features, spec_augment
from nanowhisper.config import NanoWhisperConfig
from nanowhisper.tokenizer import YorubaTokenizer


def read_manifest(path: str | Path, audio_root: str | Path | None = None) -> list[dict[str, str]]:
    manifest_path = Path(path)
    base_dir = Path(audio_root) if audio_root else manifest_path.parent
    if manifest_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    else:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))

    examples = []
    for row in rows:
        audio_path = row.get("audio_path") or row.get("path") or row.get("audio")
        text = row.get("text") or row.get("transcript") or row.get("sentence")
        if not audio_path or text is None:
            raise ValueError(f"Manifest row must include audio_path and text: {row}")
        resolved = Path(audio_path)
        if not resolved.is_absolute():
            resolved = base_dir / resolved
        examples.append({"audio_path": str(resolved), "text": str(text)})
    return examples


class YorubaSpeechDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        manifest: str | Path,
        tokenizer: YorubaTokenizer,
        config: NanoWhisperConfig,
        audio_root: str | Path | None = None,
        augment: bool = False,
    ) -> None:
        self.examples = read_manifest(manifest, audio_root)
        self.tokenizer = tokenizer
        self.config = config
        self.augment = augment
        self.extractor = LogMelExtractor(config)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.examples[index]
        waveform = load_audio(item["audio_path"], self.config)
        if self.augment:
            waveform = augment_waveform(waveform, self.config.sample_rate)
            max_samples = self.config.max_audio_samples
            if waveform.numel() > max_samples:
                waveform = waveform[:max_samples]
        with torch.no_grad():
            features = self.extractor(waveform)
        features = pad_or_trim_features(features, self.config.max_audio_frames)
        if self.augment:
            features = spec_augment(features)
        token_ids = self.tokenizer.encode(item["text"], self.config.max_text_tokens)
        return {
            "features": features,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "text": item["text"],
            "audio_path": item["audio_path"],
        }


def collate_batch(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    features = torch.stack([item["features"] for item in batch], dim=0)
    max_tokens = max(item["tokens"].numel() for item in batch)
    tokens = torch.full((len(batch), max_tokens), pad_id, dtype=torch.long)
    for row, item in enumerate(batch):
        tokens[row, : item["tokens"].numel()] = item["tokens"]
    return {
        "features": features,
        "tokens": tokens,
        "texts": [item["text"] for item in batch],
        "audio_paths": [item["audio_path"] for item in batch],
    }

