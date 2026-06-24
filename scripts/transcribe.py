#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanowhisper.audio import LogMelExtractor, load_audio, pad_or_trim_features
from nanowhisper.checkpoint import load_checkpoint
from nanowhisper.config import NanoWhisperConfig
from nanowhisper.model import NanoWhisper
from nanowhisper.tokenizer import YorubaTokenizer
from nanowhisper.training import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe one Yoruba audio file.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/yoruba_tiny.json")
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = NanoWhisperConfig(**checkpoint.get("config", NanoWhisperConfig.from_json(args.config).to_dict()))
    tokenizer = YorubaTokenizer()
    if "vocab" in checkpoint:
        tokenizer.vocab = checkpoint["vocab"]
        tokenizer.char2idx = {char: idx for idx, char in enumerate(tokenizer.vocab)}
        tokenizer.idx2char = {idx: char for idx, char in enumerate(tokenizer.vocab)}
        tokenizer.pad_id = tokenizer.char2idx["[PAD]"]
        tokenizer.bos_id = tokenizer.char2idx["[BOS]"]
        tokenizer.eos_id = tokenizer.char2idx["[EOS]"]
        tokenizer.unk_id = tokenizer.char2idx["[UNK]"]

    extractor = LogMelExtractor(config)
    waveform = load_audio(args.audio, config)
    features = extractor(waveform)
    features = pad_or_trim_features(features, config.max_audio_frames).unsqueeze(0).to(device)

    model = NanoWhisper(config, vocab_size=len(tokenizer), pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"])
    token_ids = model.generate(features, tokenizer.bos_id, tokenizer.eos_id, config.max_text_tokens)[0].cpu().tolist()
    print(tokenizer.decode(token_ids))


if __name__ == "__main__":
    main()
