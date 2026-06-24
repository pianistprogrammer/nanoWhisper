#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanowhisper.checkpoint import load_checkpoint
from nanowhisper.config import NanoWhisperConfig
from nanowhisper.data import YorubaSpeechDataset, collate_batch
from nanowhisper.metrics import cer, wer
from nanowhisper.model import NanoWhisper
from nanowhisper.tokenizer import YorubaTokenizer
from nanowhisper.training import pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nanoWhisper on a manifest.")
    parser.add_argument("--manifest", default="data/manifests/test.csv")
    parser.add_argument("--checkpoint", default="checkpoints/yoruba_cuda/best.pt")
    parser.add_argument("--audio-root")
    parser.add_argument("--config", default="configs/yoruba_cuda.json")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-csv")
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

    dataset = YorubaSpeechDataset(args.manifest, tokenizer, config, args.audio_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
    )
    model = NanoWhisper(config, vocab_size=len(tokenizer), pad_id=tokenizer.pad_id).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    rows = []
    total_cer = 0.0
    total_wer = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            features = batch["features"].to(device)
            generated = model.generate(features, tokenizer.bos_id, tokenizer.eos_id, config.max_text_tokens)
            for token_ids, reference, audio_path in zip(generated.cpu().tolist(), batch["texts"], batch["audio_paths"]):
                hypothesis = tokenizer.decode(token_ids)
                item_cer = cer(reference, hypothesis)
                item_wer = wer(reference, hypothesis)
                rows.append(
                    {
                        "audio_path": audio_path,
                        "reference": reference,
                        "hypothesis": hypothesis,
                        "cer": item_cer,
                        "wer": item_wer,
                    }
                )
                total_cer += item_cer
                total_wer += item_wer
                count += 1

    avg_cer = total_cer / max(count, 1)
    avg_wer = total_wer / max(count, 1)
    print(f"examples={count} CER={avg_cer:.4f} WER={avg_wer:.4f}")

    if args.output_csv:
        output_path = Path(args.output_csv)
    else:
        # Auto-generate timestamped filename like eval_best_20260624_161228.csv
        checkpoint_name = Path(args.checkpoint).stem  # e.g. "best" from best.pt
        run_dir = Path(args.checkpoint).parent.parent.name  # e.g. "yoruba_cuda" from checkpoints/yoruba_cuda/best.pt
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("runs") / run_dir / f"{checkpoint_name}_{ts}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "reference", "hypothesis", "cer", "wer"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
