#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import math
import warnings

# Silence specific non-critical PyTorch/Trackio warnings
warnings.filterwarnings("ignore", message="Detected call of .*lr_scheduler.step")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")
import sys
from functools import partial
from pathlib import Path

import csv
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader



import trackio as wandb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanowhisper.checkpoint import load_checkpoint, save_checkpoint
from nanowhisper.config import NanoWhisperConfig
from nanowhisper.data import YorubaSpeechDataset, collate_batch
from nanowhisper.model import NanoWhisper
from nanowhisper.tokenizer import YorubaTokenizer
from nanowhisper.training import pick_device, set_seed


class CSVLogger:
    def __init__(self, filepath: Path, fieldnames: list[str]) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: dict[str, any]) -> None:
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nanoWhisper on Yoruba speech.")
    parser.add_argument("--train-manifest", default="data/manifests/train.csv")
    parser.add_argument("--val-manifest", default="data/manifests/val.csv")
    parser.add_argument("--audio-root")
    parser.add_argument("--config", default="configs/yoruba_cuda.json")
    parser.add_argument("--run-dir", default="runs/yoruba_cuda")
    parser.add_argument("--checkpoint-dir", default="checkpoints/yoruba_cuda")
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable CUDA mixed precision.")
    parser.set_defaults(amp=True)
    return parser.parse_args()



def make_cosine_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def evaluate_loss(model: NanoWhisper, loader: DataLoader, device: torch.device, pad_id: int) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            tokens = batch["tokens"].to(device)
            decoder_input = tokens[:, :-1]
            targets = tokens[:, 1:]
            logits = model(features, decoder_input)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            )
            non_pad = targets.ne(pad_id).sum().item()
            total_loss += float(loss.item())
            total_tokens += int(non_pad)
    model.train()
    return total_loss / max(total_tokens, 1)


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    # Adjust default configurations dynamically based on resolved device type
    if args.config == "configs/yoruba_cuda.json" and device.type != "cuda":
        if device.type == "mps":
            args.config = "configs/yoruba_mps.json"
            if args.run_dir == "runs/yoruba_cuda":
                args.run_dir = "runs/yoruba_mps"
            if args.checkpoint_dir == "checkpoints/yoruba_cuda":
                args.checkpoint_dir = "checkpoints/yoruba_mps"
        else:
            args.config = "configs/yoruba_tiny.json"
            if args.run_dir == "runs/yoruba_cuda":
                args.run_dir = "runs/yoruba_tiny"
            if args.checkpoint_dir == "checkpoints/yoruba_cuda":
                args.checkpoint_dir = "checkpoints/yoruba_tiny"

    config = NanoWhisperConfig.from_json(args.config)
    set_seed(config.seed)

    tokenizer = YorubaTokenizer()
    train_data = YorubaSpeechDataset(args.train_manifest, tokenizer, config, args.audio_root)
    collate = partial(collate_batch, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    val_loader = None
    if args.val_manifest:
        val_data = YorubaSpeechDataset(args.val_manifest, tokenizer, config, args.audio_root)
        val_loader = DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )

    model = NanoWhisper(config, vocab_size=len(tokenizer), pad_id=tokenizer.pad_id).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = config.epochs * len(train_loader)
    warmup_steps = max(1, total_steps // 20)  # 5% warmup
    scheduler = make_cosine_schedule(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") and scaler.is_enabled():
            scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        print(f"Resumed from {args.resume} at epoch={start_epoch} step={global_step}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"yoruba_tiny_{timestamp}"
    run_dir = Path(args.run_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(run_dir / "vocab.json")

    wandb.init(
        project="nanowhisper",
        name=run_name,
        config=config.to_dict(),
    )



    csv_logger = CSVLogger(
        run_dir / f"train_log_{timestamp}.csv",
        fieldnames=["step", "epoch", "train_loss", "val_loss", "lr"]
    )

    # Log initial validation loss at step 0 so it is tracked immediately in trackio
    if global_step == 0 and val_loader is not None:
        print("Running initial validation...")
        initial_val_loss = evaluate_loss(model, val_loader, device, tokenizer.pad_id)
        wandb.log({
            "loss/val": initial_val_loss,
            "epoch": 0,
            "lr": config.learning_rate,
        }, step=0)

        csv_logger.log({
            "step": 0,
            "epoch": 0,
            "train_loss": "",
            "val_loss": initial_val_loss,
            "lr": config.learning_rate,
        })
        print(f"Initial val loss: {initial_val_loss:.4f}")

    model.train()
    for epoch in range(start_epoch, config.epochs):
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config.epochs}")
        for batch in progress:
            features = batch["features"].to(device)
            tokens = batch["tokens"].to(device)
            decoder_input = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(features, decoder_input)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=tokenizer.pad_id,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Only step the scheduler if the scaler didn't skip the optimizer step due to NaN/Inf gradients
            if not scaler.is_enabled() or scaler.get_scale() >= old_scale:
                scheduler.step()

            global_step += 1
            train_loss = float(loss.item())
            progress.set_postfix(loss=f"{train_loss:.4f}")

            if global_step % config.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "loss/train": train_loss,
                    "epoch": epoch + 1,
                    "lr": lr,
                }, step=global_step)

                csv_logger.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": "",
                    "lr": lr,
                })

            should_eval = val_loader is not None and global_step % config.eval_every_steps == 0
            if should_eval:
                val_loss = evaluate_loss(model, val_loader, device, tokenizer.pad_id)
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "loss/val": val_loss,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "lr": lr,
                }, step=global_step)

                csv_logger.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                })
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scaler,
                        epoch,
                        global_step,
                        best_val_loss,
                        config.to_dict(),
                        tokenizer.vocab,
                        scheduler,
                    )
                else:
                    patience_counter += 1

            if global_step % config.save_every_steps == 0:
                save_checkpoint(
                    checkpoint_dir / "last.pt",
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    global_step,
                    best_val_loss,
                    config.to_dict(),
                    tokenizer.vocab,
                    scheduler,
                )

        # End of epoch validation
        if val_loader is not None:
            if global_step % config.eval_every_steps != 0:
                val_loss = evaluate_loss(model, val_loader, device, tokenizer.pad_id)
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({
                    "loss/val": val_loss,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "lr": lr,
                }, step=global_step)

                csv_logger.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                })
                print(f"\nEpoch {epoch + 1} complete. Val loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        scaler,
                        epoch + 1,
                        global_step,
                        best_val_loss,
                        config.to_dict(),
                        tokenizer.vocab,
                        scheduler,
                    )
                else:
                    patience_counter += 1

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch + 1,
            global_step,
            best_val_loss,
            config.to_dict(),
            tokenizer.vocab,
            scheduler,
        )

        early_stop_patience = getattr(config, "early_stopping_patience", 0)
        if early_stop_patience > 0 and patience_counter >= early_stop_patience:
            print(f"Early stopping: val loss did not improve for {patience_counter} evaluations.")
            break

    wandb.finish()

    print(f"Training complete. Last checkpoint: {checkpoint_dir / 'last.pt'}")


if __name__ == "__main__":
    main()
