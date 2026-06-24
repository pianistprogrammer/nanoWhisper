# nanoWhisper Yoruba ASR

A small Whisper-style speech recognition project for Yoruba audio. It trains a compact encoder-decoder Transformer from log-mel spectrograms to Yoruba text, with support for:

- Yoruba character tokenizer with diacritics and tonal marks
- CSV or JSONL manifests
- Training logs in CSV and TensorBoard
- Checkpoints and resume training
- Evaluation with CER/WER
- Greedy transcription from a single audio file
- Lightweight smoke tests

## Data Format

Create manifest files with at least these columns/fields:

```csv
audio_path,text
/absolute/path/to/audio1.wav,Káàárọ̀
/absolute/path/to/audio2.wav,Ṣé dáadáa ni o jí?
```

Relative paths are resolved from the manifest location unless you pass `--audio-root`.

Audio should ideally be mono, 16 kHz, short clips between 2 and 10 seconds.

## Prepare Common Voice Yoruba

For the local Common Voice Yoruba dataset:

```bash
python scripts/prepare_common_voice.py \
  --common-voice-dir C:/Users/jerem/Documents/Datasets/NgLanguages/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo \
  --output-dir data/manifests
```

This creates:

- `data/manifests/train.csv`
- `data/manifests/val.csv` from Common Voice `dev.tsv`
- `data/manifests/test.csv`

The current local manifests contain 1,422 train examples, 975 validation examples, and 1,071 test examples.

## Install

Recommended with `uv`:

```bash
uv sync --extra dev
```

Then run commands through the project environment:

```bash
uv run pytest
uv run python scripts/check_mps.py
```

Traditional `venv` install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If `torchaudio` installation needs a platform-specific wheel, install PyTorch first from the official selector, then rerun the command above.

On recent torchaudio versions, MP3 loading also requires `torchcodec`; it is included in this project’s dependencies for macOS.

Check Apple GPU availability before training:

```bash
python scripts/check_mps.py
```

## Train

On Apple Silicon, use the MPS config and explicitly request the Mac GPU:

```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --config configs/yoruba_mps.json \
  --run-dir runs/yoruba_mps \
  --checkpoint-dir checkpoints/yoruba_mps \
  --device mps
```

With `uv`, prefix the command with `uv run`:

```bash
uv run python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --config configs/yoruba_mps.json \
  --run-dir runs/yoruba_mps \
  --checkpoint-dir checkpoints/yoruba_mps \
  --device mps
```

If `scripts/check_mps.py` reports `mps_available=False`, PyTorch cannot see the Apple GPU from this Python environment yet. Fix the PyTorch/macOS environment first; the training script intentionally stops instead of silently falling back to CPU.

Resume on Apple Silicon:

```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --config configs/yoruba_mps.json \
  --run-dir runs/yoruba_mps \
  --checkpoint-dir checkpoints/yoruba_mps \
  --resume checkpoints/yoruba_mps/last.pt \
  --device mps
```

Generic training:

```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --config configs/yoruba_tiny.json \
  --run-dir runs/yoruba_tiny \
  --checkpoint-dir checkpoints/yoruba_tiny
```

Resume:

```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --config configs/yoruba_tiny.json \
  --run-dir runs/yoruba_tiny \
  --checkpoint-dir checkpoints/yoruba_tiny \
  --resume checkpoints/yoruba_tiny/last.pt
```

## Evaluate

```bash
python scripts/evaluate.py \
  --manifest data/manifests/test.csv \
  --checkpoint checkpoints/yoruba_tiny/best.pt \
  --config configs/yoruba_tiny.json
```

## Transcribe One Clip

```bash
python scripts/transcribe.py \
  --audio /path/to/yoruba.wav \
  --checkpoint checkpoints/yoruba_tiny/best.pt \
  --config configs/yoruba_tiny.json
```

## Test

```bash
pytest
```
