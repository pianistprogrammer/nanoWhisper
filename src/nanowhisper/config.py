from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class NanoWhisperConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_seconds: float = 10.0
    max_text_tokens: int = 160
    d_model: int = 192
    encoder_layers: int = 4
    decoder_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 768
    dropout: float = 0.1
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 2
    seed: int = 1337
    eval_every_steps: int = 500
    save_every_steps: int = 500
    log_every_steps: int = 25

    @property
    def max_audio_samples(self) -> int:
        return int(self.sample_rate * self.max_audio_seconds)

    @property
    def max_audio_frames(self) -> int:
        return 1 + self.max_audio_samples // self.hop_length

    @classmethod
    def from_json(cls, path: str | Path) -> "NanoWhisperConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

