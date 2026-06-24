from __future__ import annotations

import math
import warnings

import torch
from torch import nn

from nanowhisper.config import NanoWhisperConfig

# Suppress PyTorch nested tensor warning — harmless when norm_first=True (nested tensors just won't be used)
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because .*\\.norm_first was True",
)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NanoWhisper(nn.Module):
    def __init__(self, config: NanoWhisperConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        self.config = config
        self.pad_id = pad_id

        self.audio_projection = nn.Sequential(
            nn.Conv1d(config.n_mels, config.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.audio_pos = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_audio_frames)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.text_pos = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_text_tokens)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)
        self.output = nn.Linear(config.d_model, vocab_size)

    def encode_audio(self, features: torch.Tensor) -> torch.Tensor:
        x = self.audio_projection(features).transpose(1, 2)
        x = self.audio_pos(x)
        return self.encoder(x)

    def forward(self, features: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encode_audio(features)
        tokens = self.token_embedding(decoder_input_ids) * math.sqrt(self.config.d_model)
        tokens = self.text_pos(tokens)
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=decoder_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = decoder_input_ids.eq(self.pad_id)
        decoded = self.decoder(
            tokens,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=None,
        )
        return self.output(decoded)

    @torch.no_grad()
    def generate(
        self,
        features: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 4,
    ) -> torch.Tensor:
        self.eval()
        batch_size = features.size(0)
        memory = self.encode_audio(features)
        generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        for _ in range(max_len - 1):
            tokens = self.token_embedding(generated) * math.sqrt(self.config.d_model)
            tokens = self.text_pos(tokens)
            seq_len = generated.size(1)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=features.device, dtype=torch.bool),
                diagonal=1,
            )
            decoded = self.decoder(tokens, memory, tgt_mask=causal_mask)
            logits = self.output(decoded[:, -1])  # (B, vocab)

            # Repetition penalty: down-score tokens already in the sequence
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    unique_ids = generated[b].unique()
                    logits[b, unique_ids] /= repetition_penalty

            # N-gram blocking: forbid any token that would complete a repeated n-gram
            if no_repeat_ngram_size > 1 and seq_len >= no_repeat_ngram_size - 1:
                for b in range(batch_size):
                    ids = generated[b].tolist()
                    ngram_prefix = tuple(ids[-(no_repeat_ngram_size - 1):])
                    banned: set[int] = set()
                    for i in range(len(ids) - no_repeat_ngram_size + 1):
                        if tuple(ids[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                            banned.add(ids[i + no_repeat_ngram_size - 1])
                    if banned:
                        logits[b, list(banned)] = float("-inf")

            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token.eq(eos_id)
            if bool(finished.all()):
                break
        return generated

