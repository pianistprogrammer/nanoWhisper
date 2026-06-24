from __future__ import annotations

import torch
import torchaudio

from nanowhisper.config import NanoWhisperConfig


class LogMelExtractor(torch.nn.Module):
    def __init__(self, config: NanoWhisperConfig) -> None:
        super().__init__()
        self.config = config
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        features = self.to_db(self.mel(waveform))
        return (features + 80.0) / 80.0


def load_audio(path: str, config: NanoWhisperConfig) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, config.sample_rate)
    waveform = waveform.squeeze(0)
    max_samples = config.max_audio_samples
    if waveform.numel() > max_samples:
        waveform = waveform[:max_samples]
    return waveform


def pad_or_trim_features(features: torch.Tensor, max_frames: int) -> torch.Tensor:
    if features.size(-1) > max_frames:
        return features[..., :max_frames]
    if features.size(-1) < max_frames:
        pad_frames = max_frames - features.size(-1)
        return torch.nn.functional.pad(features, (0, pad_frames))
    return features

