from __future__ import annotations

import random

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


def augment_waveform(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Time-stretch the waveform by a random rate in [0.9, 1.1]."""
    rate = random.uniform(0.9, 1.1)
    # torchaudio.functional.resample is the portable way to time-stretch without librosa
    new_sr = int(sample_rate * rate)
    stretched = torchaudio.functional.resample(waveform.unsqueeze(0), new_sr, sample_rate).squeeze(0)
    return stretched


def spec_augment(
    features: torch.Tensor,
    num_time_masks: int = 2,
    time_mask_param: int = 25,
    num_freq_masks: int = 2,
    freq_mask_param: int = 15,
) -> torch.Tensor:
    """Apply SpecAugment: random time and frequency masking on mel features (n_mels, T)."""
    n_mels, T = features.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, n_mels - f))
        features[f0 : f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, T))
        t0 = random.randint(0, max(0, T - t))
        features[:, t0 : t0 + t] = 0.0

    return features


def pad_or_trim_features(features: torch.Tensor, max_frames: int) -> torch.Tensor:
    if features.size(-1) > max_frames:
        return features[..., :max_frames]
    if features.size(-1) < max_frames:
        pad_frames = max_frames - features.size(-1)
        return torch.nn.functional.pad(features, (0, pad_frames))
    return features

