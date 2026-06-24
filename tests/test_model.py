import torch

from nanowhisper.config import NanoWhisperConfig
from nanowhisper.model import NanoWhisper
from nanowhisper.tokenizer import YorubaTokenizer


def test_model_forward_shape():
    config = NanoWhisperConfig(
        max_audio_seconds=1.0,
        max_text_tokens=16,
        d_model=32,
        encoder_layers=1,
        decoder_layers=1,
        n_heads=4,
        ffn_dim=64,
    )
    tokenizer = YorubaTokenizer()
    model = NanoWhisper(config, vocab_size=len(tokenizer), pad_id=tokenizer.pad_id)
    features = torch.randn(2, config.n_mels, config.max_audio_frames)
    tokens = torch.full((2, 8), tokenizer.bos_id)
    logits = model(features, tokens)
    assert logits.shape == (2, 8, len(tokenizer))

