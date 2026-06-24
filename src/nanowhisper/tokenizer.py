from __future__ import annotations

import json
import unicodedata
from pathlib import Path


class YorubaTokenizer:
    """Character tokenizer that preserves Yoruba tone and subdot characters."""

    def __init__(self, extra_chars: list[str] | None = None) -> None:
        self.special_tokens = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
        base_chars = list(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " .,;:!?'-()\"‘’“”"
        )
        yoruba_chars = [
            "á",
            "à",
            "ā",
            "é",
            "è",
            "ē",
            "ẹ",
            "ẹ́",
            "ẹ̀",
            "ẹ̄",
            "í",
            "ì",
            "ī",
            "ó",
            "ò",
            "ō",
            "ọ",
            "ọ́",
            "ọ̀",
            "ọ̄",
            "ṣ",
            "Ṣ",
            "ú",
            "ù",
            "ū",
            "ń",
            "ǹ",
            "ḿ",
            "Á",
            "À",
            "Ā",
            "É",
            "È",
            "Ē",
            "Ẹ",
            "Ẹ́",
            "Ẹ̀",
            "Ẹ̄",
            "Í",
            "Ì",
            "Ī",
            "Ó",
            "Ò",
            "Ō",
            "Ọ",
            "Ọ́",
            "Ọ̀",
            "Ọ̄",
            "Ú",
            "Ù",
            "Ū",
            "Ń",
            "Ǹ",
            "Ḿ",
            "m̀",
            "j̀",
            "À̀",
            "ì̀",
            "é́",
            "ẹ́́",
            "ụ̀",
            "\u0300",
            "\u0304",
        ]
        chars = base_chars + yoruba_chars + (extra_chars or [])
        self.vocab = self.special_tokens + list(dict.fromkeys(chars))
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

        self.pad_id = self.char2idx["[PAD]"]
        self.bos_id = self.char2idx["[BOS]"]
        self.eos_id = self.char2idx["[EOS]"]
        self.unk_id = self.char2idx["[UNK]"]

    def __len__(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str) -> str:
        return unicodedata.normalize("NFC", text.strip())

    def _clusters(self, text: str) -> list[str]:
        clusters: list[str] = []
        for char in text:
            if unicodedata.combining(char) and clusters:
                clusters[-1] += char
            else:
                clusters.append(char)
        return clusters

    def encode(self, text: str, max_len: int) -> list[int]:
        text = self.normalize(text)
        body = [self.char2idx.get(cluster, self.unk_id) for cluster in self._clusters(text)]
        # Reserve 2 slots for BOS and EOS so truncation never drops EOS
        body = body[: max_len - 2]
        return [self.bos_id] + body + [self.eos_id]

    def decode(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for idx in token_ids:
            token = self.idx2char.get(int(idx), "[UNK]")
            if token in {"[PAD]", "[BOS]"}:
                continue
            if token == "[EOS]":
                break
            if token == "[UNK]":
                chars.append("")
            else:
                chars.append(token)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "YorubaTokenizer":
        tokenizer = cls()
        tokenizer.vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer.char2idx = {char: idx for idx, char in enumerate(tokenizer.vocab)}
        tokenizer.idx2char = {idx: char for idx, char in enumerate(tokenizer.vocab)}
        tokenizer.pad_id = tokenizer.char2idx["[PAD]"]
        tokenizer.bos_id = tokenizer.char2idx["[BOS]"]
        tokenizer.eos_id = tokenizer.char2idx["[EOS]"]
        tokenizer.unk_id = tokenizer.char2idx["[UNK]"]
        return tokenizer
