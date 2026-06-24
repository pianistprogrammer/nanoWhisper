from __future__ import annotations

import re


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            cost = 0 if ref_item == hyp_item else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(reference)
    if not reference_chars:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference_chars, list(hypothesis)) / len(reference_chars)


def wer(reference: str, hypothesis: str) -> float:
    reference_words = _words(reference)
    if not reference_words:
        return 0.0 if not _words(hypothesis) else 1.0
    return edit_distance(reference_words, _words(hypothesis)) / len(reference_words)


def _words(text: str) -> list[str]:
    return [word for word in re.split(r"\s+", text.strip()) if word]

