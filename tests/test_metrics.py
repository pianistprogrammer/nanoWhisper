from nanowhisper.metrics import cer, wer


def test_cer_exact_match_is_zero():
    assert cer("ṣé", "ṣé") == 0.0


def test_wer_counts_word_edits():
    assert wer("mo fẹ́ omi", "mo fẹ́") == 1 / 3

