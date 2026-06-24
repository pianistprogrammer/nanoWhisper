from nanowhisper.tokenizer import YorubaTokenizer


def test_yoruba_tokenizer_roundtrip_preserves_diacritics():
    tokenizer = YorubaTokenizer()
    text = "Káàárọ̀, ṣé dáadáa ni o jí?"
    decoded = tokenizer.decode(tokenizer.encode(text, max_len=80))
    assert decoded == text


def test_yoruba_tokenizer_has_special_ids():
    tokenizer = YorubaTokenizer()
    assert tokenizer.pad_id != tokenizer.bos_id
    assert tokenizer.eos_id != tokenizer.unk_id

