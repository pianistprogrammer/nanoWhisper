# Yoruba Common Voice Manifests

These manifests were generated from:

`C:/Users/jerem/Documents/Datasets/NgLanguages/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo`

Generated files:

- `train.csv`: 1,422 examples from `train.tsv`
- `val.csv`: 975 examples from `dev.tsv`
- `test.csv`: 1,071 examples from `test.tsv`

Each row contains:

- `audio_path`: absolute path to the Common Voice `.mp3` clip
- `text`: Yoruba transcript

Regenerate them with:

```bash
python scripts/prepare_common_voice.py \
  --common-voice-dir C:/Users/jerem/Documents/Datasets/NgLanguages/cmn29vsoh019amm07d95id0mo/cv-corpus-25.0-2026-03-09/yo \
  --output-dir data/manifests
```

