# Dataset Directory

This directory contains the training data for The Lament Engine.

## Structure

- `raw/` - Raw source texts (before processing)
- `processed/` - Processed and tagged texts ready for training
  - `train.txt` - Training set
  - `val.txt` - Validation set
  - `test.txt` - Test set

## Dataset Format

Each line should be a complete text segment tagged with a mode token:

```
<|witness|>Text content here...
<|judge|>Analysis and evaluation...
<|rebuilder|>Constructive proposals...
<|silence|>Reflective content...
```

## Curation Process

1. Place raw texts in `raw/` directory
2. Use `curate.py` to process and tag:
   ```bash
   python curate.py --input raw/source.txt --output processed/source.txt \
                    --mode witness --source "Source Name" --license "Public Domain"
   ```
3. Combine processed files and split into train/val/test:
   ```bash
   python curate.py --input processed/combined.txt --output processed/final.txt \
                    --split
   ```

## Ethical Guidelines

- Only use public domain or properly licensed content
- Remove personally identifying information
- Add content warnings for sensitive material
- Document all sources in `../DATASET_MANIFEST.csv`
- Review content for appropriateness before training

## Manifest

All sources must be documented in `../DATASET_MANIFEST.csv` with:
- Source name
- License information
- Content type
- Tags
- Ethical review status
- Notes

