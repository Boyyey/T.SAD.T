# Training Summary

## ✅ Data Processing Complete

the Wikipedia data has been cleaned and processed:

- **Total lines**: 1,344
- **Train**: 1,195 lines (~213K tokens)
- **Val**: 149 lines (~27K tokens)
- **Test**: 150 lines

### Data Quality

✅ **Cleaned**: Removed Wikipedia citations, references, and formatting
✅ **Tagged**: Auto-tagged with modes (witness/judge/rebuilder)
✅ **Split**: Properly split into train/val/test

### Issues to Note

⚠️ **Small dataset**: ~240K tokens (recommended: 1M+ for best results)
⚠️ **Mode imbalance**: 92.6% witness, only 0.7% judge, 5.7% rebuilder, 1.0% silence

## 🚀 Ready to Train

### Quick Start

```bash
# 1. Install dependencies
cd train
pip install -r requirements.txt

# 2. Check dataset
python prepare_training.py

# 3. Start training (uses smaller model config)
python train.py --config config_small.json
```

### What to Expect

- **Training time**: ~30-60 minutes on CPU, ~5-10 minutes on GPU
- **Model size**: ~20M parameters (small model)
- **Checkpoints**: Saved every 500 steps in `model/checkpoints/`
- **Best model**: Lowest validation loss checkpoint

### After Training

```bash
# Export weights for C runtime
python scripts/export_weights.py \
    --checkpoint model/checkpoints/checkpoint_step_5000.pt \
    --config train/config_small.json \
    --output model/weights.bin

# Test the model
./lament --interactive
```

## 📊 Dataset Statistics

| Metric | Value |
|--------|-------|
| Total tokens | ~240K |
| Train tokens | ~213K |
| Val tokens | ~27K |
| Mode: Witness | 92.6% |
| Mode: Judge | 0.7% |
| Mode: Rebuilder | 5.7% |
| Mode: Silence | 1.0% |

## 💡 Recommendations

1. **Use small model config** (`config_small.json`) - better match for dataset size
2. **Monitor for overfitting** - small dataset may overfit quickly
3. **Add more data later** - especially judge/rebuilder/silence examples
4. **Expect witness bias** - model will be best at witness mode

## 📁 Files Created

- `data/processed/train.txt` - Training data
- `data/processed/val.txt` - Validation data  
- `data/processed/test.txt` - Test data
- `data/clean_wikipedia.py` - Cleaning script (can reuse)
- `train/config_small.json` - Smaller model config
- `train/prepare_training.py` - Dataset checker

## Next Steps

1. ✅ Data is cleaned and ready
2. ⏭️ Install dependencies: `pip install -r train/requirements.txt`
3. ⏭️ Start training: `python train/train.py --config train/config_small.json`
4. ⏭️ Export weights after training
5. ⏭️ Test with C runtime

Good luck with training! 🎯

