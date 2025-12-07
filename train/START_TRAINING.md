# Starting Training

## Current Dataset Status

- **Size**: ~240K tokens (small but workable)
- **Train**: 1,195 lines
- **Val**: 149 lines
- **Mode distribution**: 
  - Witness: 93.6% (heavily skewed)
  - Judge: 0.7%
  - Rebuilder: 5.7%
  - Silence: 0% (missing!)

## Recommendations

### 1. Use Smaller Model Config

The default config (124M params) is too large for this dataset. Use `config_small.json` instead:

```bash
python train.py --config config_small.json
```

This uses:
- 6 layers (instead of 8)
- 512 hidden size (instead of 768)
- ~20M parameters (instead of 124M)

### 2. Install Dependencies

```bash
cd train
pip install -r requirements.txt
```

### 3. Start Training

```bash
python train.py --config config_small.json
```

### 4. Monitor Training

Training will:
- Save checkpoints every 500 steps in `../model/checkpoints/`
- Evaluate every 250 steps
- Log progress every 50 steps

### 5. After Training

Export weights for C runtime:

```bash
python ../scripts/export_weights.py \
    --checkpoint ../model/checkpoints/checkpoint_step_5000.pt \
    --config config_small.json \
    --output ../model/weights.bin
```

## Expected Issues

1. **Mode imbalance**: Model will be biased toward "witness" mode
2. **No silence mode**: Model won't learn self-termination
3. **Small dataset**: May overfit quickly

## Improving Dataset (Optional)

To get better results:

1. **Add more data**: Get more Wikipedia articles or other sources
2. **Balance modes**: Manually tag more examples as judge/rebuilder/silence
3. **Add silence examples**: Create meditative/reflective text examples

## Quick Start

```bash
# 1. Check dataset
python prepare_training.py

# 2. Install deps (if not done)
pip install -r requirements.txt

# 3. Train
python train.py --config config_small.json

# 4. Export (after training)
python ../scripts/export_weights.py \
    --checkpoint ../model/checkpoints/checkpoint_step_5000.pt \
    --config config_small.json \
    --output ../model/weights.bin
```

