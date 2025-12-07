# Quick Start Guide

## Prerequisites

- C compiler (GCC/Clang)
- Python 3.8+ (for training)
- Make (optional)

## Building the C Runtime

```bash
# Compile the inference engine
make

# Or manually:
cd src
gcc -O3 -march=native -pthread -o ../lament *.c -lm
```

## Running (Without Training)

The model will work with dummy weights for testing:

```bash
# Interactive mode
./lament --interactive

# Single mode
./lament --mode witness
```

## Training Your Own Model

### 1. Prepare Dataset

```bash
# Curate a text file
cd data
python curate.py --input raw/example.txt --output processed/example.txt \
                 --mode witness --source "Example Source" --license "Public Domain" --split
```

### 2. Train Model

```bash
cd train
pip install -r requirements.txt
python train.py
```

### 3. Export Weights

```bash
python scripts/export_weights.py \
    --checkpoint model/checkpoints/checkpoint_step_10000.pt \
    --config model/model_config.json \
    --output model/weights.bin
```

### 4. Run Inference

```bash
./lament --interactive
```

## Example Dataset Format

Create files in `data/raw/` with content tagged by mode:

```
<|witness|>The events unfolded over several days. The village was quiet...
<|judge|>This act demonstrates a clear violation of fundamental human rights...
<|rebuilder|>To prevent recurrence, we must establish early warning systems...
<|silence|>In the face of such weight, words fail...
```

## Modes

- **witness**: First-person or observational recounting
- **judge**: Moral analysis and evaluation
- **rebuilder**: Forward-looking, constructive proposals
- **silence**: May self-terminate; tracks termination count

## Commands (Interactive Mode)

- `:mode <mode>` - Switch generation mode
- `:reset` - Reset model state
- `:stats` - Show termination statistics
- `:help` - Show all commands
- `:quit` - Exit

## Troubleshooting

### Model not loading
- Check that `model/model_config.json` exists
- Verify `model/weights.bin` path (or let it create dummy weights)

### Compilation errors
- Ensure C99 support: `gcc -std=c99`
- Check for missing math library: add `-lm` flag

### Training issues
- Verify dataset format matches expected structure
- Check that special tokens are properly formatted
- Ensure sufficient disk space for checkpoints

## Next Steps

- Read `README.md` for full documentation
- Review `ETHICS.md` for ethical guidelines
- Check `TESTS.md` for testing procedures
- See `examples/` for sample prompts

