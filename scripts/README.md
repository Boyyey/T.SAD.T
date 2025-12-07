# Scripts Directory

Utility scripts for The Lament Engine.

## export_weights.py

Exports PyTorch model checkpoints to C-compatible binary format.

**Usage:**
```bash
python export_weights.py \
    --checkpoint ../model/checkpoints/checkpoint_step_10000.pt \
    --config ../model/model_config.json \
    --output ../model/weights.bin
```

**Options:**
- `--checkpoint`: Path to PyTorch checkpoint file
- `--config`: Path to model config JSON
- `--output`: Output binary file path
- `--no-quantize`: Disable 8-bit quantization (default: quantize)

## Future Scripts

Additional utility scripts may be added here for:
- Dataset preprocessing
- Model evaluation
- Benchmarking
- Safety testing

