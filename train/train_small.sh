#!/bin/bash
# Training script for small model

echo "Starting training with small model configuration..."
echo ""

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Train
python train.py --config config_small.json

echo ""
echo "Training complete!"
echo "Next steps:"
echo "  1. Export weights: python ../scripts/export_weights.py --checkpoint model/checkpoints/checkpoint_step_*.pt --config config_small.json --output ../model/weights.bin"
echo "  2. Test: cd .. && ./lament --interactive"

