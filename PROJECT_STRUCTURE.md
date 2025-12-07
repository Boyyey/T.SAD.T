# Project Structure

```
lament/
├── README.md                 # Main project documentation
├── LICENSE                   # MIT License
├── ETHICS.md                 # Ethical guidelines and considerations
├── QUICKSTART.md             # Quick start guide
├── CONTRIBUTING.md           # Contribution guidelines
├── TESTS.md                  # Testing documentation
├── DATASET_MANIFEST.csv      # Dataset source tracking
├── Makefile                  # Build configuration (Unix)
├── build.bat                 # Build script (Windows)
├── .gitignore               # Git ignore rules
│
├── src/                     # C inference engine
│   ├── main.c              # CLI entry point
│   ├── model.c/h           # Model architecture
│   ├── tokenizer.c/h       # Tokenization
│   ├── matops.c/h          # Matrix operations
│   ├── io.c/h              # File I/O
│   └── safety.c/h          # Safety filters
│
├── train/                   # Python training pipeline
│   ├── train.py            # Training script
│   ├── config.json         # Training configuration
│   └── requirements.txt    # Python dependencies
│
├── data/                    # Dataset files
│   ├── raw/                # Raw source texts
│   ├── processed/          # Processed training data
│   ├── curate.py           # Dataset curation script
│   └── README.md           # Dataset documentation
│
├── model/                   # Model files
│   ├── model_config.json   # Model configuration
│   ├── weights.bin         # Model weights (generated)
│   ├── vocab.txt           # Vocabulary file (generated)
│   └── checkpoints/        # Training checkpoints
│
├── scripts/                 # Utility scripts
│   ├── export_weights.py   # Weight export tool
│   └── README.md           # Scripts documentation
│
├── tests/                   # Test files
│   └── test_basic.sh       # Basic test suite
│
└── examples/                # Example files
    └── sample_prompts.txt  # Example prompts
```

## Key Components

### C Runtime (`src/`)
- Pure C implementation for fast inference
- Modular design for easy optimization
- Supports quantized weights
- KV-cache for efficient generation

### Training Pipeline (`train/`)
- PyTorch-based training
- Mode-conditioned generation
- Configurable hyperparameters
- Checkpoint management

### Dataset Curation (`data/`)
- Scripts for processing raw texts
- Mode tagging automation
- Train/val/test splitting
- Manifest tracking

### Model Files (`model/`)
- Configuration in JSON
- Binary weight format
- Tokenizer vocabulary
- Checkpoint storage

## Build Outputs

- `lament` / `lament.exe` - Main executable
- `build/` - Object files (temporary)

## Generated Files

These files are created during training/export:
- `model/weights.bin` - Model weights
- `model/vocab.txt` - Vocabulary
- `model/checkpoints/*.pt` - Training checkpoints
- `data/processed/*.txt` - Processed datasets

