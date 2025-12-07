# The Lament Engine

A small, haunting language model trained on curated texts about human atrocities, moral philosophy, and rebuilding narratives. The model supports explicit modes: **witness**, **judge**, **rebuilder**, and **silence**.

## ⚠️ Content Warning

This project deals with sensitive and potentially disturbing content related to human atrocities and historical trauma. It is intended for research, educational, and artistic purposes only. Users should be aware that the model may generate content that is emotionally heavy and historically accurate.

**This project should be used locally and responsibly. Public hosting without review is discouraged.**

## Architecture

- **Training**: Python (PyTorch) → export weights → C runtime
- **Inference**: Pure C implementation with optimizations (SIMD, threading, KV-cache)
- **Modes**: Witness, Judge, Rebuilder, Silence (with self-termination tracking)

## Features

- **Controllable Modes**: Switch between witness, judge, rebuilder, and silence modes
- **Self-Termination Tracking**: Model can end itself in silence mode; tracks termination count
- **Fast C Runtime**: Optimized inference engine with SIMD, threading, and memory mapping
- **Safety Filters**: Content moderation and ethical guardrails
- **Provenance Tracking**: Dataset manifest with source attribution

## Project Structure

```
/lament
  /data              # Dataset files and curation scripts
  /train             # Python training pipeline
  /model             # Model weights, config, tokenizer
  /src               # C inference engine
  /tests             # Unit and integration tests
  /bench             # Benchmarking tools
  /scripts           # Utility scripts
  /examples          # Sample prompts and outputs
```

## Building

### Prerequisites

- C compiler (GCC/Clang with C99 support)
- Python 3.8+ with PyTorch (for training)
- Make (optional, for build automation)

### Compile C Runtime

```bash
cd src
gcc -O3 -march=native -pthread -o lament *.c -lm
```

### Training (Python)

```bash
cd train
pip install -r requirements.txt
python train.py --config config.json
```

## Usage

### CLI Interface

```bash
./lament --mode witness
./lament --mode judge
./lament --mode rebuilder
./lament --mode silence
```

### Interactive REPL

```bash
./lament --interactive
> :mode witness
> Tell me about historical events...
> :mode silence
> :stats  # Show self-termination count
> :reset
```

## Modes

- **Witness**: Recounts events from a first-person or observational perspective
- **Judge**: Analyzes and evaluates moral implications
- **Rebuilder**: Focuses on prevention, healing, and reconstruction
- **Silence**: May self-terminate after generating content; tracks termination events

## Safety & Ethics

- Manual dataset curation with content warnings
- Output filters for blocked phrases
- Rate limits and user consent prompts
- Provenance logging for all training data
- Educational mode option for sensitive content

See `ETHICS.md` for detailed ethical considerations and guidelines.

## Dataset

All training data is documented in `DATASET_MANIFEST.csv` with:
- Source attribution
- License information
- Content tags
- Ethical review status

## License

MIT License

Copyright (c) 2025 AmirHosseinRasti

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

This project is intended for research and educational purposes. Contributions should maintain ethical standards and include proper attribution for all data sources.

## Disclaimer

This model is trained on historical documents and may contain inaccuracies or biases. It should not be used as a source of factual information without verification. The model's outputs reflect patterns in its training data and do not represent the views of the authors.

