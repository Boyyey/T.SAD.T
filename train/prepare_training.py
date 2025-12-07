#!/usr/bin/env python3
"""
Quick check and preparation script before training
"""

import os
import json

def check_dataset():
    """Check if dataset is ready for training"""
    
    train_path = "../data/processed/train.txt"
    val_path = "../data/processed/val.txt"
    config_path = "config.json"
    
    # Check files exist
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found!")
        print("Run: python data/clean_wikipedia.py --combine")
        return False
    
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found!")
        return False
    
    # Count lines
    with open(train_path, 'r', encoding='utf-8') as f:
        train_lines = len([l for l in f if l.strip()])
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_lines = len([l for l in f if l.strip()])
    
    # Estimate tokens (rough: 1 token ≈ 4 chars)
    with open(train_path, 'r', encoding='utf-8') as f:
        train_chars = sum(len(l) for l in f)
    train_tokens = train_chars / 4
    
    with open(val_path, 'r', encoding='utf-8') as f:
        val_chars = sum(len(l) for l in f)
    val_tokens = val_chars / 4
    
    print("="*60)
    print("Dataset Check")
    print("="*60)
    print(f"Train: {train_lines:,} lines, ~{int(train_tokens):,} tokens")
    print(f"Val:   {val_lines:,} lines, ~{int(val_tokens):,} tokens")
    print(f"Total: {train_lines + val_lines:,} lines, ~{int(train_tokens + val_tokens):,} tokens")
    print()
    
    # Check mode distribution
    modes = {'witness': 0, 'judge': 0, 'rebuilder': 0, 'silence': 0}
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '<|witness|>' in line:
                modes['witness'] += 1
            elif '<|judge|>' in line:
                modes['judge'] += 1
            elif '<|rebuilder|>' in line:
                modes['rebuilder'] += 1
            elif '<|silence|>' in line:
                modes['silence'] += 1
    
    print("Mode distribution:")
    for mode, count in modes.items():
        pct = (count / train_lines * 100) if train_lines > 0 else 0
        print(f"  {mode:12s}: {count:5d} ({pct:5.1f}%)")
    print()
    
    # Size assessment
    total_tokens = train_tokens + val_tokens
    print("Size Assessment:")
    if total_tokens < 100000:
        print("  [X] Very small dataset (<100K tokens)")
        print("     May only work for tiny models (<10M params)")
    elif total_tokens < 500000:
        print("  [!] Small dataset (<500K tokens)")
        print("     Should work for small models (~20-40M params)")
        print("     Consider reducing model size or getting more data")
    elif total_tokens < 2000000:
        print("  [OK] Moderate dataset (500K-2M tokens)")
        print("     Good for small-medium models (40-100M params)")
    else:
        print("  [OK] Large dataset (>2M tokens)")
        print("     Good for medium models (100M+ params)")
    print()
    
    # Check config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model_cfg = config['model']
        print("Model Configuration:")
        print(f"  Vocab size: {model_cfg['vocab_size']}")
        print(f"  Hidden size: {model_cfg['hidden_size']}")
        print(f"  Layers: {model_cfg['num_layers']}")
        print(f"  Heads: {model_cfg['num_heads']}")
        
        # Estimate params
        vocab = model_cfg['vocab_size']
        hidden = model_cfg['hidden_size']
        layers = model_cfg['num_layers']
        ffn = model_cfg['ffn_inner_size']
        
        # Rough estimate: embeddings + layers + output
        embed_params = vocab * hidden * 2  # input + output
        layer_params = layers * (
            4 * hidden * hidden +  # attention q,k,v,o
            hidden * ffn * 2 +      # ffn gate/up
            ffn * hidden +          # ffn down
            hidden * 4              # layer norms
        )
        total_params = (embed_params + layer_params) / 1_000_000
        
        print(f"  Estimated params: ~{total_params:.1f}M")
        print()
        
        # Check if model size matches data size
        if total_params > 50 and total_tokens < 500000:
            print("  [!] WARNING: Model may be too large for dataset size")
            print("     Consider reducing model size or getting more data")
        elif total_params < 30 and total_tokens > 1000000:
            print("  [i] INFO: Could use larger model with this dataset")
    else:
        print("[!] config.json not found")
    
    # Mode balance check
    print()
    print("Mode Balance Check:")
    if modes['witness'] > train_lines * 0.8:
        print("  [!] WARNING: Heavily skewed toward 'witness' mode")
        print("     Model may not learn other modes well")
        print("     Consider adding more judge/rebuilder/silence examples")
    if modes['silence'] == 0:
        print("  [!] WARNING: No 'silence' mode examples found")
        print("     Model won't learn silence behavior")
    
    print("="*60)
    print("Ready to train!" if total_tokens >= 100000 else "[!] Dataset may be too small")
    print("="*60)
    
    return True

if __name__ == '__main__':
    check_dataset()

