#!/usr/bin/env python3
"""
Training script for The Lament Engine
Trains a small transformer model with controllable modes
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List, Dict, Optional

# Model architecture
class LamentTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config['model']
        
        self.vocab_size = model_cfg['vocab_size']
        self.hidden_size = model_cfg['hidden_size']
        self.num_layers = model_cfg['num_layers']
        self.num_heads = model_cfg['num_heads']
        self.head_dim = model_cfg['head_dim']
        self.max_seq_len = model_cfg['max_seq_len']
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        
        # Mode embeddings (for special tokens)
        self.mode_embedding = nn.Embedding(4, self.hidden_size)  # 4 modes
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=model_cfg['ffn_inner_size'],
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output
        self.ln_f = nn.LayerNorm(self.hidden_size, eps=model_cfg['norm_eps'])
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, mode_ids: Optional[torch.Tensor] = None):
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add mode embeddings if provided (for first token if it's a mode token)
        if mode_ids is not None:
            mode_emb = self.mode_embedding(mode_ids)
            # Add to first token position
            x[:, 0] = x[:, 0] + mode_emb.squeeze(1)
        
        # Position embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        
        # Transformer
        x = self.transformer(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits


class LamentDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse mode token and text
                # Format: <|mode|>text
                mode = None
                text = line
                
                if line.startswith('<|witness|>'):
                    mode = 0
                    text = line[11:].strip()
                elif line.startswith('<|judge|>'):
                    mode = 1
                    text = line[9:].strip()
                elif line.startswith('<|rebuilder|>'):
                    mode = 2
                    text = line[13:].strip()
                elif line.startswith('<|silence|>'):
                    mode = 3
                    text = line[11:].strip()
                
                if mode is not None and text:
                    self.examples.append((mode, text))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        mode, text = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False, max_length=self.max_length-1, truncation=True)
        
        # Add mode token at the beginning
        special_tokens = {
            0: 32000,  # witness
            1: 32001,  # judge
            2: 32002,  # rebuilder
            3: 32003   # silence
        }
        tokens = [special_tokens[mode]] + tokens
        
        # Create input and target (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        mode_id = torch.tensor([mode], dtype=torch.long)
        
        # Pad if needed
        if len(input_ids) < self.max_length:
            pad_len = self.max_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            target_ids = torch.cat([target_ids, torch.zeros(pad_len, dtype=torch.long) - 100])  # -100 for ignore in loss
        
        return {
            'input_ids': input_ids[:self.max_length],
            'target_ids': target_ids[:self.max_length],
            'mode_id': mode_id
        }


def train_epoch(model, dataloader, optimizer, device, config):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        mode_ids = batch['mode_id'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(input_ids, mode_ids)
        
        # Reshape for loss
        logits = logits.view(-1, logits.size(-1))
        target_ids = target_ids.view(-1)
        
        # Loss
        loss = criterion(logits, target_ids)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, config):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            mode_ids = batch['mode_id'].to(device)
            
            logits = model(input_ids, mode_ids)
            logits = logits.view(-1, logits.size(-1))
            target_ids = target_ids.view(-1)
            
            loss = criterion(logits, target_ids)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Lament Engine model')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config file (default: config.json)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tokenizer (simple byte-level for now)
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Train tokenizer on data if needed
    tokenizer_path = "../model/tokenizer.json"
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        print("Training tokenizer...")
        trainer = trainers.BpeTrainer(vocab_size=config['model']['vocab_size'], special_tokens=[
            "<|witness|>", "<|judge|>", "<|rebuilder|>", "<|silence|>",
            "<|silent_end|>", "<|reset|>", "<|pad|>", "<|eos|>"
        ])
        # Train on dataset
        files = [config['data']['train_path']]
        tokenizer.train(files, trainer)
        tokenizer.save(tokenizer_path)
    
    # Datasets
    train_dataset = LamentDataset(config['data']['train_path'], tokenizer, config['model']['max_seq_len'])
    val_dataset = LamentDataset(config['data']['val_path'], tokenizer, config['model']['max_seq_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Model
    model = LamentTransformer(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config['training']['warmup_steps'])
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['max_steps'] - config['training']['warmup_steps'])
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config['training']['warmup_steps']])
    
    # Training loop
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(100):  # Max epochs
        print(f"\nEpoch {epoch + 1}")
        
        train_loss = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Train loss: {train_loss:.4f}")
        
        if step % config['training']['eval_steps'] == 0:
            val_loss = evaluate(model, val_loader, device, config)
            print(f"Val loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save checkpoint
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'step': step,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, f"../model/checkpoints/checkpoint_step_{step}.pt")
                print(f"Saved checkpoint at step {step}")
        
        scheduler.step()
        step += len(train_loader)
        
        if step >= config['training']['max_steps']:
            break
    
    print("Training complete!")


if __name__ == '__main__':
    main()

