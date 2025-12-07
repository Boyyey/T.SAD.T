#!/usr/bin/env python3
"""
Export PyTorch model weights to C-compatible binary format
"""

import torch
import json
import struct
import numpy as np
from pathlib import Path

def export_weights(checkpoint_path: str, config_path: str, output_path: str, quantize: bool = True):
    """Export model weights to binary format for C runtime"""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_cfg = config['model']
    
    # Collect all weights
    weights = {}
    
    # Embeddings
    if 'token_embedding.weight' in state_dict:
        weights['embeddings'] = state_dict['token_embedding.weight'].numpy().astype(np.float32)
    
    # Output embeddings (tied or separate)
    if 'output_projection.weight' in state_dict:
        weights['output_embeddings'] = state_dict['output_projection.weight'].numpy().astype(np.float32)
    else:
        # Tied embeddings
        weights['output_embeddings'] = weights['embeddings'].T.copy()
    
    # Transformer layers
    layer_weights = []
    for i in range(model_cfg['num_layers']):
        layer_dict = {}
        prefix = f'transformer.layers.{i}.'
        
        # Self-attention
        if f'{prefix}self_attn.q_proj.weight' in state_dict:
            layer_dict['q_proj'] = state_dict[f'{prefix}self_attn.q_proj.weight'].numpy().astype(np.float32)
            layer_dict['k_proj'] = state_dict[f'{prefix}self_attn.k_proj.weight'].numpy().astype(np.float32)
            layer_dict['v_proj'] = state_dict[f'{prefix}self_attn.v_proj.weight'].numpy().astype(np.float32)
            layer_dict['o_proj'] = state_dict[f'{prefix}self_attn.o_proj.weight'].numpy().astype(np.float32)
        
        # FFN
        if f'{prefix}linear1.weight' in state_dict:
            layer_dict['ffn_gate'] = state_dict[f'{prefix}linear1.weight'].numpy().astype(np.float32)
            layer_dict['ffn_up'] = state_dict[f'{prefix}linear2.weight'].numpy().astype(np.float32)
            layer_dict['ffn_down'] = state_dict[f'{prefix}linear3.weight'].numpy().astype(np.float32)
        
        # Layer norms
        if f'{prefix}norm1.weight' in state_dict:
            layer_dict['ln1_weight'] = state_dict[f'{prefix}norm1.weight'].numpy().astype(np.float32)
            layer_dict['ln1_bias'] = state_dict[f'{prefix}norm1.bias'].numpy().astype(np.float32)
            layer_dict['ln2_weight'] = state_dict[f'{prefix}norm2.weight'].numpy().astype(np.float32)
            layer_dict['ln2_bias'] = state_dict[f'{prefix}norm2.bias'].numpy().astype(np.float32)
        
        layer_weights.append(layer_dict)
    
    # Final layer norm
    if 'ln_f.weight' in state_dict:
        weights['ln_f_weight'] = state_dict['ln_f.weight'].numpy().astype(np.float32)
        weights['ln_f_bias'] = state_dict['ln_f.bias'].numpy().astype(np.float32)
    
    # Quantize if requested
    if quantize:
        print("Quantizing weights to 8-bit...")
        # Simple quantization (in production, use proper quantization)
        for key in weights:
            if isinstance(weights[key], np.ndarray):
                # Quantize to int8
                scale = np.abs(weights[key]).max() / 127.0
                weights[key] = (weights[key] / scale).astype(np.int8)
                weights[f'{key}_scale'] = np.array([scale], dtype=np.float32)
    
    # Write binary file
    with open(output_path, 'wb') as f:
        # Write header
        header = struct.pack('I', model_cfg['vocab_size'])
        header += struct.pack('I', model_cfg['hidden_size'])
        header += struct.pack('I', model_cfg['num_layers'])
        f.write(header)
        
        # Write embeddings
        if 'embeddings' in weights:
            emb = weights['embeddings']
            f.write(struct.pack('I', emb.shape[0]))
            f.write(struct.pack('I', emb.shape[1]))
            f.write(emb.tobytes())
        
        # Write output embeddings
        if 'output_embeddings' in weights:
            out_emb = weights['output_embeddings']
            f.write(struct.pack('I', out_emb.shape[0]))
            f.write(struct.pack('I', out_emb.shape[1]))
            f.write(out_emb.tobytes())
        
        # Write layers
        f.write(struct.pack('I', len(layer_weights)))
        for layer in layer_weights:
            # Write each weight matrix in layer
            for key, value in layer.items():
                if isinstance(value, np.ndarray):
                    f.write(struct.pack('I', len(key)))
                    f.write(key.encode('ascii'))
                    f.write(struct.pack('I', len(value.shape)))
                    for dim in value.shape:
                        f.write(struct.pack('I', dim))
                    f.write(value.tobytes())
    
    print(f"Exported weights to {output_path}")
    print(f"Total size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Export PyTorch weights to C format')
    parser.add_argument('--checkpoint', type=str, required=True, help='PyTorch checkpoint path')
    parser.add_argument('--config', type=str, required=True, help='Model config JSON')
    parser.add_argument('--output', type=str, required=True, help='Output binary path')
    parser.add_argument('--no-quantize', action='store_true', help='Disable quantization')
    
    args = parser.parse_args()
    
    export_weights(args.checkpoint, args.config, args.output, quantize=not args.no_quantize)

