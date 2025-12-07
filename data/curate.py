#!/usr/bin/env python3
"""
Dataset curation script for The Lament Engine
Processes raw texts and tags them with mode tokens
"""

import os
import csv
import re
from typing import List, Tuple, Optional
from pathlib import Path

# Mode tags
MODES = {
    'witness': '<|witness|>',
    'judge': '<|judge|>',
    'rebuilder': '<|rebuilder|>',
    'silence': '<|silence|>'
}

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might interfere
    text = text.strip()
    return text

def tag_text(text: str, mode: str) -> str:
    """Tag text with mode token"""
    if mode not in MODES:
        raise ValueError(f"Invalid mode: {mode}. Must be one of {list(MODES.keys())}")
    
    cleaned = clean_text(text)
    if not cleaned:
        return None
    
    return f"{MODES[mode]}{cleaned}"

def process_file(input_path: str, output_path: str, mode: str, 
                source_name: str, license_info: str, min_length: int = 50):
    """Process a single file and tag with mode"""
    if not os.path.exists(input_path):
        print(f"Warning: {input_path} does not exist")
        return
    
    output_lines = []
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
        
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if len(para) < min_length:
                continue
            
            # Tag with mode
            tagged = tag_text(para, mode)
            if tagged:
                output_lines.append(tagged)
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"Processed {len(output_lines)} segments from {input_path}")
    
    # Update manifest
    update_manifest(source_name, input_path, license_info, mode, len(output_lines))

def update_manifest(source_name: str, source_path: str, license_info: str, 
                   content_type: str, num_segments: int):
    """Update dataset manifest CSV"""
    manifest_path = "../DATASET_MANIFEST.csv"
    
    # Read existing manifest
    rows = []
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    
    # Add new entry
    new_row = {
        'source': source_name,
        'license': license_info,
        'content_type': content_type,
        'tags': content_type,
        'ethical_review': 'Pending',
        'notes': f'{num_segments} segments from {source_path}'
    }
    
    rows.append(new_row)
    
    # Write back
    fieldnames = ['source', 'license', 'content_type', 'tags', 'ethical_review', 'notes']
    with open(manifest_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def split_dataset(input_file: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataset into train/val/test"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    import random
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    # Write splits
    base_dir = os.path.dirname(input_file)
    processed_dir = os.path.join(base_dir, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    with open(os.path.join(processed_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(processed_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    with open(os.path.join(processed_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    
    print(f"Split dataset: {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test")

def main():
    """Main curation workflow"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Curate dataset for Lament Engine')
    parser.add_argument('--input', type=str, required=True, help='Input file path')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--mode', type=str, required=True, choices=list(MODES.keys()), help='Mode tag')
    parser.add_argument('--source', type=str, required=True, help='Source name for manifest')
    parser.add_argument('--license', type=str, default='Public Domain', help='License information')
    parser.add_argument('--split', action='store_true', help='Split into train/val/test')
    
    args = parser.parse_args()
    
    # Process file
    process_file(args.input, args.output, args.mode, args.source, args.license)
    
    # Split if requested
    if args.split:
        split_dataset(args.output)

if __name__ == '__main__':
    main()

