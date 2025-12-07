#!/usr/bin/env python3
"""
Clean Wikipedia data for The Lament Engine
Removes citations, section headers, and formats text properly
"""

import re
import os
from pathlib import Path
from typing import List, Tuple

def clean_wikipedia_text(text: str) -> str:
    """Clean Wikipedia-specific formatting"""
    
    # Remove Wikipedia citations like [1], [2][3], [4]:328
    text = re.sub(r'\[\d+\](\[\d+\])*(:\d+)?', '', text)
    
    # Remove reference sections (lines starting with ^ or containing "See also", "References", etc.)
    lines = text.split('\n')
    cleaned_lines = []
    skip_section = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip reference sections
        if any(marker in line.lower() for marker in ['references', 'see also', 'external links', 
                                                      'further reading', 'notes', '^']):
            skip_section = True
            continue
        
        # Skip copyright notices
        if 'copyright' in line.lower() or 'all rights reserved' in line.lower():
            continue
        
        # Skip ISBN, publication info
        if re.match(r'^(ISBN|First published|©)', line, re.IGNORECASE):
            continue
        
        # Reset skip flag on new major section (all caps or title case)
        if line.isupper() and len(line) > 3:
            skip_section = False
            # Use as section header but clean it
            line = line.title()
        
        if not skip_section:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def split_into_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """Split text into meaningful paragraphs"""
    paragraphs = []
    
    # Split by double newlines first
    chunks = text.split('\n\n')
    
    for chunk in chunks:
        chunk = chunk.strip()
        
        # Skip very short chunks
        if len(chunk) < min_length:
            continue
        
        # Split long paragraphs by sentences
        if len(chunk) > 1000:
            sentences = re.split(r'[.!?]+\s+', chunk)
            current_para = []
            current_len = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if current_len + len(sentence) > 800:
                    if current_para:
                        paragraphs.append(' '.join(current_para))
                    current_para = [sentence]
                    current_len = len(sentence)
                else:
                    current_para.append(sentence)
                    current_len += len(sentence) + 1
            
            if current_para:
                paragraphs.append(' '.join(current_para))
        else:
            paragraphs.append(chunk)
    
    return paragraphs

def determine_mode(text: str) -> str:
    """Determine appropriate mode based on content"""
    text_lower = text.lower()
    
    # Judge mode: analysis, evaluation, legal terms
    judge_keywords = ['violation', 'crime', 'illegal', 'unlawful', 'prosecuted', 
                     'convicted', 'sentenced', 'court', 'justice', 'legal', 
                     'moral', 'ethical', 'wrong', 'right', 'should', 'must',
                     'demonstrates', 'indicates', 'shows', 'proves']
    
    # Rebuilder mode: prevention, solutions, future
    rebuilder_keywords = ['prevent', 'prevention', 'solution', 'reform', 
                         'improve', 'establish', 'create', 'build', 'reconstruct',
                         'healing', 'recovery', 'future', 'should', 'must',
                         'recommend', 'propose', 'suggest', 'way forward']
    
    # Silence mode: reflection, weight, limits of language
    silence_keywords = ['weight', 'heavy', 'words fail', 'cannot express',
                       'beyond words', 'silence', 'reflection', 'contemplate',
                       'meditate', 'ponder', 'unfathomable', 'inexpressible']
    
    judge_score = sum(1 for kw in judge_keywords if kw in text_lower)
    rebuilder_score = sum(1 for kw in rebuilder_keywords if kw in text_lower)
    silence_score = sum(1 for kw in silence_keywords if kw in text_lower)
    
    # Default to witness, but switch if strong signal
    if judge_score >= 3:
        return 'judge'
    elif rebuilder_score >= 2:
        return 'rebuilder'
    elif silence_score >= 2:
        return 'silence'
    else:
        return 'witness'  # Default: factual recounting

def process_file(input_path: str, output_path: str, mode: str = None, 
                source_name: str = None, min_para_length: int = 100):
    """Process a Wikipedia file"""
    
    print(f"Processing: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Clean Wikipedia formatting
    cleaned = clean_wikipedia_text(content)
    
    # Split into paragraphs
    paragraphs = split_into_paragraphs(cleaned, min_para_length)
    
    print(f"  Found {len(paragraphs)} paragraphs")
    
    # Tag paragraphs
    tagged_paragraphs = []
    for para in paragraphs:
        # Determine mode if not specified
        para_mode = mode if mode else determine_mode(para)
        
        # Clean paragraph
        para = re.sub(r'\s+', ' ', para).strip()
        
        if len(para) >= min_para_length:
            tagged_para = f"<|{para_mode}|>{para}"
            tagged_paragraphs.append(tagged_para)
    
    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tagged_paragraphs))
    
    print(f"  Wrote {len(tagged_paragraphs)} tagged paragraphs to {output_path}")
    
    # Statistics
    mode_counts = {}
    for para in tagged_paragraphs:
        mode_match = re.search(r'<\|(\w+)\|>', para)
        if mode_match:
            mode_name = mode_match.group(1)
            mode_counts[mode_name] = mode_counts.get(mode_name, 0) + 1
    
    print(f"  Mode distribution: {mode_counts}")
    
    return len(tagged_paragraphs), mode_counts

def combine_and_split(input_files: List[str], output_dir: str, 
                     train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Combine processed files and split into train/val/test"""
    
    all_lines = []
    
    for input_file in input_files:
        if os.path.exists(input_file):
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
                all_lines.extend(lines)
                print(f"Added {len(lines)} lines from {input_file}")
    
    print(f"\nTotal lines: {len(all_lines)}")
    
    # Shuffle
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(all_lines)
    
    # Split
    total = len(all_lines)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_lines = all_lines[:train_end]
    val_lines = all_lines[train_end:val_end]
    test_lines = all_lines[val_end:]
    
    # Write splits
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_lines))
    
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_lines))
    
    with open(os.path.join(output_dir, 'test.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(test_lines))
    
    print(f"\nSplit complete:")
    print(f"  Train: {len(train_lines)} lines")
    print(f"  Val: {len(val_lines)} lines")
    print(f"  Test: {len(test_lines)} lines")
    
    # Calculate approximate tokens (rough estimate: 1 token ≈ 4 chars)
    total_chars = sum(len(l) for l in all_lines)
    estimated_tokens = total_chars / 4
    
    print(f"\nEstimated tokens: ~{int(estimated_tokens):,}")
    print(f"  Train: ~{int(len(train_lines) * total_chars / len(all_lines) / 4):,}")
    print(f"  Val: ~{int(len(val_lines) * total_chars / len(all_lines) / 4):,}")
    print(f"  Test: ~{int(len(test_lines) * total_chars / len(all_lines) / 4):,}")
    
    # Check if enough data (rough guideline: need at least 1M tokens for small model)
    if estimated_tokens < 500000:
        print(f"\nWARNING: Dataset may be too small for training.")
        print(f"   Recommended: at least 1M tokens for a small model.")
        print(f"   Current: ~{int(estimated_tokens):,} tokens")
        print(f"   Note: This may still work for a very small model (~20M params)")
    else:
        print(f"\nDataset size looks adequate for training.")

def main():
    """Main processing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean Wikipedia data for Lament Engine')
    parser.add_argument('--input-dir', type=str, default='raw', 
                       help='Input directory (default: raw)')
    parser.add_argument('--output-dir', type=str, default='processed',
                       help='Output directory (default: processed)')
    parser.add_argument('--mode', type=str, choices=['witness', 'judge', 'rebuilder', 'silence'],
                       help='Force mode for all text (default: auto-detect)')
    parser.add_argument('--combine', action='store_true',
                       help='Combine all processed files and split into train/val/test')
    parser.add_argument('--min-length', type=int, default=100,
                       help='Minimum paragraph length (default: 100)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Process all text files in input directory
    text_files = list(input_dir.glob('*.txt'))
    
    if not text_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    processed_files = []
    
    for txt_file in text_files:
        if txt_file.name == 'example.txt':
            continue  # Skip example file
        
        output_file = output_dir / f"cleaned_{txt_file.name}"
        source_name = txt_file.stem
        
        try:
            count, modes = process_file(str(txt_file), str(output_file), 
                                       args.mode, source_name, args.min_length)
            processed_files.append(str(output_file))
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
    
    # Combine and split if requested
    if args.combine and processed_files:
        print("\n" + "="*60)
        print("Combining and splitting datasets...")
        print("="*60)
        combine_and_split(processed_files, str(output_dir))

if __name__ == '__main__':
    main()

