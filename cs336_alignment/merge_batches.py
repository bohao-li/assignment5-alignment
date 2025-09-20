#!/usr/bin/env python3
"""
Simple script to merge all batch files into a single dataset.
"""

import os
import glob
import argparse
from pathlib import Path

def merge_batches(input_dir="output_batches", output_file="merged_sft_dataset.jsonl"):
    """
    Merge all batch_XX.jsonl files in the input directory into a single file.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Find all batch files
    batch_files = sorted(glob.glob(str(input_path / "batch_*.jsonl")))
    
    if not batch_files:
        print(f"No batch files found in {input_dir}")
        return
    
    print(f"Found {len(batch_files)} batch files:")
    
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for batch_file in batch_files:
            print(f"  Processing {batch_file}...")
            
            if not os.path.exists(batch_file):
                print(f"    Warning: {batch_file} not found, skipping")
                continue
            
            lines_in_batch = 0
            try:
                with open(batch_file, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:  # Skip empty lines
                            outfile.write(line + '\n')
                            lines_in_batch += 1
                
                print(f"    Added {lines_in_batch} entries")
                total_lines += lines_in_batch
                
            except Exception as e:
                print(f"    Error reading {batch_file}: {e}")
    
    print(f"\nMerge complete!")
    print(f"Total entries: {total_lines}")
    print(f"Output file: {output_file}")
    
    # Show file size
    if os.path.exists(output_file):
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

def main():
    parser = argparse.ArgumentParser(description='Merge batch files into single dataset')
    parser.add_argument('--input-dir', '-i', default='output_batches', 
                       help='Directory containing batch files (default: output_batches)')
    parser.add_argument('--output', '-o', default='merged_sft_dataset.jsonl',
                       help='Output filename (default: merged_sft_dataset.jsonl)')
    
    args = parser.parse_args()
    
    merge_batches(args.input_dir, args.output)

if __name__ == "__main__":
    main()