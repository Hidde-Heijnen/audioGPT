#!/usr/bin/env python3
"""
Quick sampling script for different model checkpoints
Usage: python quick_sample.py <checkpoint_path> [options]
"""
import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_sample.py <checkpoint_path> [options]")
        print("Example: python quick_sample.py out/camstories_5k/medium")
        print("Example: python quick_sample.py out/camstories_10k/medium --num_samples=5 --temperature=0.9")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    # Check if checkpoint exists
    ckpt_file = os.path.join(checkpoint_path, 'ckpt.pt')
    if not os.path.exists(ckpt_file):
        print(f"Error: Checkpoint not found at {ckpt_file}")
        sys.exit(1)
    
    # Build command
    cmd = ["python", "sample.py", f"--out_dir={checkpoint_path}", "--device=cpu"]
    
    # Add any additional arguments
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    # Run the sampling
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 