"""
Convert a parquet file with story and split columns into binary format for training.

Example usage:
    # Basic usage with default settings (gpt2 tokenizer, story/split columns)
    python scripts/utils/parquet_to_bin.py \
        --parquet_path=data/camstories_10k/camstories_10k.parquet \
        --output_dir=data/camstories_10k/ss_tokenized \

    # Using a different tokenizer and custom column names
    python scripts/utils/parquet_to_bin.py \
        --parquet_path=data/stories.parquet \
        --output_dir=data/processed \
        --text_column=text \
        --split_column=dataset_split \
        --tokenizer_name=EleutherAI/gpt-neo-125M \
        --special_tokens_path=data/special_tokens.json

    # Limiting vocabulary to top 10k tokens
    python scripts/utils/parquet_to_bin.py \
        --parquet_path=data/stories.parquet \
        --output_dir=data/processed \
        --special_tokens_path=data/special_tokens.json \
        --top_k=10000

Required files:
    - Input parquet file with at least two columns:
        * A text column (default: 'story')
        * A split column (default: 'split') with values 'train' and 'val'
    - Special tokens JSON file with format:
        {
            "bos_token": {"content": "<|startoftext|>"},
            "eos_token": {"content": "<|endoftext|>"},
            "unk_token": {"content": "<|unknown|>"}
        }

Output files:
    - train.bin: Binary file containing tokenized training data
    - val.bin: Binary file containing tokenized validation data
    - meta.pkl: Pickle file containing vocabulary information
    - token_counts.json: (if --top_k specified) Token frequency counts
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import logging
from datetime import datetime

# Set up logging
def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    log_file = os.path.join(output_dir, f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_special_tokens(file_path: str) -> Dict[str, str]:
    """Load special tokens from a JSON file."""
    with open(file_path, 'r') as f:
        special_tokens_map = json.load(f)
    return {
        "bos_token": special_tokens_map["bos_token"]["content"],
        "eos_token": special_tokens_map["eos_token"]["content"],
        "unk_token": special_tokens_map["unk_token"]["content"],
    }

def generate_token_counts(
    df: pd.DataFrame,
    text_column: str,
    base_tokenizer_name: str,
    special_tokens_dict: Dict[str, str],
    output_path: str,
    logger: logging.Logger
) -> None:
    """Generate token counts from the dataset."""
    logger.info(f"Loading base tokenizer: {base_tokenizer_name} for token counting.")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer_name,
        bos_token=special_tokens_dict["bos_token"],
        eos_token=special_tokens_dict["eos_token"],
        unk_token=special_tokens_dict["unk_token"]
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        logger.info(f"Set base_tokenizer.pad_token to: {base_tokenizer.eos_token}")

    logger.info("Tokenizing data to generate token counts...")
    token_ids_list = []
    for text in tqdm(df[text_column], desc="Counting tokens", unit="texts"):
        if text and isinstance(text, str):
            encoded = base_tokenizer(text, add_special_tokens=False)
            token_ids_list.extend(encoded['input_ids'])
    
    logger.info(f"Finished tokenizing for counting. Total tokens found: {len(token_ids_list):,}")
    logger.info("Counting token frequencies...")
    counts = Counter(token_ids_list)

    # Sort by frequency (most common first), then by token_id (for tie-breaking)
    sorted_token_counts_list = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    # Convert to string keys for JSON serialization
    string_keyed_sorted_token_counts = {str(token_id): count for token_id, count in sorted_token_counts_list}

    logger.info(f"Saving token counts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(string_keyed_sorted_token_counts, f, indent=2)
    logger.info("Token counts saved.")

def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    top_k_tokens_dict: Optional[Dict[str, int]] = None,
    oov_token_idx: Optional[int] = None,
    logger: logging.Logger = None,
    max_length: int = 2048  # Add default max length
) -> List[int]:
    """Encode a list of texts using the tokenizer."""
    all_token_ids = []
    
    for text in tqdm(texts, desc="Encoding texts", unit="text"):
        if not text or not isinstance(text, str):
            continue
            
        # Tokenize using the base tokenizer
        try:
            encoded_output = tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=max_length
            )
            original_ids = encoded_output['input_ids']

            if top_k_tokens_dict is not None:
                # Map to new vocabulary, defaulting to OOV token index
                new_ids = [top_k_tokens_dict.get(str(token_id), oov_token_idx) for token_id in original_ids]
            else:
                new_ids = original_ids
                
            all_token_ids.extend(new_ids)
        except Exception as e:
            logger.warning(f"Error processing text: {str(e)[:100]}... Skipping.")
            continue
            
    return all_token_ids

def save_binary_file(token_ids: List[int], output_path: str, logger: logging.Logger) -> None:
    """Save token IDs to a binary file."""
    token_ids_np = np.array(token_ids, dtype=np.uint16)
    logger.info(f"Saving tokenized data to {output_path}...")
    token_ids_np.tofile(output_path)
    logger.info(f"Saved {len(token_ids):,} tokens to {output_path}")

def create_meta_file(
    tokenizer: AutoTokenizer,
    output_path: str,
    top_k_tokens: Optional[List[str]] = None,
    logger: logging.Logger = None
) -> None:
    """Create and save metadata file."""
    if top_k_tokens is not None:
        vocab_size = len(top_k_tokens)
        itos = [tokenizer.decode([int(token_id_str)]) for token_id_str in tqdm(top_k_tokens, desc="Creating vocabulary", unit="token")]
    else:
        vocab_size = tokenizer.vocab_size
        itos = [tokenizer.decode([i]) for i in tqdm(range(vocab_size), desc="Creating vocabulary", unit="token")]
    
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    logger.info(f"Metadata saved to {output_path}")
    logger.info(f"Vocabulary size: {vocab_size:,}")

def main():
    parser = argparse.ArgumentParser(description='Convert parquet file to binary format for training')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to input parquet file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--text_column', type=str, default='story', help='Name of the text column in parquet file')
    parser.add_argument('--split_column', type=str, default='split', help='Name of the split column in parquet file')
    parser.add_argument('--tokenizer_name', type=str, default='SimpleStories/SimpleStories-30M', help='Name of the base tokenizer to use')
    parser.add_argument('--special_tokens_path', type=str, default='scripts/utils/special_tokens_map.json', help='Path to special tokens JSON file')
    parser.add_argument('--top_k', type=int, default=None, help='Number of most frequent tokens to keep')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length for tokenization')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.output_dir)
    logger.info("Starting conversion process...")
    logger.info(f"Arguments: {vars(args)}")

    # Load special tokens
    logger.info(f"Loading special tokens from {args.special_tokens_path}")
    special_tokens = load_special_tokens(args.special_tokens_path)
    logger.info(f"Special tokens loaded: {special_tokens}")
    
    # Load parquet file
    logger.info(f"Loading parquet file from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    logger.info(f"Loaded {len(df):,} rows from parquet file")
    
    # Initialize tokenizer
    logger.info(f"Initializing tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        bos_token=special_tokens["bos_token"],
        eos_token=special_tokens["eos_token"],
        unk_token=special_tokens["unk_token"],
        model_max_length=args.max_length  # Set model max length
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to: {tokenizer.eos_token}")

    # Generate token counts if top_k is specified
    if args.top_k is not None:
        logger.info(f"Limiting vocabulary to top {args.top_k:,} tokens")
        token_counts_path = os.path.join(args.output_dir, "token_counts.json")
        generate_token_counts(df, args.text_column, args.tokenizer_name, special_tokens, token_counts_path, logger)
        
        # Load token counts and create top_k mapping
        with open(token_counts_path, 'r') as f:
            token_counts = json.load(f)
        
        # Get top k tokens
        top_k_tokens = list(token_counts.keys())[:args.top_k]
        top_k_tokens_dict = {token_id: idx for idx, token_id in enumerate(top_k_tokens)}
        oov_token_idx = len(top_k_tokens) - 1
        logger.info(f"Created vocabulary mapping with {len(top_k_tokens):,} tokens")
    else:
        top_k_tokens = None
        top_k_tokens_dict = None
        oov_token_idx = None
        logger.info("Using full vocabulary from tokenizer")

    # Process train and validation splits
    for split in ['train', 'val']:
        split_df = df[df[args.split_column] == split]
        if len(split_df) == 0:
            logger.warning(f"No {split} data found in the parquet file")
            continue
            
        logger.info(f"\nProcessing {split} split ({len(split_df):,} texts)...")
        token_ids = encode_texts(
            split_df[args.text_column].tolist(),
            tokenizer,
            top_k_tokens_dict,
            oov_token_idx,
            logger,
            args.max_length
        )
        
        output_path = os.path.join(args.output_dir, f"{split}.bin")
        save_binary_file(token_ids, output_path, logger)

    # Create meta file
    logger.info("\nCreating metadata file...")
    meta_path = os.path.join(args.output_dir, "meta.pkl")
    create_meta_file(tokenizer, meta_path, top_k_tokens, logger)

    logger.info("\nConversion complete!")
    logger.info(f"Output files are in: {args.output_dir}")

if __name__ == '__main__':
    main() 