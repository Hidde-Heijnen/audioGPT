import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union

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
    output_path: str
) -> None:
    """Generate token counts from the dataset."""
    print(f"Loading base tokenizer: {base_tokenizer_name} for token counting.")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer_name,
        bos_token=special_tokens_dict["bos_token"],
        eos_token=special_tokens_dict["eos_token"],
        unk_token=special_tokens_dict["unk_token"]
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
        print(f"Set base_tokenizer.pad_token to: {base_tokenizer.eos_token}")

    print("Tokenizing data to generate token counts...")
    token_ids_list = []
    for i, text in enumerate(df[text_column]):
        if (i + 1) % 10000 == 0:
            print(f"  Tokenizing item {i+1} for counting...")
        if text and isinstance(text, str):
            encoded = base_tokenizer(text, add_special_tokens=False)
            token_ids_list.extend(encoded['input_ids'])
    
    print(f"Finished tokenizing for counting. Total tokens found: {len(token_ids_list)}")
    print("Counting token frequencies...")
    counts = Counter(token_ids_list)

    # Sort by frequency (most common first), then by token_id (for tie-breaking)
    sorted_token_counts_list = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    # Convert to string keys for JSON serialization
    string_keyed_sorted_token_counts = {str(token_id): count for token_id, count in sorted_token_counts_list}

    print(f"Saving token counts to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(string_keyed_sorted_token_counts, f, indent=2)
    print("Token counts saved.")

def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    top_k_tokens_dict: Optional[Dict[str, int]] = None,
    oov_token_idx: Optional[int] = None
) -> List[int]:
    """Encode a list of texts using the tokenizer.
    
    Args:
        texts: List of text strings to encode
        tokenizer: The tokenizer to use for encoding
        top_k_tokens_dict: Optional mapping of token IDs to new vocabulary indices
        oov_token_idx: Optional out-of-vocabulary token index to use when mapping tokens
    """
    all_token_ids = []
    
    for i, text in enumerate(texts):
        if not text or not isinstance(text, str):
            continue
            
        # Tokenize using the base tokenizer
        encoded_output = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=True,
            max_length=tokenizer.model_max_length
        )
        original_ids = encoded_output['input_ids']

        if top_k_tokens_dict is not None:
            # Map to new vocabulary, defaulting to OOV token index
            new_ids = [top_k_tokens_dict.get(str(token_id), oov_token_idx) for token_id in original_ids]
        else:
            new_ids = original_ids
            
        all_token_ids.extend(new_ids)

        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} texts...")
            
    return all_token_ids

def save_binary_file(token_ids: List[int], output_path: str) -> None:
    """Save token IDs to a binary file."""
    token_ids_np = np.array(token_ids, dtype=np.uint16)
    print(f"Saving tokenized data to {output_path}...")
    token_ids_np.tofile(output_path)
    print(f"Saved to {output_path}.")

def create_meta_file(
    tokenizer: AutoTokenizer,
    output_path: str,
    top_k_tokens: Optional[List[str]] = None
) -> None:
    """Create and save metadata file."""
    if top_k_tokens is not None:
        vocab_size = len(top_k_tokens)
        itos = [tokenizer.decode([int(token_id_str)]) for token_id_str in top_k_tokens]
    else:
        vocab_size = tokenizer.vocab_size
        itos = [tokenizer.decode([i]) for i in range(vocab_size)]
    
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Metadata saved to {output_path}")
    print(f"  Vocabulary size: {vocab_size}")

def main():
    parser = argparse.ArgumentParser(description='Convert parquet file to binary format for training')
    parser.add_argument('--parquet_path', type=str, required=True, help='Path to input parquet file')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output files')
    parser.add_argument('--text_column', type=str, default='story', help='Name of the text column in parquet file')
    parser.add_argument('--split_column', type=str, default='split', help='Name of the split column in parquet file')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Name of the base tokenizer to use')
    parser.add_argument('--special_tokens_path', type=str, default='special_tokens_map.json', help='Path to special tokens JSON file')
    parser.add_argument('--top_k', type=int, default=None, help='Number of most frequent tokens to keep')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load special tokens
    special_tokens = load_special_tokens(args.special_tokens_path)
    
    # Load parquet file
    print(f"Loading parquet file from {args.parquet_path}...")
    df = pd.read_parquet(args.parquet_path)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        bos_token=special_tokens["bos_token"],
        eos_token=special_tokens["eos_token"],
        unk_token=special_tokens["unk_token"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate token counts if top_k is specified
    if args.top_k is not None:
        token_counts_path = os.path.join(args.output_dir, "token_counts.json")
        generate_token_counts(df, args.text_column, args.tokenizer_name, special_tokens, token_counts_path)
        
        # Load token counts and create top_k mapping
        with open(token_counts_path, 'r') as f:
            token_counts = json.load(f)
        
        # Get top k tokens
        top_k_tokens = list(token_counts.keys())[:args.top_k]
        top_k_tokens_dict = {token_id: idx for idx, token_id in enumerate(top_k_tokens)}
        oov_token_idx = len(top_k_tokens) - 1
    else:
        top_k_tokens = None
        top_k_tokens_dict = None
        oov_token_idx = None

    # Process train and validation splits
    for split in ['train', 'val']:
        split_df = df[df[args.split_column] == split]
        if len(split_df) == 0:
            print(f"Warning: No {split} data found in the parquet file")
            continue
            
        print(f"\nProcessing {split} split...")
        token_ids = encode_texts(
            split_df[args.text_column].tolist(),
            tokenizer,
            top_k_tokens_dict,
            oov_token_idx
        )
        
        output_path = os.path.join(args.output_dir, f"{split}.bin")
        save_binary_file(token_ids, output_path)

    # Create meta file
    meta_path = os.path.join(args.output_dir, "meta.pkl")
    create_meta_file(tokenizer, meta_path, top_k_tokens)

    print("\nConversion complete!")
    print(f"Output files are in: {args.output_dir}")

if __name__ == '__main__':
    main() 