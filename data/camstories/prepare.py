# %%
import os
import json
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

# Add the workspace root to path to import tokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tokenizer import get_tokenizer


# %%
def parse_dataset_name(dataset_name):
    """
    Parse dataset name to extract vocab size and tokenizer type
    Examples:
    - camstories_5000 -> vocab_size=5000, tokenizer_type=None
    - camstories_5000_pmod -> vocab_size=5000, tokenizer_type='pmod'
    - camstories_full -> vocab_size=0, tokenizer_type=None
    """
    parts = dataset_name.split('_')
    
    if 'full' in parts:
        return 0, None
    
    # Find vocab size (now as direct numbers)
    vocab_size = 0
    tokenizer_type = None
    
    for i, part in enumerate(parts):
        try:
            vocab_size = int(part)
            # Check if there's a tokenizer type after the vocab size
            if i + 1 < len(parts):
                tokenizer_type = parts[i + 1]
            break
        except ValueError:
            continue
    
    return vocab_size, tokenizer_type

# %%
def get_tokenizer_name(tokenizer_type):
    """
    Map tokenizer type to tokenizer name for get_tokenizer function
    """
    if tokenizer_type == 'pmod':
        return 'word_level_pmod'
    else:
        return 'word_level'  # Default to word_level

# %%
def encode_stories_to_binary(stories, tokenizer, output_path):
    """
    Encode stories using tokenizer and save as binary file
    """
    print(f"Encoding {len(stories)} stories...")
    
    all_token_ids = []
    
    for i, story in enumerate(tqdm(stories, desc="Encoding stories")):
        # Tokenize the story
        token_ids = tokenizer(
            story, 
            padding=False,
            truncation=True,
            max_length=1024,
            return_tensors='pt',
            add_special_tokens=True
        )['input_ids']
        
        # Convert to list and extend
        all_token_ids.extend(token_ids.cpu().numpy().flatten().tolist())
    
    print(f"Total tokens: {len(all_token_ids)}")
    
    # Convert to numpy array and save
    token_ids_np = np.array(all_token_ids, dtype=np.uint16)
    print(f"Saving to {output_path}")
    token_ids_np.tofile(output_path)
    
    return len(all_token_ids)

# %%
def create_metadata(tokenizer, output_path):
    """
    Create metadata file in nanoGPT format
    Handles both regular tokenizers and SimpleStories tokenizers with special tokens
    """
    print("Creating metadata...")
    
    # Get the actual vocabulary size from the tokenizer
    base_vocab_size = tokenizer.vocab_size
    
    # For SimpleStories tokenizer, special tokens might be added outside base vocab
    # We need to include them in the metadata
    if hasattr(tokenizer, 'bos_token_id') and hasattr(tokenizer, 'eos_token_id') and hasattr(tokenizer, 'unk_token_id'):
        max_token_id = max(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
        actual_vocab_size = max(base_vocab_size, max_token_id + 1)
        
        print(f"Base vocab size: {base_vocab_size}")
        print(f"Max token ID: {max_token_id}")
        print(f"Actual vocab size for metadata: {actual_vocab_size}")
    else:
        actual_vocab_size = base_vocab_size
    
    # Create itos (integer to string mapping)
    itos = []
    for i in tqdm(range(actual_vocab_size), desc="Creating itos mapping"):
        try:
            token = tokenizer.decode([i])
            itos.append(token)
        except:
            itos.append(f"<unk_{i}>")  # Fallback for any issues
    
    # Create stoi (string to integer mapping)
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta = {
        'vocab_size': actual_vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Metadata saved to {output_path}")
    print(f"Vocabulary size: {actual_vocab_size}")
    
    return meta

# %%
def prepare_dataset(dataset_name, tokenizer_name=None, model_name=None, dataset_file=None):
    """
    Main function to prepare a dataset
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "camstories_5000", "camstories_5000_pmod", "camstories_10000_ss")
        tokenizer_name (str, optional): Type of tokenizer to use. If None, inferred from dataset_name.
                                      Options: "word_level", "word_level_pmod", "byte_pair", "word_piece", "huggingface"
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer type
        dataset_file (str, optional): Specific dataset file to use. If None, auto-detected from dataset_name.
    """
    print(f"Preparing dataset: {dataset_name}")
    
    # Parse dataset name for vocab size and tokenizer type (if not explicitly provided)
    vocab_size, tokenizer_type = parse_dataset_name(dataset_name)
    
    # Use provided tokenizer_name or infer from dataset_name
    if tokenizer_name is None:
        # Check if dataset name indicates SimpleStories tokenizer
        if '_ss' in dataset_name:
            tokenizer_name = 'word_piece'  # SimpleStories tokenizer
        else:
            tokenizer_name = get_tokenizer_name(tokenizer_type)
    
    print(f"Vocab size: {vocab_size}, Tokenizer type: {tokenizer_type}, Tokenizer name: {tokenizer_name}")
    
    # For HuggingFace tokenizers, we don't need the vocab files
    if tokenizer_name == "huggingface":
        if model_name is None:
            raise ValueError("model_name must be provided when using 'huggingface' tokenizer")
        print(f"Using HuggingFace model: {model_name}")
    
    # Set up paths
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Auto-detect dataset file if not provided
    if dataset_file is None:
        # Try different dataset file patterns
        if 'full' in dataset_name and '_ss' in dataset_name:
            dataset_file = "camstories_full.parquet"
        elif '10000' in dataset_name and '_ss' in dataset_name:
            dataset_file = "camstories_10k_cased.parquet"
        elif dataset_name.startswith("camstories_"):
            # Extract base name without tokenizer suffix
            base_name = dataset_name.replace('_ss', '')
            dataset_file = f"{base_name}.parquet"
        else:
            dataset_file = f"{dataset_name}.parquet"
    
    dataset_path = os.path.join(data_dir, dataset_file)
    vocab_path = os.path.join(data_dir, f"{dataset_name}_vocab.parquet")
    
    # Create output directory
    if dataset_name.startswith("camstories_"):
        output_subdir = dataset_name[len("camstories_"):]  # e.g. "5000" or "5000_pmod"
    else:
        output_subdir = dataset_name  # Fallback – should not happen for official names

    # Append tokenizer info to output directory for non-default tokenizers
    if tokenizer_name == "huggingface" and model_name:
        # Clean model name for directory (replace / and : with _)
        clean_model_name = model_name.replace("/", "_").replace(":", "_")
        output_subdir = f"{output_subdir}_{clean_model_name}"
    elif tokenizer_name == "word_piece":
        # For SimpleStories tokenizer, use ss_tok suffix
        if 'full' in dataset_name:
            output_subdir = "full_cased_ss_tok"
        else:
            output_subdir = "10000_cased_ss_tok"
    elif tokenizer_name not in ["word_level", "word_level_pmod"]:
        output_subdir = f"{output_subdir}_{tokenizer_name}"

    output_dir = os.path.join(data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")
    
    print(f"Output directory: {output_dir}")
    
    # Load datasets
    print("Loading datasets...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Only check for vocab file if we need it (not for HuggingFace tokenizers)
    if tokenizer_name not in ["huggingface", "byte_pair", "word_piece"]:
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    # Try fastparquet first, fallback to pyarrow
    try:
        stories_df = pd.read_parquet(dataset_path, engine='fastparquet')
    except:
        print("Fastparquet failed, trying pyarrow...")
        stories_df = pd.read_parquet(dataset_path, engine='pyarrow')
    
    print(f"Loaded {len(stories_df)} stories")
    
    if tokenizer_name not in ["huggingface", "byte_pair", "word_piece"]:
        vocab_df = pd.read_parquet(vocab_path)
        print(f"Loaded {len(vocab_df)} vocab tokens")
    
    # Get tokenizer using the tokenizer.py function
    print(f"Initializing tokenizer: {tokenizer_name}")
    if tokenizer_name == "huggingface":
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="camstories", 
            vocab_size=vocab_size,
            built_vocab=True,
            model_name=model_name
        )
    elif tokenizer_name == "word_piece":
        # For SimpleStories tokenizer, use appropriate dataset name and don't require pre-built vocab
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="cam_stories",
            vocab_size=4096,  # SimpleStories tokenizer uses 4096 vocab size
            built_vocab=False
        )
    else:
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="camstories", 
            vocab_size=vocab_size,
            built_vocab=True
        )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, 'bos_token') and hasattr(tokenizer, 'eos_token') and hasattr(tokenizer, 'unk_token'):
        print(f"Special tokens: BOS='{tokenizer.bos_token}', EOS='{tokenizer.eos_token}', UNK='{tokenizer.unk_token}'")
        if hasattr(tokenizer, 'bos_token_id'):
            print(f"Special token IDs: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, UNK={tokenizer.unk_token_id}")
    
    # Check the columns and split data
    print(f"Columns: {stories_df.columns.tolist()}")
    
    # Try different split column names and handle missing splits
    if 'split' in stories_df.columns:
        print("Split distribution:")
        print(stories_df['split'].value_counts())
        train_stories = stories_df[stories_df['split'] == 'train']['story'].tolist()
        val_stories = stories_df[stories_df['split'] == 'val']['story'].tolist()
        
        # If no val split exists, try test
        if len(val_stories) == 0:
            val_stories = stories_df[stories_df['split'] == 'test']['story'].tolist()
    else:
        # If no split column, create a simple 90/10 split
        print("No 'split' column found, creating 90/10 train/val split")
        n_train = int(0.9 * len(stories_df))
        train_stories = stories_df['story'][:n_train].tolist()
        val_stories = stories_df['story'][n_train:].tolist()
    
    print(f"Train stories: {len(train_stories)}")
    print(f"Validation stories: {len(val_stories)}")
    
    # Encode and save binary files
    print(f"\nEncoding training data ({len(train_stories)} stories)...")
    train_tokens = encode_stories_to_binary(train_stories, tokenizer, train_bin_path)
    
    print(f"\nEncoding validation data ({len(val_stories)} stories)...")
    val_tokens = encode_stories_to_binary(val_stories, tokenizer, val_bin_path)
    
    # Create metadata
    create_metadata(tokenizer, meta_path)
    
    print(f"\nDataset preparation complete!")
    print(f"Output files:")
    print(f"  - Train binary: {train_bin_path} ({train_tokens} tokens)")
    print(f"  - Val binary: {val_bin_path} ({val_tokens} tokens)")
    print(f"  - Metadata: {meta_path}")
    
    return {
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'vocab_size': tokenizer.vocab_size,
        'output_dir': output_dir
    }



# %%
def create_dataset(dataset_name, tokenizer_name=None, model_name=None, dataset_file=None):
    """
    Create dataset with specified tokenizer
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "camstories_5000", "camstories_10000_ss")
        tokenizer_name (str, optional): Type of tokenizer to use
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer
        dataset_file (str, optional): Specific dataset file to use
    """
    result = prepare_dataset(dataset_name, tokenizer_name, model_name, dataset_file) 
    print(f"Outputed meta and binary files to {result['output_dir']}")
    print(f"Amount of train tokens: {result['train_tokens']}")
    print(f"Amount of val tokens: {result['val_tokens']}")
    print(f"Vocabulary size: {result['vocab_size']}")
    return result


def create_dataset_with_huggingface_tokenizer(dataset_name, model_name):
    """
    Convenience function to create dataset with any HuggingFace tokenizer
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "camstories_5000") 
        model_name (str): HuggingFace model name (e.g., "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-125M")
    """
    return create_dataset(dataset_name, tokenizer_name="huggingface", model_name=model_name)


def create_dataset_with_simplestories_tokenizer(dataset_name="camstories_10000_ss", dataset_file=None):
    """
    Convenience function to create dataset with SimpleStories tokenizer
    
    Args:
        dataset_name (str): Name of the dataset (e.g., "camstories_10000_ss", "camstories_full_ss")
        dataset_file (str, optional): Specific dataset file to use. Auto-detected if None.
    """
    return create_dataset(dataset_name, tokenizer_name="word_piece", dataset_file=dataset_file)


def create_camstories_10k_ss():
    """
    Create camstories 10k dataset with SimpleStories tokenizer
    Equivalent to the old prepare_camstories_ss.py functionality
    """
    return create_dataset_with_simplestories_tokenizer("camstories_10000_ss", "camstories_10k_cased.parquet")


def create_camstories_full_ss():
    """
    Create camstories full dataset with SimpleStories tokenizer  
    Equivalent to the old prepare_camstories_full_ss.py functionality
    """
    return create_dataset_with_simplestories_tokenizer("camstories_full_ss", "camstories_full.parquet")


# Example usage:
# create_dataset("camstories_5000_pmod")  # Use default word_level_pmod tokenizer
# create_dataset("camstories_5000", tokenizer_name="byte_pair")  # Use byte_pair tokenizer
# create_dataset_with_huggingface_tokenizer("camstories_5000", "EleutherAI/gpt-neo-125M")  # Use GPT-Neo tokenizer
# create_dataset_with_huggingface_tokenizer("camstories_5000", "microsoft/DialoGPT-medium")  # Use DialoGPT tokenizer
# create_dataset_with_simplestories_tokenizer("camstories_10000_ss")  # Use SimpleStories tokenizer
# create_camstories_10k_ss()  # Convenience function for 10k dataset with SimpleStories tokenizer
# create_camstories_full_ss()  # Convenience function for full dataset with SimpleStories tokenizer

# %%
# Test the default word-level tokenizer
# create_dataset("camstories_10000"

# %%
# Uncomment the lines below to test other tokenizers
# create_dataset_with_huggingface_tokenizer("camstories_10000", "SimpleStories/SimpleStories-35M")
create_camstories_full_ss()

# %%