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
def build_vocab_from_stories_with_progress(stories: list, pattern: str, max_vocab_size: int):
    """
    Build vocabulary from stories list with progress bars - more memory efficient
    """
    import regex as re
    import gc
    from collections import Counter
    
    print("Processing stories in small batches to build vocabulary...")
    
    token_counts = Counter()
    batch_size = 1000  # Much smaller batches - 1k stories at a time
    
    for i in tqdm(range(0, len(stories), batch_size), desc="Processing story batches"):
        batch_stories = stories[i:i+batch_size]
        
        # Process each story individually to save memory
        for story in batch_stories:
            story_tokens = re.findall(pattern, story.lower())
            batch_counter = Counter(story_tokens)
            token_counts.update(batch_counter)
            
            # Clean up immediately
            del story_tokens, batch_counter
        
        # Force garbage collection every 10 batches
        if i % (batch_size * 10) == 0:
            gc.collect()
        
        # Free memory
        del batch_stories
    
    # Final cleanup
    gc.collect()
    print(f"Found {len(token_counts)} unique tokens")
    
    # Import SPECIAL_TOKENS from tokenizer to ensure consistency  
    from tokenizer import SPECIAL_TOKENS
    special_token_values = list(set(SPECIAL_TOKENS.values()))  # Remove duplicates
    num_reserved = 0
    for st in special_token_values:
        if st not in token_counts:
            num_reserved += 1

    if max_vocab_size > 0:
        print(f"Selecting top {max_vocab_size - num_reserved} tokens...")
        most_common_tokens = token_counts.most_common(max_vocab_size - num_reserved)
        vocab_tokens = [token for token, _ in most_common_tokens]
    else:
        vocab_tokens = list(token_counts.keys())

    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    print(f"Ensuring special tokens are in vocab: {special_token_values}")
    for special_token in special_token_values:
        if special_token not in vocab:
            vocab[special_token] = len(vocab)
            print(f"Added special token: {special_token}")

    if max_vocab_size > 0:
        assert len(vocab) <= max_vocab_size

    return vocab, token_counts


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
    """
    print("Creating metadata...")
    
    vocab_size = tokenizer.vocab_size
    
    # Create itos (integer to string mapping)
    itos = []
    for i in tqdm(range(vocab_size), desc="Creating itos mapping"):
        try:
            token = tokenizer.decode([i])
            itos.append(token)
        except:
            itos.append(f"<unk_{i}>")  # Fallback for any issues
    
    # Create stoi (string to integer mapping)
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"Metadata saved to {output_path}")
    print(f"Vocabulary size: {vocab_size}")
    
    return meta


# %%
def prepare_simplestories_dataset(
    tokenizer_name="word_level", 
    vocab_size=0, 
    output_subdir="word",
    model_name=None
):
    """
    Main function to prepare simplestories dataset
    
    Args:
        tokenizer_name (str): Type of tokenizer to use. 
                            Options: "word_level", "word_level_pmod", "byte_pair", "word_piece", "huggingface"
        vocab_size (int): Vocabulary size (0 for full vocabulary)
        output_subdir (str): Name of output subdirectory (default: "word")
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer type
    """
    print(f"Preparing simplestories dataset with {tokenizer_name} tokenizer")
    print(f"Vocab size: {vocab_size} (0 = full vocab)")
    
    # Set up paths
    data_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(data_dir, "simplestories.parquet")
    
    # Create output directory
    output_dir = os.path.join(data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")
    
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("Loading simplestories dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    stories_df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(stories_df)} stories")
    
    # Split data first to get the stories for vocab building
    train_stories = stories_df[stories_df['split'] == 'train']['story'].tolist()
    val_stories = stories_df[stories_df['split'] == 'test']['story'].tolist()  # Note: using 'test' for val
    
    # If no test split exists, use 'val' split
    if len(val_stories) == 0:
        val_stories = stories_df[stories_df['split'] == 'val']['story'].tolist()
    
    print(f"Train stories: {len(train_stories)}")
    print(f"Validation stories: {len(val_stories)}")
    
    # Get tokenizer using the tokenizer.py function
    print(f"Initializing tokenizer: {tokenizer_name}")
    if tokenizer_name == "huggingface":
        if model_name is None:
            raise ValueError("model_name must be provided when using 'huggingface' tokenizer")
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="simple_stories",  # Use simple_stories dataset name
            vocab_size=vocab_size,
            built_vocab=False,  # Will download/build vocab as needed
            model_name=model_name
        )
    elif tokenizer_name in ("word_level", "word_level_pmod"):
        # For word-level tokenizers, we need to build vocab from our local data
        # since the tokenizer.py doesn't have our specific simplestories.parquet file
        from tokenizer import get_word_level_tokenizer, build_vocab_from_data, get_regex_pattern, turn_list_of_stories_into_string, SPECIAL_TOKENS
        
        print("Building vocabulary from local simplestories data...")
        
        # For testing, let's use a smaller subset first
        # Comment out these lines to use full dataset
        print("Using subset for testing - processing first 50k stories")
        train_subset = train_stories[:40000]  # First 40k train stories
        val_subset = val_stories[:10000]      # First 10k val stories
        all_stories = train_subset + val_subset
        
        # Uncomment these lines to use full dataset:
        # all_stories = train_stories + val_stories
        
        print(f"Total stories to process: {len(all_stories)}")
        
        # Get the regex pattern for tokenization
        separate_possessive = tokenizer_name == "word_level_pmod"
        pattern = get_regex_pattern(seperate_possesive=separate_possessive)
        
        # Build vocabulary with progress bar (memory efficient)
        vocab, token_counts = build_vocab_from_stories_with_progress(all_stories, pattern, vocab_size)
        
        # Ensure special tokens are in vocab
        special_tokens_added = []
        for special_token in SPECIAL_TOKENS.values():
            if special_token not in vocab:
                vocab[special_token] = len(vocab)
                special_tokens_added.append(special_token)
        
        print(f"Built vocabulary with {len(vocab)} tokens")
        print(f"Special tokens: {list(set(SPECIAL_TOKENS.values()))}")
        if special_tokens_added:
            print(f"Added missing special tokens: {special_tokens_added}")
        
        # Create the tokenizer
        tokenizer = get_word_level_tokenizer(vocab, seperate_possesive=separate_possessive)
    else:
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="simple_stories",  # Use simple_stories dataset name
            vocab_size=vocab_size,
            built_vocab=False  # Will download/build vocab as needed
        )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Use the same subset for encoding that we used for vocabulary building
    if tokenizer_name in ("word_level", "word_level_pmod"):
        # We used subsets for vocab building, so use them for encoding too
        encoding_train_stories = train_subset if 'train_subset' in locals() else train_stories
        encoding_val_stories = val_subset if 'val_subset' in locals() else val_stories
    else:
        encoding_train_stories = train_stories
        encoding_val_stories = val_stories
    
    # Encode and save binary files
    print(f"\nEncoding training data ({len(encoding_train_stories)} stories)...")
    train_tokens = encode_stories_to_binary(encoding_train_stories, tokenizer, train_bin_path)
    
    print(f"\nEncoding validation data ({len(encoding_val_stories)} stories)...")
    val_tokens = encode_stories_to_binary(encoding_val_stories, tokenizer, val_bin_path)
    
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
def create_simplestories_dataset(
    tokenizer_name="word_level",
    vocab_size=0,
    output_subdir="word",
    model_name=None
):
    """
    Create simplestories dataset with specified tokenizer
    
    Args:
        tokenizer_name (str): Type of tokenizer to use (default: "word_level")
        vocab_size (int): Vocabulary size (0 for full vocabulary)
        output_subdir (str): Name of output subdirectory (default: "word")
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer
    """
    result = prepare_simplestories_dataset(
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        output_subdir=output_subdir,
        model_name=model_name
    )
    print(f"Outputed meta and binary files to {result['output_dir']}")
    print(f"Amount of train tokens: {result['train_tokens']}")
    print(f"Amount of val tokens: {result['val_tokens']}")
    print(f"Vocabulary size: {result['vocab_size']}")


def create_dataset_with_huggingface_tokenizer(model_name, output_subdir="word_hf"):
    """
    Convenience function to create dataset with any HuggingFace tokenizer
    
    Args:
        model_name (str): HuggingFace model name (e.g., "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-125M")
        output_subdir (str): Name of output subdirectory (default: "word_hf")
    """
    return create_simplestories_dataset(
        tokenizer_name="huggingface", 
        model_name=model_name,
        output_subdir=output_subdir
    )


# Example usage:
# create_simplestories_dataset()  # Use default word_level tokenizer, full vocab, output to "word" folder
# create_simplestories_dataset("word_level_pmod", output_subdir="word_pmod")  # Use possessive-modified tokenizer
# create_simplestories_dataset("byte_pair", output_subdir="byte_pair")  # Use byte_pair tokenizer
# create_dataset_with_huggingface_tokenizer("EleutherAI/gpt-neo-125M", "gpt_neo")  # Use GPT-Neo tokenizer

# %%
if __name__ == "__main__":
    # Create simplestories dataset with word_level tokenizer, full vocab, in "word" subfolder
    create_simplestories_dataset()