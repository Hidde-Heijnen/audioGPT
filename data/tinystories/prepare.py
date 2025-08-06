# %%
import os
import json
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import sys
import pickle
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add the workspace root to path to import tokenizer
workspace_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, workspace_root)
from tokenizer import get_tokenizer, Tokenizer

# DATA_DIR is the directory where this script (prepare.py) is located.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Legacy files for original custom tokenizer approach
RAW_TRAIN_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT3-train-og.txt")
RAW_VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT3-valid-og.txt")
TOKEN_COUNTS_FILE = os.path.join(DATA_DIR, "tinystories_token_counts.json")
SPECIAL_TOKENS_MAP_FILE = os.path.join(DATA_DIR, "special_tokens_map.json")
TOKENIZER_CONFIG_NAME = "EleutherAI/gpt-neo-125M"
TOP_K = 10000

# Legacy output files
TRAIN_BIN_FILE = os.path.join(DATA_DIR, "train-og.bin")
VALID_BIN_FILE = os.path.join(DATA_DIR, "val-og.bin")
META_FILE = os.path.join(DATA_DIR, "meta.pkl")

# %%
def build_vocab_from_stories_with_progress(stories: list, pattern: str, max_vocab_size: int):
    """
    Build vocabulary from stories list - process all at once
    """
    import regex as re
    from collections import Counter
    
    print("Processing all stories to build vocabulary...")
    
    # Process all stories at once
    all_tokens = []
    for story in tqdm(stories, desc="Tokenizing stories"):
        story_tokens = re.findall(pattern, story.lower())
        all_tokens.extend(story_tokens)
    
    # Count tokens
    print("Counting token frequencies...")
    token_counts = Counter(all_tokens)
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
    
    for story in tqdm(stories, desc="Encoding stories"):
        # Tokenize the story with special tokens
        token_ids = tokenizer(
            story, 
            padding=False,
            truncation=True,
            max_length=1024,
            return_tensors='pt',
            add_special_tokens=True  # This adds BOS and EOS tokens
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
    
    # Get the actual vocabulary size from the tokenizer
    base_vocab_size = tokenizer.vocab_size
    
    # For some tokenizers, special tokens are added outside base vocab
    # We need to include them in the metadata
    try:
        max_token_id = max(tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)
        actual_vocab_size = max(base_vocab_size, max_token_id + 1)
    except:
        actual_vocab_size = base_vocab_size
    
    print(f"Base vocab size: {base_vocab_size}")
    print(f"Actual vocab size for metadata: {actual_vocab_size}")
    
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
def load_special_tokens(file_path):
    with open(file_path, 'r') as f:
        special_tokens_map = json.load(f)
    return {
        "bos_token": special_tokens_map["bos_token"]["content"],
        "eos_token": special_tokens_map["eos_token"]["content"],
        "unk_token": special_tokens_map["unk_token"]["content"],
    }

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading 'roneneldan/TinyStories' dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True) # Added trust_remote_code

    if not os.path.exists(RAW_TRAIN_FILE):
        print(f"Writing train data to {RAW_TRAIN_FILE}...")
        with open(RAW_TRAIN_FILE, "w", encoding="utf-8") as f:
            for example in dataset["train"]:
                f.write(example["text"] + "\\n") # Each story on a new line
        print("Train data written.")
    else:
        print(f"{RAW_TRAIN_FILE} already exists.")

    if not os.path.exists(RAW_VALID_FILE):
        print(f"Writing validation data to {RAW_VALID_FILE}...")
        with open(RAW_VALID_FILE, "w", encoding="utf-8") as f:
            for example in dataset["validation"]:
                f.write(example["text"] + "\\n") # Each story on a new line
        print("Validation data written.")
    else:
        print(f"{RAW_VALID_FILE} already exists.")

def generate_token_counts(base_tokenizer_name, special_tokens_dict):
    if os.path.exists(TOKEN_COUNTS_FILE):
        print(f"{TOKEN_COUNTS_FILE} already exists. Skipping generation.")
        return

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

    print(f"Reading training data from {RAW_TRAIN_FILE} to generate token counts...")
    # Reading file line by line to handle potentially very large files
    token_ids_list = []
    with open(RAW_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i + 1) % 10000 == 0:
                print(f"  Tokenizing line {i+1} for counting...")
            line_content = line.strip()
            if line_content: # Ensure not empty line
                encoded = base_tokenizer(line_content, add_special_tokens=False) # No extra special tokens for counting
                token_ids_list.extend(encoded['input_ids'])
    
    print(f"Finished tokenizing for counting. Total tokens found: {len(token_ids_list)}")
    print("Counting token frequencies...")
    counts = Counter(token_ids_list)

    # Sort by frequency (most common first), then by token_id (for tie-breaking)
    sorted_token_counts_list = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    
    # The custom tokenizer expects keys to be strings, and dict preserves insertion order (Python 3.7+)
    string_keyed_sorted_token_counts = {str(token_id): count for token_id, count in sorted_token_counts_list}

    print(f"Saving token counts to {TOKEN_COUNTS_FILE}...")
    with open(TOKEN_COUNTS_FILE, 'w') as f:
        json.dump(string_keyed_sorted_token_counts, f, indent=2)
    print("Token counts saved.")


# %%
def prepare_tinystories_dataset(
    tokenizer_name="word_level", 
    vocab_size=0, 
    output_subdir="word",
    model_name=None,
    use_parquet=True
):
    """
    Unified function to prepare tinystories dataset with any tokenizer
    
    Args:
        tokenizer_name (str): Type of tokenizer to use. 
                            Options: "word_level", "word_level_pmod", "byte_pair", "word_piece", "huggingface", "custom"
        vocab_size (int): Vocabulary size (0 for full vocabulary)
        output_subdir (str): Name of output subdirectory (default: "word")
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer type
        use_parquet (bool): Use tinystories.parquet if available, otherwise download from HuggingFace
    """
    print(f"Preparing tinystories dataset with {tokenizer_name} tokenizer")
    print(f"Vocab size: {vocab_size} (0 = full vocab)")
    
    # Set up paths
    data_dir = DATA_DIR
    dataset_path = os.path.join(data_dir, "tinystories.parquet")
    
    # Create output directory
    output_dir = os.path.join(data_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output files
    train_bin_path = os.path.join(output_dir, "train.bin")
    val_bin_path = os.path.join(output_dir, "val.bin")
    meta_path = os.path.join(output_dir, "meta.pkl")
    
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("Loading tinystories dataset...")
    if use_parquet and os.path.exists(dataset_path):
        print(f"Loading from parquet: {dataset_path}")
        stories_df = pd.read_parquet(dataset_path, engine='fastparquet')
        train_stories = stories_df[stories_df['split'] == 'train']['story'].tolist()
        val_stories = stories_df[stories_df['split'] == 'val']['story'].tolist()
    else:
        print("Loading from HuggingFace dataset...")
        dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
        train_stories = [example["text"] for example in dataset["train"]]
        val_stories = [example["text"] for example in dataset["validation"]]
    
    print(f"Train stories: {len(train_stories)}")
    print(f"Validation stories: {len(val_stories)}")
    
    # Get tokenizer based on type
    if tokenizer_name == "custom":
        # Use the original custom tokenizer approach
        return prepare_with_custom_tokenizer(train_stories, val_stories, output_dir)
    
    # Use the unified tokenizer from tokenizer.py
    print(f"Initializing tokenizer: {tokenizer_name}")
    if tokenizer_name == "huggingface":
        if model_name is None:
            raise ValueError("model_name must be provided when using 'huggingface' tokenizer")
        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_name,
            dataset_name="tiny_stories",
            vocab_size=vocab_size,
            built_vocab=False,
            model_name=model_name
        )
    elif tokenizer_name in ("word_level", "word_level_pmod"):
        # For word-level tokenizers, we need to build vocab from our local data
        from tokenizer import get_word_level_tokenizer, build_vocab_from_data, get_regex_pattern, turn_list_of_stories_into_string, SPECIAL_TOKENS
        
        print("Building vocabulary from local tinystories data...")
        
        # Use full dataset
        all_stories = train_stories + val_stories
        print(f"Total stories to process: {len(all_stories)}")
        
        # Get the regex pattern for tokenization
        separate_possessive = tokenizer_name == "word_level_pmod"
        pattern = get_regex_pattern(seperate_possesive=separate_possessive)
        
        # Build vocabulary with progress bar
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
            dataset_name="tiny_stories",
            vocab_size=vocab_size,
            built_vocab=False
        )
    
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
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
def prepare_with_custom_tokenizer(train_stories, val_stories, output_dir):
    """
    Legacy function for custom tokenizer approach (original prepare.py behavior)
    """
    print("Using custom tokenizer approach...")
    
    # Save stories to text files for custom tokenizer
    train_file = os.path.join(output_dir, "train_temp.txt")
    val_file = os.path.join(output_dir, "val_temp.txt")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for story in train_stories:
            f.write(story + '\n')
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for story in val_stories:
            f.write(story + '\n')
    
    # Use original custom tokenizer logic
    special_tokens = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>"
    }
    
    class TokenizerConfig:
        def __init__(self, name):
            self.name = name
    
    custom_tokenizer_config = TokenizerConfig(TOKENIZER_CONFIG_NAME)
    custom_tokenizer = Tokenizer(
        config=custom_tokenizer_config,
        k=TOP_K,
        file_path=None,  # Will build from data
        device="cpu"
    )
    
    # Encode files with custom tokenizer
    train_bin = os.path.join(output_dir, "train.bin")
    val_bin = os.path.join(output_dir, "val.bin")
    
    encode_file_with_custom_tokenizer(train_file, custom_tokenizer, train_bin)
    encode_file_with_custom_tokenizer(val_file, custom_tokenizer, val_bin)
    
    # Create metadata
    vocab_size = custom_tokenizer.vocab_size
    itos = [custom_tokenizer.tokenizer.decode([int(token_id_str)]) for token_id_str in custom_tokenizer.top_k_tokens]
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    meta_path = os.path.join(output_dir, "meta.pkl")
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    
    # Clean up temp files
    os.remove(train_file)
    os.remove(val_file)
    
    print(f"Custom tokenizer dataset complete!")
    return {
        'train_tokens': len(itos),  # Approximate
        'val_tokens': len(itos),    # Approximate
        'vocab_size': vocab_size,
        'output_dir': output_dir
    }


# %%
def encode_file_with_custom_tokenizer(filepath, tokenizer_instance, output_path):
    all_token_ids = []
    print(f"Tokenizing {filepath} with custom tokenizer logic...")

    # Determine the OOV token index from the custom tokenizer (typically EOS's new index)
    eos_token_id_str_from_base = str(tokenizer_instance.tokenizer.eos_token_id)
    oov_token_new_idx = tokenizer_instance.top_k_tokens_dict.get(eos_token_id_str_from_base)
    if oov_token_new_idx is None:
        if tokenizer_instance.top_k_tokens_dict:
            oov_token_new_idx = len(tokenizer_instance.top_k_tokens_dict) - 1 
            print(f"Warning: EOS token ID {eos_token_id_str_from_base} not found in custom vocab map. Using {oov_token_new_idx} as OOV index.")
        elif tokenizer_instance.k:
             raise ValueError("Critical: Custom tokenizer's top_k_tokens_dict is empty when k is active.")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            encoded_output = tokenizer_instance.tokenizer(
                line,
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=tokenizer_instance.tokenizer.model_max_length
            )
            original_ids = encoded_output['input_ids']

            new_ids_for_line = []
            if tokenizer_instance.k:
                for token_id in original_ids:
                    mapped_id = tokenizer_instance.top_k_tokens_dict.get(str(token_id), oov_token_new_idx)
                    new_ids_for_line.append(mapped_id)
            else:
                new_ids_for_line.extend(original_ids)
            
            all_token_ids.extend(new_ids_for_line)

            if (line_num + 1) % 10000 == 0:
                print(f"  Processed {line_num + 1} lines from {filepath} for final encoding...")
    
    print(f"Finished tokenizing {filepath}. Total tokens for .bin file: {len(all_token_ids)}")
    token_ids_np = np.array(all_token_ids, dtype=np.uint16)
    print(f"Saving tokenized data to {output_path}...")
    token_ids_np.tofile(output_path)
    print(f"Saved to {output_path}.")


# %%
def create_tinystories_dataset(
    tokenizer_name="word_level",
    vocab_size=0,
    output_subdir="word",
    model_name=None,
    use_parquet=True
):
    """
    Create tinystories dataset with specified tokenizer
    
    Args:
        tokenizer_name (str): Type of tokenizer to use (default: "word_level")
            Options: "word_level", "word_level_pmod", "byte_pair", "word_piece", "huggingface", "custom"
        vocab_size (int): Vocabulary size (0 for full vocabulary)
        output_subdir (str): Name of output subdirectory (default: "word")
        model_name (str, optional): HuggingFace model name for "huggingface" tokenizer
        use_parquet (bool): Use tinystories.parquet if available, otherwise download from HuggingFace
    """
    result = prepare_tinystories_dataset(
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
        output_subdir=output_subdir,
        model_name=model_name,
        use_parquet=use_parquet
    )
    print(f"Outputed meta and binary files to {result['output_dir']}")
    print(f"Amount of train tokens: {result['train_tokens']}")
    print(f"Amount of val tokens: {result['val_tokens']}")
    print(f"Vocabulary size: {result['vocab_size']}")
    return result


# %%
def create_dataset_with_huggingface_tokenizer(model_name, output_subdir="word_hf"):
    """
    Convenience function to create dataset with any HuggingFace tokenizer
    
    Args:
        model_name (str): HuggingFace model name (e.g., "microsoft/DialoGPT-medium", "EleutherAI/gpt-neo-125M")
        output_subdir (str): Name of output subdirectory (default: "word_hf")
    """
    return create_tinystories_dataset(
        tokenizer_name="huggingface", 
        model_name=model_name,
        output_subdir=output_subdir
    )


# %%
def prepare_simplestories_tokenizer():
    """
    Prepare dataset with SimpleStories tokenizer (vocab size 4096)
    """
    print("Preparing tinystories dataset with SimpleStories tokenizer")
    print("Vocab size: 4096")
    
    return create_tinystories_dataset(
        tokenizer_name="word_piece",
        vocab_size=4096,
        output_subdir="dataset_ss_tok"
    )


# %%
def main():
    """
    Main function - creates multiple datasets with different tokenizers
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("="*80)
    print("UNIFIED TINYSTORIES DATASET PREPARATION")
    print("="*80)
    print("This script can create datasets with different tokenizers:")
    print("1. word_level - Custom word-level tokenizer")
    print("2. word_level_pmod - Word-level with possessive modification")
    print("3. byte_pair - Byte-pair encoding (TinyStories)")
    print("4. word_piece - Word-piece (SimpleStories)")
    print("5. huggingface - Any HuggingFace tokenizer")
    print("6. custom - Original custom tokenizer approach")
    print("="*80)
    
    # Create default word-level dataset
    print("\n" + "="*60)
    print("Creating dataset with word_level tokenizer...")
    print("="*60)
    create_tinystories_dataset()
    
    # Create SimpleStories tokenized dataset
    print("\n" + "="*60)
    print("Creating dataset with SimpleStories tokenizer...")
    print("="*60)
    prepare_simplestories_tokenizer()
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE!")
    print("="*80)
    print("Available datasets in subdirectories:")
    print("- word/ (word_level tokenizer)")
    print("- dataset_ss_tok/ (SimpleStories tokenizer)")
    print("\nTo create additional datasets, use:")
    print("python prepare.py with different function calls")
    print("Examples:")
    print('  create_tinystories_dataset("byte_pair", output_subdir="byte_pair")')
    print('  create_dataset_with_huggingface_tokenizer("EleutherAI/gpt-neo-125M", "gpt_neo")')
    print('  create_tinystories_dataset("custom", output_subdir="custom_legacy")')


# Example usage functions for demonstration:
def example_usage():
    """
    Example usage patterns
    """
    # Use default word_level tokenizer, full vocab, output to "word" folder
    create_tinystories_dataset()
    
    # Use possessive-modified tokenizer
    create_tinystories_dataset("word_level_pmod", output_subdir="word_pmod")
    
    # Use byte_pair tokenizer
    create_tinystories_dataset("byte_pair", output_subdir="byte_pair")
    
    # Use GPT-Neo tokenizer
    create_dataset_with_huggingface_tokenizer("EleutherAI/gpt-neo-125M", "gpt_neo")
    
    # Use SimpleStories tokenizer with specific vocab size
    create_tinystories_dataset("word_piece", vocab_size=4096, output_subdir="simplestories_4k")
    
    # Use original custom tokenizer approach
    create_tinystories_dataset("custom", output_subdir="custom_legacy")

if __name__ == '__main__':
    main() 