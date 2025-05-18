import os
import json
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import sys
import pickle # Added for .pkl output

# Path and import changes for prepare.py moved into data/tinystories
from tokenizer import Tokenizer # Import directly as tokenizer.py is in the same directory

# DATA_DIR is the directory where this script (prepare.py) is located.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_TRAIN_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT3-train-og.txt")
RAW_VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT3-valid-og.txt")
TOKEN_COUNTS_FILE = os.path.join(DATA_DIR, "tinystories_token_counts.json")
SPECIAL_TOKENS_MAP_FILE = os.path.join(DATA_DIR, "special_tokens_map.json")
TOKENIZER_CONFIG_NAME = "EleutherAI/gpt-neo-125M" # Changed from "gpt2"
TOP_K = 10000

TRAIN_BIN_FILE = os.path.join(DATA_DIR, "train-og.bin")
VALID_BIN_FILE = os.path.join(DATA_DIR, "val-og.bin")
META_FILE = os.path.join(DATA_DIR, "meta.pkl") # Changed from meta.json to meta.pkl

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


def encode_file_with_custom_tokenizer(filepath, tokenizer_instance, output_path):
    all_token_ids = []
    print(f"Tokenizing {filepath} with custom tokenizer logic...")

    # Determine the OOV token index from the custom tokenizer (typically EOS's new index)
    # The tokenizer.py now ensures eos_token_id_str is in top_k_tokens_dict if k is used.
    eos_token_id_str_from_base = str(tokenizer_instance.tokenizer.eos_token_id)
    oov_token_new_idx = tokenizer_instance.top_k_tokens_dict.get(eos_token_id_str_from_base)
    if oov_token_new_idx is None:
        if tokenizer_instance.top_k_tokens_dict: # If dict is not empty but EOS is missing (should not happen with new tokenizer.py)
            # Fallback to the last token in the custom vocab as OOV if EOS is unexpectedly not there
            oov_token_new_idx = len(tokenizer_instance.top_k_tokens_dict) - 1 
            print(f"Warning: EOS token ID {eos_token_id_str_from_base} not found in custom vocab map. Using {oov_token_new_idx} as OOV index.")
        elif tokenizer_instance.k : # k is active but dict is empty (error)
             raise ValueError("Critical: Custom tokenizer's top_k_tokens_dict is empty when k is active.")
        # if not k, then oov_token_new_idx won't be used by the mapping loop below if not self.k in tokenizer.encoder

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Tokenize using the base tokenizer. tokenizer.py's encoder now defaults to add_special_tokens=True.
            # This call simulates what tokenizer_instance.encoder() would do for the first part.
            encoded_output = tokenizer_instance.tokenizer(
                line,
                add_special_tokens=True, # Let base tokenizer add BOS/EOS
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True, # Recommended to add truncation and max_length matching training
                max_length=tokenizer_instance.tokenizer.model_max_length
            )
            original_ids = encoded_output['input_ids']

            new_ids_for_line = []
            if tokenizer_instance.k: # Mapping to k-limited vocab is only if k is set
                for token_id in original_ids:
                    # Map to new vocabulary, defaulting to the new OOV token index (EOS's new index)
                    mapped_id = tokenizer_instance.top_k_tokens_dict.get(str(token_id), oov_token_new_idx)
                    new_ids_for_line.append(mapped_id)
            else: # If not k-limited, use original_ids directly
                new_ids_for_line.extend(original_ids)
            
            # Add the new OOV/EOS token ID at the end of each story's tokens
            # new_ids_for_line.append(oov_token_new_idx) # No longer needed, add_special_tokens=True handles EOS.
            
            all_token_ids.extend(new_ids_for_line)

            if (line_num + 1) % 10000 == 0:
                print(f"  Processed {line_num + 1} lines from {filepath} for final encoding...")
    
    print(f"Finished tokenizing {filepath}. Total tokens for .bin file: {len(all_token_ids)}")
    token_ids_np = np.array(all_token_ids, dtype=np.uint16)
    print(f"Saving tokenized data to {output_path}...")
    token_ids_np.tofile(output_path)
    print(f"Saved to {output_path}.")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Step 1: Loading special tokens...")
    if not os.path.exists(SPECIAL_TOKENS_MAP_FILE):
        print(f"Error: Special tokens map file not found at {SPECIAL_TOKENS_MAP_FILE}")
        print("Please ensure the file exists and contains the necessary token definitions.")
        # Example: {"bos_token": {"content": "<|endoftext|>"}, ...}
        # For now, creating a dummy one if it's missing, for the script to proceed.
        # This part should ideally be handled by the user ensuring the file is present.
        print(f"Attempting to create a dummy {SPECIAL_TOKENS_MAP_FILE} for demonstration if it is missing.")
        # Check if the directory exists before creating the file
        os.makedirs(os.path.dirname(SPECIAL_TOKENS_MAP_FILE), exist_ok=True)
        dummy_special_tokens = {
            "bos_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "eos_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
            "unk_token": {"content": "<|endoftext|>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False}
        }
        with open(SPECIAL_TOKENS_MAP_FILE, 'w') as f_dummy:
            json.dump(dummy_special_tokens, f_dummy, indent=2)
        print(f"Dummy {SPECIAL_TOKENS_MAP_FILE} created. Please verify its contents.")
        
    special_tokens = load_special_tokens(SPECIAL_TOKENS_MAP_FILE)
    print(f"Special tokens loaded: {special_tokens}")

    print("\\nStep 2: Downloading raw dataset files...")
    download_data()

    print("\\nStep 3: Generating token counts...")
    generate_token_counts(TOKENIZER_CONFIG_NAME, special_tokens)

    print(f"\\nStep 4: Initializing custom tokenizer with top {TOP_K} tokens...")
    class TokenizerConfig:
        def __init__(self, name):
            self.name = name
    custom_tokenizer_config = TokenizerConfig(TOKENIZER_CONFIG_NAME)
    
    # The custom Tokenizer from tokenizer.py will initialize its own AutoTokenizer
    # using `config.name`. It sets pad_token = eos_token.
    # It's assumed that for TOKENIZER_CONFIG_NAME (e.g., "gpt2"), the default special tokens
    # align with those in special_tokens_map.json (e.g., "<|endoftext|>").
    custom_tokenizer = Tokenizer(
        config=custom_tokenizer_config,
        k=TOP_K,
        file_path=TOKEN_COUNTS_FILE,
        device="cpu" # Can be "cuda" if available and preferred
    )
    print("Custom tokenizer initialized.")
    tokenizer_details = custom_tokenizer.get_config()
    print(f"Custom tokenizer details: {json.dumps(tokenizer_details, indent=2)}")


    print("\\nStep 5: Tokenizing train and validation data and saving as .bin files...")
    encode_file_with_custom_tokenizer(RAW_TRAIN_FILE, custom_tokenizer, TRAIN_BIN_FILE)
    encode_file_with_custom_tokenizer(RAW_VALID_FILE, custom_tokenizer, VALID_BIN_FILE)

    print("\\nStep 6: Saving metadata in nanoGPT format (meta.pkl)...")
    
    # vocab_size is already correctly calculated by our custom_tokenizer
    vocab_size = custom_tokenizer.vocab_size
    
    # Create itos (integer to string mapping)
    # custom_tokenizer.top_k_tokens is a list of original token ID strings,
    # ordered by their new vocabulary index.
    itos = [custom_tokenizer.tokenizer.decode([int(token_id_str)]) for token_id_str in custom_tokenizer.top_k_tokens]
    
    # Create stoi (string to integer mapping)
    stoi = {s: i for i, s in enumerate(itos)}
    
    meta_nano_gpt = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    
    with open(META_FILE, 'wb') as f: # Open in binary mode for pickle
        pickle.dump(meta_nano_gpt, f)
    print(f"nanoGPT metadata saved to {META_FILE}")
    print(f"  Vocabulary size: {vocab_size}")
    # Example of itos/stoi content for verification (first 5 and last 5)
    if vocab_size > 10:
        print(f"  itos (first 5):")
        for i, s_token in list(enumerate(itos))[:5]:
            print(f"    {i}: '{s_token}'")
        print(f"  itos (last 5):")
        for i, s_token in list(enumerate(itos))[-5:]:
            print(f"    {i}: '{s_token}'")
    else:
        print(f"  itos:")
        for i, s_token in enumerate(itos):
            print(f"    {i}: '{s_token}'")


    print("\\nData preparation complete.")
    print(f"Output files are in: {DATA_DIR}")
    print(f"  Token counts: {TOKEN_COUNTS_FILE}")
    print(f"  Train tokens: {TRAIN_BIN_FILE}")
    print(f"  Valid tokens: {VALID_BIN_FILE}")
    print(f"  Metadata: {META_FILE}")

if __name__ == '__main__':
    main() 