# prepare_word.py
import os
import json
import re
from collections import Counter
from datasets import load_dataset
import numpy as np
import pickle

# --- Configuration ---
# DATA_DIR is the directory where this script (prepare_word.py) is located.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SPECIAL_TOKENS_MAP_FILE = os.path.join(DATA_DIR, "special_tokens_map.json") # Path to special_tokens_map.json

RAW_TRAIN_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
RAW_VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

# Output files for word tokenization
WORD_COUNTS_FILE = os.path.join(DATA_DIR, "tinystories_word_counts.json")
TRAIN_BIN_FILE = os.path.join(DATA_DIR, "train_word.bin")
VALID_BIN_FILE = os.path.join(DATA_DIR, "val_word.bin")
META_FILE = os.path.join(DATA_DIR, "meta_word.pkl")

# Special tokens - These will be overridden by special_tokens_map.json if present
# UNK_TOKEN = "<UNK>"
# EOS_TOKEN = "<EOS>"
# BOS_TOKEN = "<BOS>" # Optional, nanoGPT doesn't strictly require it for its typical setup
# SPECIAL_TOKENS = [UNK_TOKEN, EOS_TOKEN] # Add BOS_TOKEN here if you decide to use it

# --- Helper Functions ---

def download_data():
    """Downloads TinyStories dataset and saves as raw text files if not present."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading 'roneneldan/TinyStories' dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)

    if not os.path.exists(RAW_TRAIN_FILE):
        print(f"Writing train data to {RAW_TRAIN_FILE}...")
        with open(RAW_TRAIN_FILE, "w", encoding="utf-8") as f:
            for example in dataset["train"]:
                f.write(example["text"] + "\n") # Each story on a new line
        print("Train data written.")
    else:
        print(f"{RAW_TRAIN_FILE} already exists.")

    if not os.path.exists(RAW_VALID_FILE):
        print(f"Writing validation data to {RAW_VALID_FILE}...")
        with open(RAW_VALID_FILE, "w", encoding="utf-8") as f:
            for example in dataset["validation"]:
                f.write(example["text"] + "\n") # Each story on a new line
        print("Validation data written.")
    else:
        print(f"{RAW_VALID_FILE} already exists.")

def word_tokenize_line(line_text):
    """
    Tokenizes a line of text into words and punctuation.
    Converts to lowercase.
    """
    line_text = line_text.strip().lower()
    if not line_text:
        return []
    # \w+ matches sequences of word characters (letters, numbers, underscore)
    # [^\s\w] matches any character that is not whitespace and not a word character (i.e., punctuation)
    tokens = re.findall(r"\w+|[^\s\w]", line_text)
    return tokens

def process_dataset():
    """
    Builds vocabulary from training data, saves word counts,
    encodes train/validation sets, and saves metadata.
    """

    # --- Load Special Tokens ---
    print(f"Attempting to load special tokens from {SPECIAL_TOKENS_MAP_FILE}...")
    loaded_special_tokens_map = {}
    if os.path.exists(SPECIAL_TOKENS_MAP_FILE):
        with open(SPECIAL_TOKENS_MAP_FILE, 'r', encoding='utf-8') as f:
            loaded_special_tokens_map = json.load(f)
        print(f"Successfully loaded special tokens: {list(loaded_special_tokens_map.keys())}")
    else:
        print(f"Warning: {SPECIAL_TOKENS_MAP_FILE} not found. Using default special token values.")

    # Define special tokens based on the loaded map or defaults
    # The order in SPECIAL_TOKENS list matters for vocab indexing if they are the first ones added.
    # We'll prioritize standard ones like UNK, EOS, BOS if they are to be used.
    
    DEFAULT_UNK_CONTENT = "<UNK>"
    DEFAULT_EOS_CONTENT = "<EOS>"
    DEFAULT_BOS_CONTENT = "<BOS>" # Only used if explicitly added to SPECIAL_TOKENS list later

    UNK_TOKEN_CONTENT = loaded_special_tokens_map.get("unk_token", {}).get("content", DEFAULT_UNK_CONTENT)
    EOS_TOKEN_CONTENT = loaded_special_tokens_map.get("eos_token", {}).get("content", DEFAULT_EOS_CONTENT)
    
    # Decide which special tokens will be actively used and added to the vocab first.
    # For this script, UNK and EOS are primary. BOS can be added if needed.
    SPECIAL_TOKENS_LIST = [UNK_TOKEN_CONTENT, EOS_TOKEN_CONTENT]

    # If bos_token is defined in the map and you want to use it, add it.
    # For example, if you plan to prepend BOS to sequences:
    if "bos_token" in loaded_special_tokens_map and loaded_special_tokens_map["bos_token"].get("content"):
        BOS_TOKEN_CONTENT = loaded_special_tokens_map["bos_token"]["content"]
        if BOS_TOKEN_CONTENT not in SPECIAL_TOKENS_LIST: # Avoid duplicates if content is same as UNK/EOS
             # Decide if BOS should be part of the initial SPECIAL_TOKENS_LIST
             # For now, we are not adding it to SPECIAL_TOKENS_LIST to maintain current script's primary use of UNK/EOS
             # but it will be available in meta.pkl if defined in special_tokens_map.json
             pass # print(f"BOS token ('{BOS_TOKEN_CONTENT}') loaded but not added to active SPECIAL_TOKENS_LIST for vocab.")
    else:
        # BOS_TOKEN_CONTENT = DEFAULT_BOS_CONTENT # It's already default
        pass


    # --- Step 1: Build Vocabulary from Training Data & Save Word Counts ---
    print(f"\nStep 1: Building vocabulary and counting word frequencies from {RAW_TRAIN_FILE}...")
    word_counts = Counter()
    num_train_stories = 0
    with open(RAW_TRAIN_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            tokens = word_tokenize_line(line)
            if tokens:
                word_counts.update(tokens)
                num_train_stories += 1
            if (i + 1) % 50000 == 0:
                print(f"  Processed {i+1} lines for word counting...")
    print(f"Found {len(word_counts)} unique word/punctuation tokens in {num_train_stories} training stories.")

    # Save all unique tokens and their frequencies
    sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
    with open(WORD_COUNTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_word_counts, f, indent=2, ensure_ascii=False)
    print(f"Word counts saved to {WORD_COUNTS_FILE}")

    # Create vocabulary (stoi, itos) including special tokens
    # Special tokens come first
    stoi = {token: i for i, token in enumerate(SPECIAL_TOKENS_LIST)}
    itos = list(SPECIAL_TOKENS_LIST)
    
    # Add words from training data
    # Option: Limit vocabulary size here if desired, e.g., by taking top N from sorted_word_counts
    # For now, using all unique words from training + special tokens
    for word in sorted_word_counts.keys(): # Iterating in frequency order (though set order is not guaranteed for `in`)
        if word not in stoi:
            itos.append(word)
            stoi[word] = len(itos) - 1
            
    vocab_size = len(itos)
    print(f"Vocabulary size (including special tokens {SPECIAL_TOKENS_LIST}): {vocab_size}")
    
    unk_token_id = stoi[UNK_TOKEN_CONTENT]
    eos_token_id = stoi[EOS_TOKEN_CONTENT]

    # --- Step 2: Encode Train and Validation Data ---
    def encode_file(input_filepath, output_filepath, current_stoi, current_eos_id, current_unk_id):
        print(f"Encoding {input_filepath}...")
        all_token_ids = []
        num_stories = 0
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                story_tokens = word_tokenize_line(line)
                if not story_tokens:
                    continue
                
                story_ids = [current_stoi.get(token, current_unk_id) for token in story_tokens]
                story_ids.append(current_eos_id) # Append EOS to each story
                all_token_ids.extend(story_ids)
                num_stories +=1
                if (i + 1) % 50000 == 0:
                    print(f"  Encoded {i+1} lines from {input_filepath}...")
        
        token_ids_np = np.array(all_token_ids, dtype=np.uint16)
        print(f"Finished encoding {input_filepath}. Total stories: {num_stories}, Total tokens: {len(token_ids_np)}")
        token_ids_np.tofile(output_filepath)
        print(f"Saved tokenized data to {output_filepath}.")
        return len(token_ids_np), num_stories

    train_tokens_count, train_stories_count = encode_file(RAW_TRAIN_FILE, TRAIN_BIN_FILE, stoi, eos_token_id, unk_token_id)
    val_tokens_count, val_stories_count = encode_file(RAW_VALID_FILE, VALID_BIN_FILE, stoi, eos_token_id, unk_token_id)

    # --- Step 3: Save Metadata (compatible with nanoGPT train.py) ---
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
        'special_tokens_map': loaded_special_tokens_map # Store the loaded special tokens map
    }
    with open(META_FILE, 'wb') as f: # Save as pickle binary
        pickle.dump(meta, f)
    print(f"Metadata saved to {META_FILE}")
    # Updated print examples to use dynamic content
    if UNK_TOKEN_CONTENT in stoi and EOS_TOKEN_CONTENT in stoi:
        print(f"  stoi example: {{'{UNK_TOKEN_CONTENT}': {stoi[UNK_TOKEN_CONTENT]}, '{EOS_TOKEN_CONTENT}': {stoi[EOS_TOKEN_CONTENT]}, ...}}")
    else:
        print("  stoi example: UNK/EOS tokens not found in stoi for example print.")
    print(f"  itos example: {itos[:min(len(itos), 3)]} ...")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Starting TinyStories Word-Level Data Preparation...")

    print("\nStep A: Downloading raw dataset files (if needed)...")
    download_data()

    print("\nStep B: Processing dataset (token counts, encoding, metadata)...")
    process_dataset()

    print("\nData preparation complete.")
    print(f"Output files are in: {DATA_DIR}")
    print(f"  Word counts from training data: {WORD_COUNTS_FILE}")
    print(f"  Train tokens (.bin): {TRAIN_BIN_FILE}")
    print(f"  Valid tokens (.bin): {VALID_BIN_FILE}")
    print(f"  Metadata (.pkl): {META_FILE}")

if __name__ == '__main__':
    main()