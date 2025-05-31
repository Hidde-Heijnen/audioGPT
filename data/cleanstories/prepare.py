"""
Prepare the CleanStories dataset for language modeling using GPT-Neo tokenizer.
Will save train.bin, val.bin containing the token ids, and meta.pkl containing the
tokenizer info and other related metadata.
"""
import os
import pickle
import numpy as np
import json
from transformers import AutoTokenizer

# Set up paths
script_dir = os.path.dirname(__file__)
train_file_path = 'cleaned/lines/tinystories_train_gpt4_lines.txt'
val_file_path = 'cleaned/lines/tinystories_valid_gpt4_lines.txt'
special_tokens_map_path = os.path.join(script_dir, 'special_tokens_map.json')
output_dir = os.path.join(script_dir, 'neo')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load special tokens first
defined_special_tokens_content = {}
if os.path.exists(special_tokens_map_path):
    with open(special_tokens_map_path, 'r') as f:
        special_tokens_data = json.load(f)
    
    # Extract special token contents
    for token_name, token_spec in special_tokens_data.items():
        if isinstance(token_spec, dict) and 'content' in token_spec:
            content = token_spec['content']
            defined_special_tokens_content[token_name] = content
else:
    print(f"Warning: Special tokens map file not found at {special_tokens_map_path}")

# Load the tokenizer with custom special tokens
print("Loading GPT-Neo tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neo-125M",
    bos_token=defined_special_tokens_content.get('bos_token', None),
    eos_token=defined_special_tokens_content.get('eos_token', None),
    unk_token=defined_special_tokens_content.get('unk_token', None),
)

# Ensure the tokenizer has a pad token (needed for batch processing)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"BOS token: {tokenizer.bos_token}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"UNK token: {tokenizer.unk_token}")
print(f"PAD token: {tokenizer.pad_token}")
print(f"Tokenizer vocab size: {len(tokenizer)}")

def process_file(file_path, file_type):
    """Process a single file and return tokenized data using efficient batch processing."""
    print(f"Processing {file_type} file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Read {len(lines):,} lines from {file_type} file")
    
    # Process lines in batches for efficiency
    all_tokens = []
    batch_size = 1000
    
    for batch_start in range(0, len(lines), batch_size):
        batch_end = min(batch_start + batch_size, len(lines))
        batch_lines = []
        
        for i in range(batch_start, batch_end):
            line = lines[i].strip()
            if line:  # Skip empty lines
                batch_lines.append(line)
        
        if batch_lines:
            # Use tokenizer's built-in BOS/EOS addition with batch processing
            # add_special_tokens=True will automatically add BOS and EOS tokens
            batch_tokens = tokenizer(
                batch_lines,
                add_special_tokens=True,  # This adds BOS/EOS automatically
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=False,
                truncation=False
            )['input_ids']
            
            # Flatten the batch results
            for tokens in batch_tokens:
                all_tokens.extend(tokens)
        
        if batch_end % 10000 == 0:
            print(f"Processed {batch_end:,} lines...")
    
    print(f"{file_type} has {len(all_tokens):,} tokens")
    return all_tokens

# Process training data
train_tokens = process_file(train_file_path, "train")

# Process validation data
val_tokens = process_file(val_file_path, "val")

# Convert to numpy arrays
print("Converting to numpy arrays...")
train_ids = np.array(train_tokens, dtype=np.uint16)
val_ids = np.array(val_tokens, dtype=np.uint16)

# Check if any token IDs exceed uint16 range
max_token_id = max(max(train_tokens) if train_tokens else 0, max(val_tokens) if val_tokens else 0)
if max_token_id >= 65536:
    print(f"Warning: Max token ID {max_token_id} exceeds uint16 range. Using uint32 instead.")
    train_ids = np.array(train_tokens, dtype=np.uint32)
    val_ids = np.array(val_tokens, dtype=np.uint32)

# Save binary files
print("Saving binary files...")
train_ids.tofile(os.path.join(output_dir, 'train.bin'))
val_ids.tofile(os.path.join(output_dir, 'val.bin'))

# Save metadata
meta = {
    'vocab_size': len(tokenizer),
    'tokenizer_name': "EleutherAI/gpt-neo-125M",
    'special_tokens': defined_special_tokens_content,
    'bos_token': tokenizer.bos_token,
    'eos_token': tokenizer.eos_token,
    'unk_token': tokenizer.unk_token,
    'pad_token': tokenizer.pad_token,
    'dtype': str(train_ids.dtype),
}

with open(os.path.join(output_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Dataset preparation complete!")
print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")
print(f"Vocab size: {len(tokenizer):,}")
print(f"Files saved to: {output_dir}")
print(f"Data type: {train_ids.dtype}")
