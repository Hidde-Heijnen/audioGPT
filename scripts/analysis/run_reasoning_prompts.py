# %%
"""
Run inference on reasoning prompts using a trained model and save results to CSV.
Notebook-style script for interactive execution.
"""

import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime
from contextlib import nullcontext
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model import GPTConfig, GPT

# %%
ts_prompts = [
  'Alice was so tired when she got back home so she went',
  'Samuel and Lily saw a rainbow after a rainy day. They were amazed by the colours. Samuel said, "Look, Lily. A rainbow has',
  'Samuel and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was',
  'Samuel wanted to read a book, so he went to',
  '"Can cows fly?", Alice asked her mother. ',
  '"What do birds like to eat?", Tom asked his mother.',
  '"What language do they speak in France?", Tom asked his mother',
  'If I throw a ball up in the air, eventually it will',
  'It was winter and cold outside so his mother told him, "You should',
  'Lily likes cats and dogs. She asked her mum for a dog and her mum said no, so instead she asked',
  'Samuel told Alice, "If you give me your banana, I\'ll give you my apple". Alice gave Samuel her banana so',
  'On weekends Samuel went to visit his grandmother whereas on weekdays he would go to school. Last weekend, when Samuel was on his way to',
  'Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that',
  'Lily and Ben are having an argument. They are trying to decide between the park and the swimming pool. Ben says, "I want to go to the park". Lily says',
  'Samuel\'s mother was not home, and his father was at home. When Samuel came home, he said hello to',
  'Lily doesn\'t like swimming. When her father wants to take her to the swimming pool, she says',
  'Both Ben and Lily wanted cake. Father said that there was only one piece of cake left. They',
  'Ben went to visit Lily in her house, but she was not at home. Ben knocked on the door',
  '"Hi Samuel, have you seen Alice? I can\'t find her anywhere", said Samuel.',
  'Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, "I see you have a brown dog. I also have',
  'Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Anne\'s mum asked her, "Anne, what is that you have in your left pocket?"',
  'Alice had both an apple and a carrot in her bag. She took the apple out of the bag and gave it to Samuel. She reached into the bag again and took',
  'Alice and Samuel walked up the street and met a girl in a red dress. The girl said to them, "Hi, I\'m Lily. What are your names?"',
  'Lily was hungry, and wanted to bake a cake, but she didn\'t have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, "Would you happen'
]

cm_prompts = [
    'I have a red ball and a blue ball. I give the red ball to Alice, so she plays with the red ball, and I play with the',
    'There are two boxes: one big and one small. The big box is too heavy for Kim, so she carries the',
    'The soup was too hot to eat, but the ice cream was too cold. Maria wanted something warm, so she chose the',
    'When the sun goes down, it becomes dark. When the sun comes up, it becomes',
    'Leo had two cookies. He ate one cookie, so now he has',
    'Rita is cold. The window was open, so she',
    'It was raining, so Tom took his umbrella. It was sunny, so he wore his',
    'The cow says "moo" and the cat says "meow". The dog says',
    'Kim was very thirsty, so she drank some water. Tim was very hungry, so he ate some',
    'There were three birds on the tree. Two birds flew away, so now there is',
    'Alice was lost, she wanted to go home. She met a stranger and asked him if he'
]

# %%
# Configuration
import os
base_dir = '/home/hrah2/rds/hpc-work/audioGPT'  # Absolute path to project root

CONFIG = {
    'checkpoint_path': os.path.join(base_dir, 'out/camstories_10k/small/ckpt.pt'),
    'data_dir': os.path.join(base_dir, 'data/camstories_10k/ss_tokenized'),
    'output_path': os.path.join(base_dir, 'results/camstories_small_reasoning.csv'),
    'tokenizer_name': 'SimpleStories/SimpleStories-30M',
    'special_tokens_path': os.path.join(base_dir, 'scripts/utils/special_tokens_map.json'),
    'max_new_tokens': 100,
    'temperature': 0.0,
    'top_k': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# %%
# Helper functions
def load_special_tokens(file_path: str):
    """Load special tokens from a JSON file."""
    with open(file_path, 'r') as f:
        special_tokens_map = json.load(f)
    return {
        "bos_token": special_tokens_map["bos_token"]["content"],
        "eos_token": special_tokens_map["eos_token"]["content"],
        "unk_token": special_tokens_map["unk_token"]["content"],
    }

def setup_tokenizer(tokenizer_name: str, special_tokens_path: str):
    """Setup tokenizer with special tokens."""
    special_tokens = load_special_tokens(special_tokens_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        bos_token=special_tokens["bos_token"],
        eos_token=special_tokens["eos_token"],
        unk_token=special_tokens["unk_token"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, special_tokens

def load_model(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    
    state_dict = checkpoint['model']
    # Remove unwanted prefix if present
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if device.startswith('cpu'):
        model = model.float()
    
    return model, checkpoint

def load_vocabulary_mapping(data_dir: str):
    """Load vocabulary mapping from meta.pkl if it exists."""
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta.get('stoi'), meta.get('itos'), meta.get('vocab_size')
    return None, None, None

# %%
# Setup device and precision
# Determine the effective device and autocast manager
requested_device_str = CONFIG['device']
device = requested_device_str # This will be the effective device, potentially overridden below
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_manager = nullcontext() # Default to no autocasting, used for CPU

print(f"Requested device from CONFIG: {requested_device_str}")

if device_type == 'cuda':
    if torch.cuda.is_available():
        selected_dtype_str = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[selected_dtype_str]
        autocast_manager = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        print(f"Effective device: {device} (CUDA). Autocast dtype for CUDA: {selected_dtype_str}.")
    else:
        print(f"Warning: CUDA device '{requested_device_str}' specified in CONFIG but CUDA not available. Falling back to CPU.")
        device = 'cpu' # Override effective device
        device_type = 'cpu' # Update device_type accordingly
        # autocast_manager remains nullcontext()
        print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")
elif device_type == 'cpu':
    # autocast_manager remains nullcontext()
    print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")
else: # Should not be reached
    print(f"Warning: Unknown device type for '{requested_device_str}'. Defaulting to CPU.")
    device = 'cpu'
    device_type = 'cpu'
    print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")

# The variable `device` now holds the true effective device string (e.g. 'cpu' or 'cuda:0')
# The variable `autocast_manager` holds the appropriate context manager for inference.

# %%
# Load model
print(f"Loading model from {CONFIG['checkpoint_path']}...")
model, checkpoint = load_model(CONFIG['checkpoint_path'], device) # Pass the effective device
print(f"Model loaded successfully!")
print(f"Model config: {model.config}")
print(f"Vocab size: {model.config.vocab_size}")

# %%
# Setup tokenizer
print(f"Setting up tokenizer: {CONFIG['tokenizer_name']}")
tokenizer, special_tokens = setup_tokenizer(CONFIG['tokenizer_name'], CONFIG['special_tokens_path'])
print(f"Tokenizer loaded: {type(tokenizer).__name__}")
print(f"Special tokens: {special_tokens}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# %%
# Load vocabulary mapping if available
stoi, itos, vocab_size = load_vocabulary_mapping(CONFIG['data_dir'])
if stoi is not None:
    print(f"Loaded vocabulary mapping from meta.pkl")
    print(f"Meta vocab_size: {vocab_size}")
    print(f"Using character-level encoding")
else:
    print("No meta.pkl found, using tokenizer encoding")

# %%
# Encoding/Decoding functions
def encode_prompt(prompt: str, tokenizer):
    """Encode a prompt using the tokenizer without trailing eos token."""
    # Tokenizer encoding (do not add special tokens to avoid EOS at end)
    return tokenizer.encode(prompt, add_special_tokens=False)

def decode_tokens(tokens: list, tokenizer):
    """Decode tokens to text using the tokenizer."""
    # Tokenizer decoding with proper cleanup
    return tokenizer.decode(
        tokens, 
        skip_special_tokens=True,  # Skip special tokens like <bos>, <eos>
        clean_up_tokenization_spaces=True  # Properly handle subword tokens and spacing
    )

def decode_tokens_raw(tokens: list, tokenizer):
    """Decode tokens to text without cleanup (for debugging)."""
    # Raw tokenizer decoding without cleanup
    return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

def generate_completion(
    model, 
    prompt: str, 
    tokenizer, 
    device: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200
):
    """Generate completion for a single prompt."""
    # Encode prompt
    input_ids = encode_prompt(prompt, tokenizer)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Process generated sequence, stopping at special tokens
    seq_ids = y[0].tolist()
    prompt_length = len(input_ids)
    # extract generated ids after prompt
    gen_ids = seq_ids[prompt_length:]
    # identify special token ids to stop at
    special_ids = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id}
    special_ids.discard(None)
    # find first special token position
    stop_pos = next((i for i, tok in enumerate(gen_ids) if tok in special_ids), len(gen_ids))
    # truncate generated ids
    gen_ids = gen_ids[:stop_pos]
    # trim full sequence to include only prompt + truncated generation
    full_ids = seq_ids[:prompt_length + stop_pos]
    full_text = decode_tokens(full_ids, tokenizer)
    # decode generated tokens
    generated_text = decode_tokens(gen_ids, tokenizer)
    # remove any occurrence of 'once upon a time' (case-insensitive) and everything after it
    generated_text = re.split(r'(?i)once upon a time', generated_text)[0].rstrip()
    
    return full_text, generated_text

# %%
# Test with a single prompt
test_prompt = "Alice was so tired when she got back home so she went"
print(f"Testing with prompt: '{test_prompt}'")

with autocast_manager:
    full_text, generated_text = generate_completion(
        model=model,
        prompt=test_prompt,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=CONFIG['max_new_tokens'],
        temperature=CONFIG['temperature'],
        top_k=CONFIG['top_k']
    )

print(f"\nFull text: {full_text}")
print(f"\nGenerated: {generated_text}")

# %%
# Prepare all prompts
all_prompts = [
    ('ts', i, prompt) for i, prompt in enumerate(ts_prompts)
] + [
    ('cm', i, prompt) for i, prompt in enumerate(cm_prompts)
]

print(f"Total prompts to process: {len(all_prompts)}")
print(f"TS prompts: {len(ts_prompts)}")
print(f"CM prompts: {len(cm_prompts)}")

# Show first few prompts
print("\nFirst few prompts:")
for i, (prompt_set, idx, prompt) in enumerate(all_prompts[:3]):
    print(f"{i+1}. [{prompt_set}_{idx}] {prompt[:80]}...")

# %%
# Run inference on all prompts
results = []

print(f"\nProcessing {len(all_prompts)} prompts...")

with autocast_manager:
    for prompt_set, idx, prompt in tqdm(all_prompts, desc="Generating completions"):
        try:
            full_text, generated_text = generate_completion(
                model=model,
                prompt=prompt,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                top_k=CONFIG['top_k']
            )
            
            results.append({
                'prompt_set': prompt_set,
                'prompt_idx': idx,
                'prompt': prompt,
                'generated': generated_text,
                'full_text': full_text,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"\nError processing prompt {prompt_set}_{idx}: {str(e)}")
            results.append({
                'prompt_set': prompt_set,
                'prompt_idx': idx,
                'prompt': prompt,
                'generated': f"ERROR: {str(e)}",
                'full_text': f"ERROR: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })

print(f"\nCompleted processing all prompts!")

# %%
# Save results and show summary
df = pd.DataFrame(results)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(CONFIG['output_path']) if os.path.dirname(CONFIG['output_path']) else '.', exist_ok=True)

# Save to CSV
df.to_csv(CONFIG['output_path'], index=False)
print(f"\nResults saved to {CONFIG['output_path']}")

# Print summary
print(f"\nSummary:")
print(f"Total prompts processed: {len(results)}")
print(f"Successful completions: {sum(1 for r in results if not r['generated'].startswith('ERROR'))}")
print(f"Errors: {sum(1 for r in results if r['generated'].startswith('ERROR'))}")

# Show some examples
print(f"\nExample results:")
for i, row in df.head(3).iterrows():
    print(f"\n{i+1}. [{row['prompt_set']}_{row['prompt_idx']}]")
    print(f"Prompt: {row['prompt']}")
    print(f"Generated: {row['generated'][:100]}...")

# %%
# Display results by prompt set
print("\nResults by prompt set:")
print(df.groupby('prompt_set').size())

# Show any errors
error_count = sum(1 for r in results if r['generated'].startswith('ERROR'))
if error_count > 0:
    print(f"\nErrors found: {error_count}")
    error_df = df[df['generated'].str.startswith('ERROR')]
    print(error_df[['prompt_set', 'prompt_idx', 'generated']])
else:
    print("\nNo errors found!")

print("\nInference complete! 🎉") 