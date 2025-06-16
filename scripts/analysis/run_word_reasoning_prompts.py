# %%
"""
Run inference on reasoning prompts using word tokenizer models and save results to CSV.
Generates completions from 5 different model sizes.
Notebook-style script for interactive execution.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from contextlib import nullcontext
import torch
from tqdm import tqdm
import re

# Add parent directory to path to import model and tokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model import GPTConfig, GPT

# Add tokenizers directory to path
tokenizers_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tokenizers')
sys.path.append(tokenizers_dir)
from tokenizer import get_tokenizer

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
base_dir = '/home/hrah2/rds/hpc-work/audioGPT'  # Absolute path to project root

CONFIG = {
    'model_paths': {
        'tiny': os.path.join(base_dir, 'out/camstories_10k/word_tiny/ckpt.pt'),
        'small': os.path.join(base_dir, 'out/camstories_10k/word_small/ckpt.pt'),
        'medium': os.path.join(base_dir, 'out/camstories_10k/word_medium/ckpt.pt'),
        'frozen_medium': os.path.join(base_dir, 'out/camstories_10k/word_frozen_medium/ckpt.pt'),
        'large': os.path.join(base_dir, 'out/camstories_10k/word_large/ckpt.pt')
    },
    'vocab_csv_path': os.path.join(base_dir, 'data/camstories_10k/word_tokenized_v1/vocab.csv'),
    'output_path': os.path.join(base_dir, 'results/word_models_reasoning.csv'),
    'max_new_tokens': 100,
    'temperature': 0.0,
    'top_k': 200,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print("Configuration:")
for key, value in CONFIG.items():
    if key == 'model_paths':
        print(f"  {key}:")
        for model_name, path in value.items():
            print(f"    {model_name}: {path}")
    else:
        print(f"  {key}: {value}")

# %%
# Setup device and precision
requested_device_str = CONFIG['device']
device = requested_device_str
device_type = 'cuda' if 'cuda' in device else 'cpu'
autocast_manager = nullcontext()

print(f"Requested device from CONFIG: {requested_device_str}")

if device_type == 'cuda':
    if torch.cuda.is_available():
        selected_dtype_str = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
        ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[selected_dtype_str]
        autocast_manager = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        print(f"Effective device: {device} (CUDA). Autocast dtype for CUDA: {selected_dtype_str}.")
    else:
        print(f"Warning: CUDA device '{requested_device_str}' specified in CONFIG but CUDA not available. Falling back to CPU.")
        device = 'cpu'
        device_type = 'cpu'
        print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")
elif device_type == 'cpu':
    print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")
else:
    print(f"Warning: Unknown device type for '{requested_device_str}'. Defaulting to CPU.")
    device = 'cpu'
    device_type = 'cpu'
    print(f"Effective device: {device} (CPU). Model will be cast to float32. No autocast.")

# %%
# Load word tokenizer
print("Loading word tokenizer...")

# Load vocabulary directly from the existing CSV file
from tokenizer import load_vocab_from_csv, get_word_level_tokenizer

vocab_csv_path = CONFIG['vocab_csv_path']
print(f"Loading vocabulary from: {vocab_csv_path}")

if not os.path.exists(vocab_csv_path):
    raise FileNotFoundError(f"Vocabulary file not found: {vocab_csv_path}")

vocab = load_vocab_from_csv(vocab_csv_path)
print(f"Loaded vocabulary with {len(vocab)} tokens")

# Create the word-level tokenizer with the loaded vocabulary
tokenizer = get_word_level_tokenizer(vocab)

print(f"Tokenizer loaded: {type(tokenizer).__name__}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"Special tokens: {tokenizer.special_tokens_map}")

# %%
# Helper functions
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
    
    return model

def encode_prompt(prompt: str, tokenizer):
    """Encode a prompt using the tokenizer."""
    # Add special tokens=False to avoid adding EOS at the end
    return tokenizer.encode(prompt, add_special_tokens=False)

def decode_tokens(tokens: list, tokenizer):
    """Decode tokens to text using the tokenizer."""
    return tokenizer.decode(
        tokens, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

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
    
    # Process generated sequence
    seq_ids = y[0].tolist()
    prompt_length = len(input_ids)
    gen_ids = seq_ids[prompt_length:]
    
    # Stop at special tokens
    special_ids = {tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id}
    special_ids.discard(None)
    stop_pos = next((i for i, tok in enumerate(gen_ids) if tok in special_ids), len(gen_ids))
    gen_ids = gen_ids[:stop_pos]
    
    # Decode generated tokens
    generated_text = decode_tokens(gen_ids, tokenizer)
    
    # Remove 'once upon a time' pattern
    generated_text = re.split(r'(?i)once upon a time', generated_text)[0].rstrip()
    
    return generated_text

# %%
# Load all models
print("\nLoading models...")
models = {}
for model_name, model_path in CONFIG['model_paths'].items():
    print(f"Loading {model_name} model from {model_path}...")
    try:
        models[model_name] = load_model(model_path, device)
        print(f"  {model_name} model loaded successfully!")
    except Exception as e:
        print(f"  Error loading {model_name} model: {str(e)}")
        models[model_name] = None

# %%
# Test with a single prompt
test_prompt = "Alice was so tired when she got back home so she went"
print(f"\nTesting with prompt: '{test_prompt}'")

for model_name, model in models.items():
    if model is not None:
        with autocast_manager:
            generated = generate_completion(
                model=model,
                prompt=test_prompt,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=CONFIG['max_new_tokens'],
                temperature=CONFIG['temperature'],
                top_k=CONFIG['top_k']
            )
        print(f"\n{model_name}: {generated}")

# %%
# Prepare all prompts
all_prompts = ts_prompts + cm_prompts
print(f"\nTotal prompts to process: {len(all_prompts)}")
print(f"TS prompts: {len(ts_prompts)}")
print(f"CM prompts: {len(cm_prompts)}")

# %%
# Generate completions for all prompts and models
results = []

print(f"\nProcessing {len(all_prompts)} prompts...")

with autocast_manager:
    for prompt in tqdm(all_prompts, desc="Generating completions"):
        row = {'prompt': prompt}
        
        # Generate completion for each model
        for model_name, model in models.items():
            if model is not None:
                try:
                    generated = generate_completion(
                        model=model,
                        prompt=prompt,
                        tokenizer=tokenizer,
                        device=device,
                        max_new_tokens=CONFIG['max_new_tokens'],
                        temperature=CONFIG['temperature'],
                        top_k=CONFIG['top_k']
                    )
                    row[model_name] = generated
                except Exception as e:
                    print(f"\nError with {model_name} on prompt '{prompt[:50]}...': {str(e)}")
                    row[model_name] = f"ERROR: {str(e)}"
            else:
                row[model_name] = "MODEL_NOT_LOADED"
        
        results.append(row)

print(f"\nCompleted processing all prompts!")

# %%
# Save results to CSV
df = pd.DataFrame(results)

# Reorder columns to have prompt first, then models in order
column_order = ['prompt', 'tiny', 'small', 'medium', 'frozen_medium', 'large']
df = df[column_order]

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(CONFIG['output_path']) if os.path.dirname(CONFIG['output_path']) else '.', exist_ok=True)

# Save to CSV
df.to_csv(CONFIG['output_path'], index=False)
print(f"\nResults saved to {CONFIG['output_path']}")

# %%
# Display summary
print(f"\nSummary:")
print(f"Total prompts processed: {len(results)}")

# Count successful completions per model
for model_name in ['tiny', 'small', 'medium', 'frozen_medium', 'large']:
    if model_name in df.columns:
        successful = df[model_name].apply(lambda x: not (x.startswith('ERROR') or x == 'MODEL_NOT_LOADED')).sum()
        print(f"{model_name}: {successful}/{len(df)} successful completions")

# %%
# Show sample results
print("\nSample results (first 3 prompts):")
print(df.head(3).to_string(max_colwidth=50))

# %%
# Show any errors
error_summary = {}
for model_name in ['tiny', 'small', 'medium', 'frozen_medium', 'large']:
    if model_name in df.columns:
        errors = df[df[model_name].str.startswith('ERROR', na=False)]
        not_loaded = df[df[model_name] == 'MODEL_NOT_LOADED']
        
        if len(errors) > 0 or len(not_loaded) > 0:
            error_summary[model_name] = {
                'errors': len(errors),
                'not_loaded': len(not_loaded)
            }

if error_summary:
    print("\nError summary:")
    for model_name, counts in error_summary.items():
        print(f"{model_name}: {counts['errors']} errors, {counts['not_loaded']} not loaded")
else:
    print("\nNo errors found!")

print("\nInference complete! 🎉") 