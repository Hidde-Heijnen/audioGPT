import os
import sys
import json
import pickle
import argparse
from typing import List, Dict, Tuple, Optional

import torch
import pandas as pd
from tqdm import tqdm

# Ensure project root is on sys.path to import local modules when running from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model import GPTConfig, GPT
from tokenizer import get_word_level_tokenizer


PUNCTUATION_CHARS = {".", "!", "?", ";"}


def read_prompts(file_path: str) -> List[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    # Filter out empty lines while preserving order
    return [ln for ln in lines if ln.strip()]


def load_checkpoint(out_dir: str, device: torch.device) -> Tuple[Optional[GPT], Optional[dict]]:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Missing checkpoint: {ckpt_path}")
        return None, None
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])  # type: ignore[index]
    model = GPT(gptconf)

    # Clean unwanted prefixes in state dict
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if device.type == "cpu":
        model.float()

    return model, checkpoint


def build_word_tokenizer_from_meta(checkpoint: dict) -> Optional[object]:
    """Create a word-level tokenizer (our custom wrapper) using meta.pkl vocabulary from the dataset.

    This ensures IDs exactly match the model's embedding IDs.
    """
    dataset = checkpoint.get("config", {}).get("dataset")
    if not dataset:
        print("[WARN] No dataset found in checkpoint config; cannot locate meta.pkl for tokenizer.")
        return None

    meta_path = os.path.join("data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        print(f"[WARN] meta.pkl not found at {meta_path}")
        return None

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    if not ("stoi" in meta and isinstance(meta["stoi"], dict)):
        print("[WARN] meta.pkl does not contain 'stoi' mapping; cannot build word-level tokenizer")
        return None

    vocab: Dict[str, int] = meta["stoi"]
    # Our helper expects a token->index mapping; 'stoi' is already token->id
    tokenizer = get_word_level_tokenizer(vocab)
    return tokenizer


@torch.no_grad()
def greedy_until_punct_custom(
    model: GPT,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    # Lowercase per requirement for custom models
    prompt_proc = prompt.lower()
    x = tokenizer(prompt_proc, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)

    original_len = x.size(1)
    tokens_generated = 0

    for _ in range(max_new_tokens):
        # Respect block size by cropping from the left if needed
        block_size = model.config.block_size
        idx_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        # temperature=0 => greedy
        next_id = torch.argmax(logits, dim=-1, keepdim=True)  # [B,1]
        x = torch.cat((x, next_id), dim=1)
        tokens_generated += 1

        # Check last generated token for punctuation
        # Only stop if this is NOT the first token and contains punctuation
        last_id = next_id.item()
        last_text = tokenizer.decode([last_id], skip_special_tokens=True)
        if tokens_generated > 1 and any(ch in last_text for ch in PUNCTUATION_CHARS):
            break

    new_tokens = x[0, original_len:].tolist()
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return completion


@torch.no_grad()
def greedy_until_punct_gpt2(
    hf_tokenizer,
    hf_model,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 20,
) -> str:
    # Do not lowercase for GPT-2
    # Use generate method with greedy decoding to avoid numerical issues
    input_ids = hf_tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    
    # Generate tokens one by one until punctuation
    generated_ids = input_ids.clone()
    original_len = input_ids.size(1)
    tokens_generated = 0
    
    for _ in range(max_new_tokens):
        outputs = hf_model.generate(
            generated_ids,
            max_new_tokens=1,
            do_sample=False,  # Greedy
            pad_token_id=hf_tokenizer.eos_token_id,
            eos_token_id=None,  # Don't stop on EOS
            return_dict_in_generate=True,
            output_scores=False
        )
        
        # Get the newly generated token
        new_token_id = outputs.sequences[0, -1].item()
        generated_ids = outputs.sequences
        tokens_generated += 1
        
        # Check if it contains punctuation
        # Only stop if this is NOT the first token and contains punctuation
        new_token_text = hf_tokenizer.decode([new_token_id], skip_special_tokens=True)
        if tokens_generated > 1 and any(ch in new_token_text for ch in PUNCTUATION_CHARS):
            break
    
    # Extract just the completion
    new_tokens = generated_ids[0, original_len:].tolist()
    completion = hf_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return completion


def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Complete reasoning prompts with multiple models (temperature=0, stop at punctuation)")
    parser.add_argument("--prompts", default="scripts/reasoning-tests/reasoning-prompts.txt", type=str)
    parser.add_argument("--output", default="out/reasoning_prompts_completions.csv", type=str)
    parser.add_argument("--max_new_tokens", default=20, type=int)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), type=str)
    args = parser.parse_args()

    device = torch.device(args.device)

    prompts = read_prompts(args.prompts)
    model_dirs = [
        "out/dataset_tests/simplestories_word",
        "out/dataset_tests/camstories_10k_word_run1",
        "out/dataset_tests/tinystories_rope",
        "out/dataset_tests/tinystories_word",
        "out/dataset_tests/camstories_10k_15m_8L6H384",
    ]
    model_colnames = [
        "simplestories_word",
        "camstories_10k_word_run1",
        "tinystories_rope",
        "tinystories_word",
        "camstories_10k_15m_8L6H384",
        "gpt2",
    ]

    # Load custom checkpoints + tokenizers
    custom_models: Dict[str, Tuple[GPT, object]] = {}
    for mdir in model_dirs:
        model, checkpoint = load_checkpoint(mdir, device)
        if model is None or checkpoint is None:
            continue
        tokenizer = build_word_tokenizer_from_meta(checkpoint)
        if tokenizer is None:
            print(f"[WARN] Skipping {mdir} due to tokenizer build failure")
            continue
        custom_models[mdir] = (model, tokenizer)

    # Load GPT-2 medium from Hugging Face
    from transformers import GPT2Tokenizer, GPT2LMHeadModel  # Imported here to avoid global dependency if unused

    try:
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Add pad token for GPT-2 (it doesn't have one by default)
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_model.eval().to(device)
    except Exception as e:
        print(f"[WARN] Failed to load gpt2-medium: {e}")
        gpt2_tokenizer, gpt2_model = None, None

    rows = []
    ensure_dir(args.output)

    for prompt in tqdm(prompts, desc="Processing prompts", unit="prompt"):
        row: Dict[str, str] = {"prompt": prompt}

        # Custom local models
        for mdir, colname in zip(model_dirs, model_colnames):
            if colname == "gpt2":
                # will fill later
                continue
            if mdir not in custom_models:
                row[colname] = ""
                continue
            model, tok = custom_models[mdir]
            try:
                completion = greedy_until_punct_custom(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                completion = f"[error: {str(e)[:100]}]"
            row[colname] = completion

        # GPT-2 (no lowercasing)
        if gpt2_tokenizer is not None and gpt2_model is not None:
            try:
                gpt2_completion = greedy_until_punct_gpt2(
                    hf_tokenizer=gpt2_tokenizer,
                    hf_model=gpt2_model,
                    prompt=prompt,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as e:
                gpt2_completion = f"[error: {str(e)[:100]}]"
        else:
            gpt2_completion = ""
        row["gpt2"] = gpt2_completion

        rows.append(row)

    # Order columns: prompt first, then models
    cols = ["prompt"] + model_colnames
    # If some custom model failed to load, ensure column exists
    for col in cols:
        if col != "prompt":
            for r in rows:
                if col not in r:
                    r[col] = ""

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()


