#!/usr/bin/env python3
"""
Test script to demonstrate different tokenizer options.
This script shows how to use various tokenizers including HuggingFace models.
"""

import sys
import os

# Add the current directory to Python path to import tokenizer module
sys.path.append(os.path.dirname(__file__))

from tokenizer import get_tokenizer
from data.camstories.prepare import create_dataset, create_dataset_with_huggingface_tokenizer


def test_tokenizer(tokenizer_name, model_name=None):
    """Test a tokenizer with sample text."""
    print(f"\n{'='*50}")
    print(f"Testing {tokenizer_name} tokenizer")
    if model_name:
        print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    try:
        # Get the tokenizer
        if tokenizer_name == "huggingface":
            tokenizer = get_tokenizer(
                tokenizer_name=tokenizer_name,
                dataset_name="camstories",
                vocab_size=5000,
                model_name=model_name
            )
        else:
            tokenizer = get_tokenizer(
                tokenizer_name=tokenizer_name,
                dataset_name="camstories", 
                vocab_size=5000,
                built_vocab=False  # Don't require pre-built vocab for test
            )
        
        # Test with sample text
        sample_text = "Once upon a time, there was a little girl named Emma who loved to explore."
        
        # Tokenize
        tokens = tokenizer(sample_text, return_tensors="pt")
        
        print(f"Sample text: {sample_text}")
        print(f"Tokenized IDs: {tokens['input_ids'].tolist()[0]}")
        print(f"Number of tokens: {len(tokens['input_ids'][0])}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        
        # Decode back
        decoded = tokenizer.decode(tokens['input_ids'][0])
        print(f"Decoded: {decoded}")
        
        print("✅ Tokenizer test successful!")
        
    except Exception as e:
        print(f"❌ Error testing {tokenizer_name}: {e}")


def main():
    """Main function to test different tokenizers."""
    print("Testing different tokenizer options...")
    
    # Test HuggingFace tokenizers
    hf_models = [
        "EleutherAI/gpt-neo-125M",
        "microsoft/DialoGPT-medium", 
        "gpt2",
        "roneneldan/TinyStories"  # This should work since it's already used in byte_pair
    ]
    
    for model in hf_models:
        test_tokenizer("huggingface", model_name=model)
    
    # Test built-in tokenizers (these might fail if vocab files don't exist)
    print(f"\n{'='*50}")
    print("Testing built-in tokenizers (may fail if vocab files missing)")
    print(f"{'='*50}")
    
    try:
        test_tokenizer("byte_pair")
    except Exception as e:
        print(f"❌ byte_pair test failed: {e}")
    
    try:
        test_tokenizer("word_piece")
    except Exception as e:
        print(f"❌ word_piece test failed: {e}")


def demo_dataset_creation():
    """Demonstrate how to create datasets with different tokenizers."""
    print(f"\n{'='*60}")
    print("DATASET CREATION EXAMPLES")
    print(f"{'='*60}")
    
    print("\n# Example 1: Create dataset with GPT-Neo tokenizer")
    print('create_dataset_with_huggingface_tokenizer("camstories_5000", "EleutherAI/gpt-neo-125M")')
    
    print("\n# Example 2: Create dataset with DialoGPT tokenizer")
    print('create_dataset_with_huggingface_tokenizer("camstories_5000", "microsoft/DialoGPT-medium")')
    
    print("\n# Example 3: Create dataset with GPT-2 tokenizer")
    print('create_dataset("camstories_5000", tokenizer_name="huggingface", model_name="gpt2")')
    
    print("\n# Example 4: Create dataset with byte-pair tokenizer")
    print('create_dataset("camstories_5000", tokenizer_name="byte_pair")')
    
    print("\n# Example 5: Create dataset with default word-level tokenizer")
    print('create_dataset("camstories_5000_pmod")')


if __name__ == "__main__":
    main()
    demo_dataset_creation()
    
    print(f"\n{'='*60}")
    print("To actually create datasets, uncomment and run the desired commands in the examples above.")
    print("Make sure you have the required dataset files (camstories_5000.parquet, etc.)")
    print(f"{'='*60}") 