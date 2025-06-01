#!/usr/bin/env python3
"""
Test script to demonstrate proper tokenizer decoding vs raw decoding.
Shows how to fix ## subword token artifacts.
"""

from transformers import AutoTokenizer

def test_tokenizer_decoding():
    """Test different decoding methods with a tokenizer."""
    
    # Example text that might produce subword tokens
    test_texts = [
        "##ac##lethathermightmakehersp##ic##ul##if##low##erfact.sheeventriedtok##icktheda##is##ywiththehelpof",
        "The quick brown fox jumps over the lazy dog.",
        "Hello world! How are you today?"
    ]
    
    # Try with SimpleStories tokenizer (same as your script)
    tokenizer_name = 'SimpleStories/SimpleStories-30M'
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Using tokenizer: {tokenizer_name}")
        print(f"Tokenizer type: {type(tokenizer).__name__}")
        print("=" * 80)
        
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: {text[:50]}...")
            
            # If the text already has ## tokens, show different decoding approaches
            if "##" in text:
                print("Raw text with ## tokens:")
                print(f"  {text}")
                
                # Manual cleanup (remove ## tokens)
                manual_clean = text.replace("##", "")
                print(f"Manual cleanup (remove ##): {manual_clean}")
                
                # Try to tokenize the cleaned text and decode properly
                if len(manual_clean) > 0:
                    tokens = tokenizer.encode(manual_clean, add_special_tokens=False)
                    
                    # Different decoding methods
                    decoded_clean = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    decoded_raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    
                    print(f"Proper decode (clean): {decoded_clean}")
                    print(f"Raw decode: {decoded_raw}")
                
            else:
                # Normal tokenization test
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Different decoding methods
                decoded_clean = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_raw = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                
                print(f"Original: {text}")
                print(f"Tokens: {tokens}")
                print(f"Proper decode: {decoded_clean}")
                print(f"Raw decode: {decoded_raw}")
            
            print("-" * 60)
    
    except Exception as e:
        print(f"Error with tokenizer {tokenizer_name}: {e}")
        print("This might be because the tokenizer isn't available or requires special setup.")

def manual_fix_subword_tokens(text):
    """Manually fix subword token artifacts."""
    import re
    
    if not text:
        return text
    
    # Remove ## prefixes (WordPiece/BERT-style subword tokens)
    cleaned = text.replace('##', '')
    
    # Try to add spaces at likely word boundaries
    # This is a heuristic approach and may not be perfect
    
    # Add space before common English words
    common_words = ['the', 'that', 'her', 'might', 'make', 'she', 'even', 'tried', 'to', 'with', 'help', 'of', 'and', 'a', 'an']
    
    for word in common_words:
        # Look for word boundaries (preceding lowercase letter)
        pattern = f'([a-z]){re.escape(word)}([a-z]|$)'
        cleaned = re.sub(pattern, rf'\1 {word} \2', cleaned)
    
    # Clean up multiple spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

if __name__ == "__main__":
    print("Testing tokenizer decoding methods...")
    test_tokenizer_decoding()
    
    print("\n" + "=" * 80)
    print("Testing manual subword token fixing...")
    
    test_text = "##ac##lethathermightmakehersp##ic##ul##if##low##erfact.sheeventriedtok##icktheda##is##ywiththehelpof"
    
    print(f"Original: {test_text}")
    print(f"Manual fix: {manual_fix_subword_tokens(test_text)}") 