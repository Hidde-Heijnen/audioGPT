#!/usr/bin/env python3
"""
Text cleaner utility to process generated text with tokenization artifacts.
Removes ## concat tokens and adds proper spacing.
"""

import re

def clean_generated_text(text):
    """
    Clean generated text by removing ## tokens and adding proper spacing.
    
    Args:
        text (str): Raw generated text with ## tokens
        
    Returns:
        str: Cleaned text with proper spacing
    """
    if not text:
        return text
    
    # Remove ## tokens
    cleaned = text.replace('##', '')
    
    # Add spaces between words where needed
    # This regex finds transitions from lowercase to uppercase, or letter to punctuation
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', cleaned)
    spaced = re.sub(r'([a-z])([.!?])', r'\1\2', spaced)
    spaced = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', spaced)
    
    # Handle common word boundaries
    spaced = re.sub(r'([a-z])([a-z]+)([a-z])([A-Z])', r'\1\2\3 \4', spaced)
    
    # Fix specific patterns found in the text
    spaced = re.sub(r'(\w)(\w+)(might|that|her|the|with|and|she|even|tried|to)', r'\1\2 \3', spaced)
    spaced = re.sub(r'(might|that|her|the|with|and|she|even|tried|to)(\w)', r'\1 \2', spaced)
    
    # Clean up multiple spaces
    spaced = re.sub(r'\s+', ' ', spaced)
    
    return spaced.strip()

def clean_text_advanced(text):
    """
    More sophisticated cleaning using common English word patterns.
    """
    if not text:
        return text
    
    # Remove ## tokens
    cleaned = text.replace('##', '')
    
    # Common English words to help with word boundary detection
    common_words = [
        'the', 'that', 'her', 'might', 'make', 'special', 'flower', 'fact',
        'she', 'even', 'tried', 'to', 'kick', 'daisy', 'with', 'help', 'of',
        'and', 'a', 'an', 'is', 'was', 'were', 'are', 'have', 'has', 'had',
        'will', 'would', 'could', 'should', 'can', 'may', 'might'
    ]
    
    # Try to identify word boundaries by looking for common word patterns
    result = cleaned
    
    # Insert spaces before common words (except at start)
    for word in common_words:
        pattern = f'([a-z]){word}([a-z]|$)'
        result = re.sub(pattern, r'\1 ' + word + r' \2', result)
    
    # Fix punctuation spacing
    result = re.sub(r'([a-z])([.!?])', r'\1\2', result)
    result = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', result)
    
    # Clean up multiple spaces
    result = re.sub(r'\s+', ' ', result)
    
    return result.strip()

if __name__ == "__main__":
    # Example usage
    test_text = "##ac##lethathermightmakehersp##ic##ul##if##low##erfact.sheeventriedtok##icktheda##is##ywiththehelpof"
    
    print("Original text:")
    print(test_text)
    print()
    
    print("Basic cleaning:")
    basic_cleaned = clean_generated_text(test_text)
    print(basic_cleaned)
    print()
    
    print("Advanced cleaning:")
    advanced_cleaned = clean_text_advanced(test_text)
    print(advanced_cleaned)
    print()
    
    # Manual reconstruction for this specific case
    manual = test_text.replace('##', '')
    # Based on the pattern, this appears to be:
    manual_fixed = "able that her might make her special flower fact. she even tried to kick the daisy with the help of"
    print("Manual reconstruction:")
    print(manual_fixed) 