# %% analysis of the original data
# I want to do analysis on the words like how many different words there are and also comparisons between tran and validation. Punctuation is split from the words, but not for words like don't and i'm. I also want to 
# Do all for train, valid and total, unless it doesn't make sense. 
# %% imports
from collections import Counter
import re
import string # Added for string.punctuation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# %% These are big files, so we might want to read them in chunks. each line is a story.
TRAIN_FILE_INPUT = "cleaned/lines/tinystories_train_gpt4_lines.txt"
VALID_FILE_INPUT = "cleaned/lines/tinystories_valid_gpt4_lines.txt"

# %%
with open(TRAIN_FILE_INPUT, 'r') as file:
    train_lines = file.readlines()

with open(VALID_FILE_INPUT, 'r') as file:
    valid_lines = file.readlines()

# %% Check for duplicates in train and valid individually
train_duplicates = [line for line, count in Counter(train_lines).items() if count > 1]
valid_duplicates = [line for line, count in Counter(valid_lines).items() if count > 1]

print(f"Number of duplicate stories in train: {len(train_duplicates)}")
print(f"Number of duplicate stories in valid: {len(valid_duplicates)}")

if train_duplicates:
    print("Example train duplicate:", train_duplicates[0])
if valid_duplicates:
    print("Example valid duplicate:", valid_duplicates[0])

# %% Show train duplicates in a table
if train_duplicates:
    train_duplicates_df = pd.DataFrame(train_duplicates, columns=['Story'])
    print("\n--- Duplicate Stories in Training Set ---")
    print(train_duplicates_df)
else:
    print("\nNo duplicate stories found in training set.")

# %% Show validation duplicates in a table
if valid_duplicates:
    valid_duplicates_df = pd.DataFrame(valid_duplicates, columns=['Story'])
    print("\n--- Duplicate Stories in Validation Set ---")
    print(valid_duplicates_df)
else:
    print("\nNo duplicate stories found in validation set.")

# %% Check if there are any validation stories are in train
valid_in_train_count = 0
valid_set = set(valid_lines)
train_set = set(train_lines) # Use a set for faster lookups, though Counter could also be used if we want to remove duplicates from train first

for story in valid_set:
    if story in train_set:
        valid_in_train_count += 1

print(f"Number of validation stories also found in train: {valid_in_train_count}")

# %% How many stories start with Once upon a time? (train, validation, total) (number out of total, and percentage)
PREFIX = "once upon a time" # Use lowercase for case-insensitive comparison

train_prefix_count = 0
for story in train_lines:
    if story.lower().startswith(PREFIX):
        train_prefix_count += 1

valid_prefix_count = 0
for story in valid_lines:
    if story.lower().startswith(PREFIX):
        valid_prefix_count += 1

total_stories = len(train_lines) + len(valid_lines)
total_prefix_count = train_prefix_count + valid_prefix_count

# Calculate percentages
percent_train_prefix = (train_prefix_count / len(train_lines)) * 100 if len(train_lines) > 0 else 0
percent_valid_prefix = (valid_prefix_count / len(valid_lines)) * 100 if len(valid_lines) > 0 else 0
percent_total_prefix = (total_prefix_count / total_stories) * 100 if total_stories > 0 else 0

print(f"\n--- Stories Starting with '{PREFIX}' ---")
print(f"Train: {train_prefix_count} out of {len(train_lines)} ({percent_train_prefix:.2f}%)")
print(f"Valid: {valid_prefix_count} out of {len(valid_lines)} ({percent_valid_prefix:.2f}%)")
print(f"Total: {total_prefix_count} out of {total_stories} ({percent_total_prefix:.2f}%)")

# %% Create a file with the tokens (so word or punctuation) and the number of times it appears in train and valid, treat numbers as words as well

def _tokenize_line(line):
    """Helper function to tokenize a single line according to the project's rules."""
    # Regex to find words (including contractions like "don't") or any non-alphanumeric, non-whitespace character (as punctuation)
    # \w+(?:'\w+)* : matches words like "hello", "don't", "o'clock"
    # [^\w\s]       : matches single punctuation characters like ",", "!", "."
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", line.lower())

def get_tokens(lines):
    tokens = Counter()
    for line in lines:
        # Split by space, this will keep punctuation attached to words if not separated by space
        # The problem description says: "Punctuation is split from the words, but not for words like don't and i'm."
        # This implies the data is already pre-processed to handle this. If not, we'd need more complex tokenization.
        # Let's assume simple space splitting is enough based on the `cleanstories` name and prior processing.
        # A more robust tokenizer would use regex to handle various cases of punctuation and contractions.
        # For now, we convert to lowercase and split by space.
        # words = line.lower().split() # Old method
        token_list = _tokenize_line(line) # Use new helper
        tokens.update(token_list)
    return tokens

train_tokens = get_tokens(train_lines)
valid_tokens = get_tokens(valid_lines)

all_tokens_combined = train_tokens + valid_tokens

TOKEN_COUNTS_OUTPUT_FILE = "data/cleanstories/token_counts.txt"

with open(TOKEN_COUNTS_OUTPUT_FILE, 'w') as f:
    f.write("Token\tTrain_Count\tValid_Count\tTotal_Count\n")
    sorted_tokens = sorted(all_tokens_combined.keys())
    for token in sorted_tokens:
        train_count = train_tokens.get(token, 0)
        valid_count = valid_tokens.get(token, 0)
        total_count = all_tokens_combined.get(token, 0)
        f.write(f"{token}\t{train_count}\t{valid_count}\t{total_count}\n")

print(f"Token counts written to {TOKEN_COUNTS_OUTPUT_FILE}")
print(f"Total unique tokens in train: {len(train_tokens)}")
print(f"Total unique tokens in valid: {len(valid_tokens)}")
print(f"Total unique tokens overall: {len(all_tokens_combined)}")

# %% Type-Token Ratio (Vocabulary Richness)
total_token_count_train = sum(train_tokens.values())
total_token_count_valid = sum(valid_tokens.values())
total_token_count_combined = sum(all_tokens_combined.values())

if total_token_count_train > 0:
    ttr_train = len(train_tokens) / total_token_count_train
    print(f"Type-Token Ratio (TTR) for train: {ttr_train:.4f}")
else:
    print("Type-Token Ratio (TTR) for train: N/A (no tokens)")

if total_token_count_valid > 0:
    ttr_valid = len(valid_tokens) / total_token_count_valid
    print(f"Type-Token Ratio (TTR) for valid: {ttr_valid:.4f}")
else:
    print("Type-Token Ratio (TTR) for valid: N/A (no tokens)")

if total_token_count_combined > 0:
    ttr_combined = len(all_tokens_combined) / total_token_count_combined
    print(f"Type-Token Ratio (TTR) for overall: {ttr_combined:.4f}")
else:
    print("Type-Token Ratio (TTR) for overall: N/A (no tokens)")


# %% Story Length Analysis (Number of Tokens per Story)
# This uses the _tokenize_line helper defined earlier for consistent tokenization.

def get_story_lengths(lines):
    lengths = []
    for line in lines:
        story_tokens = _tokenize_line(line) # Uses the helper
        lengths.append(len(story_tokens))
    return lengths

train_story_lengths = get_story_lengths(train_lines)
valid_story_lengths = get_story_lengths(valid_lines)

# For median, import statistics module if desired e.g.
# import statistics

print(f"\n--- Story Length Analysis ---")
if train_story_lengths:
    avg_len_train = sum(train_story_lengths) / len(train_story_lengths)
    min_len_train = min(train_story_lengths)
    max_len_train = max(train_story_lengths)
    # median_len_train = statistics.median(train_story_lengths) # Example for median
    print(f"Train Stories ({len(train_story_lengths)}):")
    print(f"  Tokens per story: Avg={avg_len_train:.2f}, Min={min_len_train}, Max={max_len_train}")
else:
    print("Train Stories: No stories found to analyze length.")

if valid_story_lengths:
    avg_len_valid = sum(valid_story_lengths) / len(valid_story_lengths)
    min_len_valid = min(valid_story_lengths)
    max_len_valid = max(valid_story_lengths)
    # median_len_valid = statistics.median(valid_story_lengths) # Example for median
    print(f"Validation Stories ({len(valid_story_lengths)}):")
    print(f"  Tokens per story: Avg={avg_len_valid:.2f}, Min={min_len_valid}, Max={max_len_valid}")
else:
    print("Validation Stories: No stories found to analyze length.")

# %% Any weird words that fall outside normal words, simple letters, and punctuation?

# This regex aims to match common English words, numbers, basic punctuation, and contractions.
# It allows: 
#   - words with letters (a-z, A-Z)
#   - numbers (0-9)
#   - apostrophes within words (e.g., don't, i'm)
#   - common punctuation marks if they are standalone tokens or correctly attached (e.g., ., ,, !, ?)
#   - It will treat hyphenated words as single tokens if they are not split during initial tokenization.

# Given the problem description: "Punctuation is split from the words, but not for words like don\'t and i\'m."
# This means tokens like '.' or ',' should be separate. Our get_tokens function uses line.lower().split(),
# which might not separate all punctuation correctly if it's attached without a space.
# For example, "hello." would be one token "hello.".
# We will refine the tokenization if needed, but for now, let\'s assume tokens are reasonably well-formed.

# A simpler approach for "weird words" is to identify tokens that are NOT purely alphabetic, 
# NOT purely numeric, NOT common contractions, and NOT common standalone punctuation.

# Let\'s refine the regex to identify normal tokens more accurately.
# - ^[a-z]+$ : purely alphabetic words (after lowercasing)
# - ^[0-9]+(?:\\.[0-9]+)?$ : purely numeric 
# - ^[a-z]+\'[a-z]+$ : simple contractions like "don\'t", "i\'m"
# - ^[\\.,!?;:\"]+$ : common standalone punctuation (assuming they are tokenized as such)

# Updated pattern:
# - Standard words: ^[a-z]+$
# - Numbers (int/float): ^[0-9]+(?:\.[0-9]+)?$
# - Contractions: ^[a-z]+'[a-z]+$
# - Single standard punctuation characters from string.punctuation.
escaped_punctuation = re.escape(string.punctuation)
normal_token_pattern = re.compile(rf"^[a-z]+$|^[0-9]+(?:\.[0-9]+)?$|^[a-z]+'[a-z]+$|^[{escaped_punctuation}]$")

weird_tokens = Counter()
for token, count in all_tokens_combined.items():
    if not normal_token_pattern.match(token):
        # The improved tokenizer and normal_token_pattern should handle most cases.
        # Original checks for "word." or quoted strings are removed as the tokenizer now separates them.
        weird_tokens[token] = count

print(f"\nFound {len(weird_tokens)} types of potentially \'weird\' tokens.")
print(f"Total occurrences of \'weird\' tokens: {sum(weird_tokens.values())}")

if weird_tokens:
    print("Examples of weird tokens (up to 20):")
    for token, count in weird_tokens.most_common(20):
        print(f"    '{token}': {count}")

# %% Any words that are in validation but not in train?

print(f"Number of unique words in train (ignoring capitalization): {len(train_tokens)}")
valid_only_tokens = []
train_token_set = set(train_tokens.keys())
for token in valid_tokens.keys():
    if token not in train_token_set:
        valid_only_tokens.append(token)

print(f"\nNumber of tokens present in validation but not in train: {len(valid_only_tokens)}")
if valid_only_tokens:
    print(f"Examples of tokens in validation but not train (up to 20): {valid_only_tokens[:20]}")

# %% How many different numbers in the train valid and total. 

def get_numbers_from_tokens(token_counter):
    numbers = set()
    for token in token_counter.keys():
        # Basic check for integers. For floats, a regex like re.match(r'^-?\d+(\.\d+)?$', token) would be better.
        # Considering the context of "tinystories", primarily whole numbers might be expected.
        # Let's also consider tokens that might be numbers with punctuation, e.g. "10."
        # cleaned_token = token.strip('.,!?') # Removed: Improved tokenizer provides clean tokens
        if token.isdigit(): # Checks for positive integers
            numbers.add(token)
        else:
            # Try to convert to float to catch things like "3.14", "-5"
            try:
                float(token) # This will succeed for "3.14", "-5", "10" etc.
                numbers.add(token) # Add the original token if it represents a number
            except ValueError:
                pass # Not a number
    return numbers

train_numbers = get_numbers_from_tokens(train_tokens)
valid_numbers = get_numbers_from_tokens(valid_tokens)
all_lines = train_lines + valid_lines
all_tokens_for_numbers = get_tokens(all_lines) # Re-tokenize all lines to get all tokens for number counting
total_numbers = get_numbers_from_tokens(all_tokens_for_numbers)

print(f"\nNumber of unique numeric tokens in train: {len(train_numbers)}")
if train_numbers:
    print(f"Examples from train (up to 10): {sorted(list(train_numbers))[:10]}")
print(f"Number of unique numeric tokens in valid: {len(valid_numbers)}")
if valid_numbers:
    print(f"Examples from valid (up to 10): {sorted(list(valid_numbers))[:10]}")
print(f"Number of unique numeric tokens in total: {len(total_numbers)}")
if total_numbers:
    print(f"Examples from total (up to 10): {sorted(list(total_numbers))[:10]}") 