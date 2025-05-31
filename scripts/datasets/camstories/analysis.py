# %% Imports
import pandas as pd
import hashlib
from collections import defaultdict
import re
from collections import Counter

# %% dataset anaylysis in data.csv

df = pd.read_csv('data.csv')

# %%
# columns
print(df.columns)

# types of splits
print('splits')
print(df['split'].value_counts()) # train, val
# types of origins
print('origins')
print(df['origin'].value_counts()) # TinyStories, SimpleStories

# %% Check for duplicates efficiently for large datasets
story_hashes = defaultdict(list)
duplicate_count = 0

for idx, story in enumerate(df['story']):
    if pd.notna(story):  # Skip NaN values
        # Create hash of lowercase story
        story_hash = hashlib.md5(str(story).lower().encode()).hexdigest()
        story_hashes[story_hash].append(idx)

# Count duplicates
for hash_key, indices in story_hashes.items():
    if len(indices) > 1:
        duplicate_count += len(indices) - 1  # All but the first occurrence

print(f"Number of duplicate stories (hash method): {duplicate_count}")
# %% Give me all capitalised words in the story column that have origin SimpleStories

# Filter for SimpleStories origin first (reduces data size)
simple_stories_df = df[df['origin'] == 'SimpleStories']
print(f"SimpleStories rows: {len(simple_stories_df)}")

# Using regex to find and count capitalized words efficiently
print("Finding and counting capitalized words...")
word_counter = Counter()

# Process in chunks to manage memory
chunk_size = 50000
total_chunks = (len(simple_stories_df) + chunk_size - 1) // chunk_size

for i, chunk_start in enumerate(range(0, len(simple_stories_df), chunk_size)):
    chunk_end = min(chunk_start + chunk_size, len(simple_stories_df))
    chunk = simple_stories_df.iloc[chunk_start:chunk_end]
    
    # Extract all capitalized words from this chunk
    for story in chunk['story']:
        if pd.notna(story):
            # Find words that start with capital letter and have at least one lowercase
            # This excludes ALL CAPS words and single letters
            caps_words = re.findall(r'\b[A-Z][a-z]+\b', str(story))
            word_counter.update(caps_words)
    
    print(f"Processed chunk {i+1}/{total_chunks}")

print(f"Total unique capitalized words found: {len(word_counter)}")

# %%
# Show most common capitalized words
print("Top 20 most common capitalized words:")
for word, count in word_counter.most_common(20):
    print(f"{word}: {count}")

# Show some statistics
print(f"\nStatistics:")
print(f"Total unique capitalized words: {len(word_counter)}")
print(f"Total capitalized word occurrences: {sum(word_counter.values())}")

# Optional: Save results to file for further analysis
print("\nSaving results...")
# Save unique words
with open('capitalized_words_simplestories.txt', 'w') as f:
    for word in sorted(word_counter.keys()):
        f.write(f"{word}\n")

# Save word frequencies
with open('capitalized_words_frequency_simplestories.txt', 'w') as f:
    for word, count in word_counter.most_common():
        f.write(f"{word}\t{count}\n")

print("Results saved to:")
print("- capitalized_words_simplestories.txt (unique words)")
print("- capitalized_words_frequency_simplestories.txt (word frequencies)")

# %% How many words only contain the letters a and h to lower on both origins combined (so whole dataset)

print("Finding words that only contain letters 'a' and 'h' (case-insensitive)...")

# Counter for words containing only 'a' and 'h'
ah_words_counter = Counter()

# Process the entire dataset
chunk_size = 50000
total_chunks = (len(df) + chunk_size - 1) // chunk_size

for i, chunk_start in enumerate(range(0, len(df), chunk_size)):
    chunk_end = min(chunk_start + chunk_size, len(df))
    chunk = df.iloc[chunk_start:chunk_end]
    
    # Extract words from this chunk
    for story in chunk['story']:
        if pd.notna(story):
            # Find all words (sequences of letters)
            words = re.findall(r'\b[a-zA-Z]+\b', str(story))
            
            # Check each word to see if it only contains 'a' and 'h' (case-insensitive)
            for word in words:
                # Convert to lowercase and check if all characters are 'a' or 'h'
                word_lower = word.lower()
                if len(word_lower) > 0 and all(char in 'ah' for char in word_lower):
                    ah_words_counter[word] += 1
    
    print(f"Processed chunk {i+1}/{total_chunks}")

print(f"\nResults:")
print(f"Total unique words containing only 'a' and 'h': {len(ah_words_counter)}")
print(f"Total occurrences of such words: {sum(ah_words_counter.values())}")

# Show the words found
if ah_words_counter:
    print(f"\nAll words containing only 'a' and 'h' (sorted by frequency):")
    for word, count in ah_words_counter.most_common():
        print(f"{word}: {count}")
else:
    print("\nNo words found that contain only letters 'a' and 'h'")

# Save results
print("\nSaving results...")
with open('ah_only_words.txt', 'w') as f:
    f.write("Words containing only letters 'a' and 'h' (case-insensitive)\n")
    f.write("=" * 50 + "\n")
    f.write(f"Total unique words: {len(ah_words_counter)}\n")
    f.write(f"Total occurrences: {sum(ah_words_counter.values())}\n\n")
    
    if ah_words_counter:
        f.write("Word frequencies:\n")
        for word, count in ah_words_counter.most_common():
            f.write(f"{word}\t{count}\n")
    else:
        f.write("No words found.\n")

print("Results saved to: ah_only_words.txt")
# %%
