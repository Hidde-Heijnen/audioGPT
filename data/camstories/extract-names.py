# %% Extract names from capitalized words (filter out common words)

import nltk
from nltk.corpus import words, stopwords
import os

# %%

# Download required NLTK data (only if not already downloaded)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# %%

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# %%


# Load English word lists
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

# %%

# Common words that might be capitalized but aren't names
common_capitalized_words = {
    # Days and months
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
    'September', 'October', 'November', 'December',
    # Common sentence starters
    'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'And', 'But', 'Or',
    'So', 'Then', 'When', 'Where', 'Why', 'How', 'What', 'Who', 'Which',
    'After', 'Before', 'During', 'Since', 'Until', 'While', 'Because',
    'Although', 'Though', 'However', 'Therefore', 'Moreover', 'Furthermore',
    'Nevertheless', 'Meanwhile', 'Finally', 'First', 'Second', 'Third',
    'Next', 'Last', 'Again', 'Also', 'Still', 'Yet', 'Just', 'Only',
    'Even', 'Almost', 'Always', 'Never', 'Sometimes', 'Often', 'Usually',
    'Maybe', 'Perhaps', 'Probably', 'Certainly', 'Definitely', 'Absolutely',
    # Common nouns that might start sentences
    'People', 'Person', 'Man', 'Woman', 'Boy', 'Girl', 'Child', 'Children',
    'Family', 'Friend', 'Friends', 'Mother', 'Father', 'Sister', 'Brother',
    'House', 'Home', 'School', 'Work', 'Time', 'Day', 'Night', 'Morning',
    'Afternoon', 'Evening', 'Year', 'Month', 'Week', 'Hour', 'Minute',
    'Water', 'Food', 'Money', 'Book', 'Car', 'Tree', 'Dog', 'Cat',
    # Titles and honorifics
    'Mr', 'Mrs', 'Ms', 'Dr', 'Professor', 'Sir', 'Madam'
}

# Read the capitalized words file
file_path = '/home/hrah2/rds/hpc-work/audioGPT/data/camstories/capitalized_words_simplestories.txt'

if os.path.exists(file_path):
    with open(file_path, 'r') as f:
        capitalized_words = [line.strip() for line in f if line.strip()]
    
    print(f"Total capitalized words loaded: {len(capitalized_words)}")
    
    # Filter to get likely names
    likely_names = []
    
    for word in capitalized_words:
        # Skip if it's a common English word (lowercase version exists in dictionary)
        if word.lower() in english_words:
            continue
        
        # Skip if it's a stop word
        if word.lower() in stop_words:
            continue
            
        # Skip if it's in our common capitalized words list
        if word in common_capitalized_words:
            continue
            
        # Skip very short words (likely abbreviations or initials)
        if len(word) < 3:
            continue
            
        # Skip words with numbers
        if any(char.isdigit() for char in word):
            continue
            
        # If it passes all filters, it's likely a name
        likely_names.append(word)
    
    # Sort names alphabetically
    likely_names.sort()
    
    print(f"Likely names found: {len(likely_names)}")
    print(f"Filtered out: {len(capitalized_words) - len(likely_names)} common words")
    
    # Show first 50 names as sample
    print("\nFirst 50 likely names:")
    for i, name in enumerate(likely_names[:50]):
        print(f"{i+1:2d}. {name}")
    
    if len(likely_names) > 50:
        print(f"... and {len(likely_names) - 50} more names")
    
    # Save names to file
    names_file = 'extracted_names_simplestories.txt'
    with open(names_file, 'w') as f:
        for name in likely_names:
            f.write(f"{name}\n")
    
    print(f"\nAll names saved to: {names_file}")
    
    # Show some statistics
    print(f"\nName statistics:")
    print(f"- Shortest name: {min(likely_names, key=len) if likely_names else 'None'}")
    print(f"- Longest name: {max(likely_names, key=len) if likely_names else 'None'}")
    
    # Show names by length
    name_lengths = {}
    for name in likely_names:
        length = len(name)
        name_lengths[length] = name_lengths.get(length, 0) + 1
    
    print(f"- Names by length:")
    for length in sorted(name_lengths.keys()):
        print(f"  {length} chars: {name_lengths[length]} names")

else:
    print(f"File not found: {file_path}")
    print("Please run the capitalized words analysis first.")

# %%
