# %% [markdown]
# # TinyStories Data Preprocessing Script
#
# This script preprocesses a raw TinyStories data file where stories are concatenated
# and separated by a specific delimiter (e.g., `<|endoftext|>`).
#
# ## Functionality:
# 1. Reads the raw input file.
# 2. Splits the content into individual stories based on the specified delimiter.
# 3. Cleans each story by:
#    - Stripping leading/trailing whitespace.
#    - Replacing multiple consecutive spaces with a single space.
#    - Normalizing multiple newlines (and newlines with spaces) to a single newline.
# 4. Writes the cleaned stories to an output file, with each story on a new line.
#    This format is suitable for subsequent tokenization scripts.

# %%
import re
import os
import sys

# %% [markdown]
# ## Configuration
# Define the input and output file paths, and the delimiter used to separate stories.

# %%
# --- Configuration ---
# Path to your raw input file (e.g., your TinyStoriesV2-GPT4-train.txt)
# IMPORTANT: Place this script in a directory where it can access the input file,
# or provide the full path to the input file.
# Example: INPUT_RAW_STORY_FILE_TRAIN = "path/to/your/TinyStoriesV2-GPT4-train.txt"
INPUT_RAW_STORY_FILE_TRAIN = "TinyStoriesV2-GPT4-train.txt" # Replace with your actual train file name/path
INPUT_RAW_STORY_FILE_VALID = "TinyStoriesV2-GPT4-valid.txt" # Replace with your actual validation file name/path (if you have one)

# Directory where the cleaned files will be saved.
# This script will create this directory if it doesn't exist.
# The prepare.py and prepare_word.py scripts expect cleaned files in 'data/tinystories/'
OUTPUT_DIR = "data/tinystories"
CLEANED_TRAIN_FILE = os.path.join(OUTPUT_DIR, "tinystories_train_cleaned.txt")
CLEANED_VALID_FILE = os.path.join(OUTPUT_DIR, "tinystories_valid_cleaned.txt")

# Delimiter used to separate stories in your raw input file
STORY_DELIMITER = "<|endoftext|>"

# %% [markdown]
# ## Preprocessing Function

# %%
def preprocess_raw_story_file(input_path, output_path, delimiter):
    """
    Reads a raw story file separated by a delimiter,
    cleans each story (removes extra spaces, normalizes newlines),
    and writes one story per line to the output file.
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found at {input_path}")
        print("Please ensure the INPUT_RAW_STORY_FILE path is correct.")
        return 0  # Return 0 stories processed if file not found

    print(f"Starting preprocessing for: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
    except Exception as e:
        print(f"Error reading file {input_path}: {e}")
        return 0

    stories = content.split(delimiter)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    num_processed_stories = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for i, story_text in enumerate(stories):
            # 1. Strip leading/trailing whitespace from the raw story segment
            cleaned_story = story_text.strip()

            if cleaned_story: # Process only if the story is not empty after stripping
                # 2. Replace multiple spaces with a single space
                cleaned_story = re.sub(r' +', ' ', cleaned_story)

                # 3. Normalize newlines:
                #    - Replace multiple newlines (possibly with spaces between them) with a single newline.
                #    - This helps ensure consistent line breaks within a story if desired,
                #      but the main goal is one story per line in the output.
                cleaned_story = re.sub(r'\n\s*\n', '\n', cleaned_story)
                
                # 4. Remove any remaining newlines within the story if you want each story
                #    to be a single continuous block of text on its line in the output file.
                #    If you want to preserve intra-story newlines, comment out the next line.
                # cleaned_story = cleaned_story.replace('\n', ' ') # Optional: makes each story one long line

                f_out.write(cleaned_story + '\n')
                num_processed_stories += 1
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1}/{len(stories)} story segments from {input_path}...")

    print(f"Finished preprocessing for: {input_path}")
    print(f"Cleaned data written to: {output_path}")
    print(f"Total stories processed and written: {num_processed_stories}")
    return num_processed_stories

# %% [markdown]
# ## Main Execution Block

# %%
if __name__ == "__main__":
    print("--- Starting TinyStories Data Preprocessing ---")

    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    # Process the training data
    if os.path.exists(INPUT_RAW_STORY_FILE_TRAIN):
        print(f"\nProcessing Training Data: {INPUT_RAW_STORY_FILE_TRAIN}")
        train_stories_count = preprocess_raw_story_file(INPUT_RAW_STORY_FILE_TRAIN, CLEANED_TRAIN_FILE, STORY_DELIMITER)
        if train_stories_count == 0 and os.path.exists(INPUT_RAW_STORY_FILE_TRAIN):
             print(f"WARNING: No stories were processed from {INPUT_RAW_STORY_FILE_TRAIN}. Check the file content and delimiter.")
    else:
        print(f"INFO: Training file '{INPUT_RAW_STORY_FILE_TRAIN}' not found. Skipping training data processing.")
        print("      Please set INPUT_RAW_STORY_FILE_TRAIN to your actual file path if you have one.")

    # Process the validation data (if specified and exists)
    if INPUT_RAW_STORY_FILE_VALID and os.path.exists(INPUT_RAW_STORY_FILE_VALID):
        print(f"\nProcessing Validation Data: {INPUT_RAW_STORY_FILE_VALID}")
        valid_stories_count = preprocess_raw_story_file(INPUT_RAW_STORY_FILE_VALID, CLEANED_VALID_FILE, STORY_DELIMITER)
        if valid_stories_count == 0 and os.path.exists(INPUT_RAW_STORY_FILE_VALID):
            print(f"WARNING: No stories were processed from {INPUT_RAW_STORY_FILE_VALID}. Check the file content and delimiter.")
    elif INPUT_RAW_STORY_FILE_VALID: # Path was given but file doesn't exist
        print(f"INFO: Validation file '{INPUT_RAW_STORY_FILE_VALID}' not found. Skipping validation data processing.")
        print("      Please set INPUT_RAW_STORY_FILE_VALID to your actual file path if you have one, or leave it empty.")
    else: # Path was not given
        print("\nINFO: No validation file specified (INPUT_RAW_STORY_FILE_VALID is empty). Skipping validation data processing.")

    print("\n--- Preprocessing Script Finished ---")
    print(f"Cleaned files are located in: {os.path.abspath(OUTPUT_DIR)}")
    print("You can now use these cleaned files (e.g., tinystories_train_cleaned.txt) as input for")
    print("the 'prepare.py' or 'prepare_word.py' scripts by adjusting their RAW_TRAIN_FILE and RAW_VALID_FILE paths.")
