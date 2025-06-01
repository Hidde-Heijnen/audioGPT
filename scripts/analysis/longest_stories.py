# load  parquet_path=data/camstories_10k/camstories_10k.parquet and find the longest story in the story column

# %%
import pandas as pd
import os

# Get the absolute path to the parquet file
current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
parquet_path = os.path.join(workspace_root, 'data', 'camstories_10k', 'camstories_10k.parquet')

print(f"Loading parquet file from: {parquet_path}")
df = pd.read_parquet(parquet_path)

# find the longest story in the story column
longest_story_idx = df['story'].str.len().idxmax()
longest_story = df.loc[longest_story_idx, 'story']
print(f"Length of longest story: {len(longest_story)}")
print(f"Longest story:\n{longest_story}")

# %%