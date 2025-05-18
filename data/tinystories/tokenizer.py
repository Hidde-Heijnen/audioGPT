import torch
from transformers import AutoTokenizer
import json
import os

class Tokenizer:
  def __init__(self, config, k=None, file_path=None, device="cpu"):
    self.k = k
    self.file_path = file_path
    self.device = device
    current_dir = os.path.dirname(__file__)
    special_tokens_map_path = os.path.join(current_dir, 'special_tokens_map.json')
    self.tokenizer = AutoTokenizer.from_pretrained(config.name, special_tokens_map_file=special_tokens_map_path)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.vocab_size = self.tokenizer.vocab_size if not self.k else self.k
    self.initialize()

  def get_config(self):
    config = {
        "initl_vocab_size": self.tokenizer.vocab_size,
        "final_vocab_size": self.vocab_size,
        "vocab_size": self.vocab_size,
        "total_tokens": self.total_tokens,
        "total_tokens_used": self.tokens_used if self.k else self.total_tokens,
        "total_unsed_tokens": self.total_tokens - self.tokens_used if self.k else 0
    }
    return config

  def initialize(self):
    with open(self.file_path, 'r') as file:
      # Already sorted E.g {"and":5, "the": 3, "to": 2, "a": 1, "of": 1}
      tokens_counts = json.load(file)

    self.total_tokens = sum(tokens_counts.values())

    if self.k:
      self.tokens_used = sum([i for i in tokens_counts.values()][:self.k])
      # Get the top_k tokens based on frequency from the counts file
      top_k_from_counts = [id_str for id_str in tokens_counts.keys()][:self.k]

      # Prepare a list of essential special token IDs (as strings)
      essential_special_ids = []
      self.eos_token_id_str = str(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
      bos_token_id_str = str(self.tokenizer.bos_token_id) if self.tokenizer.bos_token_id is not None else None
      # unk_token_id_str = str(self.tokenizer.unk_token_id) if self.tokenizer.unk_token_id is not None else None # Optional: if UNK needs special handling in k-limited vocab

      if self.eos_token_id_str: essential_special_ids.append(self.eos_token_id_str)
      if bos_token_id_str: essential_special_ids.append(bos_token_id_str)
      # if unk_token_id_str: essential_special_ids.append(unk_token_id_str)

      # Combine: start with essential special tokens (unique), then add from top_k_from_counts until k (or k + specials) size is met.
      # This gives priority to special tokens if they were not already in the most frequent.
      final_top_k_set = set()
      self.top_k_tokens = []

      for s_id in essential_special_ids:
          if s_id not in final_top_k_set:
              self.top_k_tokens.append(s_id)
              final_top_k_set.add(s_id)
      
      for token_id_str in top_k_from_counts:
          if token_id_str not in final_top_k_set:
              if len(self.top_k_tokens) < self.k: # Fill up to k
                  self.top_k_tokens.append(token_id_str)
                  final_top_k_set.add(token_id_str)
              # else: # If already k elements and special tokens pushed it over, stop or cap at k
                  # For now, allow vocab_size to exceed k slightly if special tokens are added
                  # and were not part of the original top_k_from_counts.
                  # If strict k is needed, logic here would be different (e.g. k - num_specials from counts)

      # self.top_k_tokens = [i for i in tokens_counts.keys()][:self.k]# We will only use top k tokens, others will be ignored
      # eos_token_id_str = str(self.tokenizer.eos_token_id) # Use dynamic EOS token ID
      # self.top_k_tokens.append(eos_token_id_str) # Append stringified EOS ID
      # self.vocab_size +=1

      self.vocab_size = len(self.top_k_tokens) # vocab_size is the actual size of our list

      self.top_k_tokens_dict =  {token: index for index, token in enumerate(self.top_k_tokens)}
      self.reversed_top_k_tokens_dict = {value: int(key) for key, value in self.top_k_tokens_dict.items()}  # This is for decoding to reverse map and jump back to original 50k vocab
      

  def encoder(self, input, padding=False, max_length=256, truncation=False, add_special_tokens=True):
    # Let the base tokenizer handle adding BOS/EOS if add_special_tokens=True
    tokens = self.tokenizer(input , return_tensors='pt', padding=padding, max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)['input_ids'].to(self.device)
    
    if self.k:
      # Default OOV mapping to the new index of EOS token, if EOS is in vocab.
      # Otherwise, this could error if eos_token_id_str is not in top_k_tokens_dict.
      # The initialization logic above should ensure eos_token_id_str is present.
      default_map_idx = self.top_k_tokens_dict.get(self.eos_token_id_str) 
      if default_map_idx is None and self.top_k_tokens: # Fallback if EOS somehow missing, use last token in vocab as OOV
          default_map_idx = len(self.top_k_tokens) -1 
      elif default_map_idx is None: # No tokens at all, cannot map
          return torch.empty_like(tokens) # Or raise error

      tokens = torch.tensor([self.top_k_tokens_dict.get(str(token.item()), default_map_idx) for token in tokens.view(-1)], device=self.device).view(tokens.shape)

    return tokens

  def decoder(self, tokens):
    if self.k:
      tokens = torch.tensor([[self.reversed_top_k_tokens_dict[token.item()] for token in row] for row in tokens], device=tokens.device)
    
    output = [self.tokenizer.decode(x, skip_special_tokens=True) for x in tokens]

    return output