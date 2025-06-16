import os
import regex as re
import pandas as pd
import csv
import zipfile
import contractions
from collections import Counter
from tokenizers.models import WordLevel
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tokenizers import Regex
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence, WhitespaceSplit, Split
from tokenizers.normalizers import Sequence as NormalizerSequence, NFKC, BertNormalizer, Replace
from tokenizers import Tokenizer
from typing import List, Tuple, Dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer

import structlog
log = structlog.get_logger()


def preprocess_text(stories: List[str]) -> List[str]: # Done for new text that is not from preprocessed dataset before tokenizing
    return [contractions.fix(story) for story in stories]


def get_regex_pattern() -> str:
    # return r"<\|[^|]+?\|>|\p{Emoji}|\b\w+\?|\b\w+(?:'\w+)?\b|[^\s\w]" # matches word? even if no white space after it; ignore all whitespace characters (spaces, tabs, newlines, etc.)
    return r"<\|[^|]+?\|>|\p{Emoji}|\b\w+(?:'\w+)?\b|[^\s\w]" # separates question mark and word; ignore all whitespace characters (spaces, tabs, newlines, etc.)


def get_word_level_tokenizer(vocab: Dict[str, int]) -> PreTrainedTokenizerFast:
    """Splits text into disctinct words and characters. It ignores whitespace."""

    log.warn("TODO: preprocessing with contraction-expansion")
    pattern = get_regex_pattern()

    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<|unknown|>"))
    tokenizer.normalizer = NormalizerSequence([
        NFKC(),       # Unicode normalization
        BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=None,
            lowercase=True
        ),
        Replace(Regex(r"[\n\r\t\xa0\u2028\u2029]"), " "),
        Replace(Regex(r"\s+"), " ")
    ])
    tokenizer.pre_tokenizer = PreTokenizerSequence([
        WhitespaceSplit(), # ignore white space
        Split(
            pattern=Regex(pattern), # use regex pattern to split
            behavior="isolated", 
            invert=False)
    ])
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    special_tokens = {
        "unk_token": "<|unknown|>",
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
    hf_tokenizer.add_special_tokens(special_tokens)
    log.info("Original vocabulary size of Word Level tokenizer: ", vocab=hf_tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=hf_tokenizer.special_tokens_map)
    return hf_tokenizer


def save_vocab_to_csv(vocab: dict, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        for token, index in vocab.items():
            writer.writerow([token, index])


def load_vocab_from_csv(filepath: str) -> dict:
    vocab = {}
    with open(filepath, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) == 2:
                token, index = row
                vocab[token] = int(index)
    return vocab


def build_vocab_from_data(text: str, pattern: str, max_vocab_size: int) -> Tuple[dict, Counter]:
    text = text.lower()
    tokens = re.findall(pattern, text)
    token_counts = Counter(tokens)

    special_tokens = ["<|endoftext|>", "<|unknown|>"]
    num_reserved = 0
    for st in special_tokens:
        if st not in token_counts:
            num_reserved += 1

    if max_vocab_size > 0:
        most_common_tokens = token_counts.most_common(max_vocab_size - num_reserved)
        vocab_tokens = [token for token, _ in most_common_tokens]
    else:
        vocab_tokens = list(token_counts.keys())

    vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

    for special_token in special_tokens:
        if special_token not in vocab:
            vocab[special_token] = len(vocab)

    if max_vocab_size > 0:
        assert len(vocab) <= max_vocab_size

    return vocab, token_counts


def get_vocab(text: str, dataset_name: str, vocab_size: int, built_vocab: bool = True) -> Dict[str, int]:
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    vocab_size_name = vocab_size
    if vocab_size == 0:
        vocab_size_name = "full"
    vocab_file = os.path.join(base_dir, "word_level_tokenizer", f"{dataset_name}_{vocab_size_name}_vocab.csv")
    log.info("Getting the vocab from", path=vocab_file)

    pattern = get_regex_pattern()
    log.info("Getting vocab for word-level tokenizer")

    if os.path.exists(vocab_file):
        log.info(f"Loading existing vocab from: {vocab_file}")
        vocab = load_vocab_from_csv(vocab_file)
    elif built_vocab:
        log.info(f"No vocab found. Creating and saving to: {vocab_file}")
        vocab, _ = build_vocab_from_data(text, pattern, vocab_size)
        if "<|unknown|>" not in vocab:
            vocab["<|unknown|>"] = len(vocab)
        save_vocab_to_csv(vocab, vocab_file)
    else: 
        raise ValueError("No vocab found.")
    
    if "<|unknown|>" not in vocab:
        vocab["<|unknown|>"] = len(vocab)
    return vocab 


def load_tiny_stories_gpt4_dataset() -> Tuple[List[str], List[str], pd.DataFrame]:
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    data_dir = os.path.join(base_dir, "project_data", "tiny_stories")
    train_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    val_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")

    train_zip = train_path + ".zip"
    val_zip = val_path + ".zip"

    # Unzip if only zipped version exists
    if not os.path.exists(train_path) and os.path.exists(train_zip):
        with zipfile.ZipFile(train_zip, "r") as zipf:
            zipf.extractall(data_dir)

    if not os.path.exists(val_path) and os.path.exists(val_zip):
        with zipfile.ZipFile(val_zip, "r") as zipf:
            zipf.extractall(data_dir)

    # Raise error if still missing
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing: {train_path} (or zipped version)")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Missing: {val_path} (or zipped version)")
    

    def load_and_process(path: str) -> Tuple[List[str]]:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        stories = [s.strip() for s in text.split("<|endoftext|>") if s.strip()]
        return stories

    train_lst = load_and_process(train_path)
    val_lst = load_and_process(val_path)

    df = pd.DataFrame({
        "story": train_lst + val_lst,
        "split": ["train"] * len(train_lst) + ["val"] * len(val_lst),
    })

    return train_lst, val_lst, df


def download_simple_stories_dataset() -> Tuple[List[str], List[str], pd.DataFrame]:
    dataset = load_dataset("SimpleStories/SimpleStories") 
    train_lst = dataset["train"]['story']
    val_lst = dataset["test"]['story']
    df = pd.DataFrame({
        "story": train_lst + val_lst,
        "split": ["train"] * len(train_lst) + ["val"] * len(val_lst),
    })
    return train_lst, val_lst, df


def download_little_stories_dataset() -> Tuple[List[str], List[str], pd.DataFrame]: 
    dataset = load_dataset("Corianas/LittleStories")
    train_lst = dataset['train']['story']
    val_lst = dataset['test']['story']
    df = pd.DataFrame({
        "story": train_lst + val_lst,
        "split": ["train"] * len(train_lst) + ["val"] * len(val_lst),
    })
    return train_lst, val_lst, df


def load_preprocessed_stories(vocab_size: int = 0) -> Tuple[List[str], List[str], pd.DataFrame]:
    preprocessed_dir = os.path.join(os.path.dirname(__file__), "preprocessed_stories")

    if vocab_size == 0:
        file_name = "preprocessed_stories_full_vocab.csv"
    else:
        file_name = f"preprocessed_stories_{vocab_size}_vocab.csv"

    input_path = os.path.join(preprocessed_dir, file_name)

    if not os.path.exists(input_path):
        raise ValueError(f"Preprocessed stories not found at: {input_path}")

    df = pd.read_csv(input_path)

    train_lst = df[df["split"] == "train"]["story"].tolist()
    val_lst = df[df["split"] == "val"]["story"].tolist()

    return train_lst, val_lst, df


def load_camstories_10k_dataset() -> Tuple[List[str], List[str], pd.DataFrame]:
    """Load the camstories_10k dataset from parquet file."""
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    parquet_path = os.path.join(base_dir, "data", "camstories_10k", "camstories_10k.parquet")
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Missing: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Create train/val split (80/20 split)
    split_idx = int(len(df) * 0.8)
    
    train_stories = df.iloc[:split_idx]['story'].tolist()
    val_stories = df.iloc[split_idx:]['story'].tolist()
    
    # Update dataframe with split information
    df['split'] = ['train'] * split_idx + ['val'] * (len(df) - split_idx)
    
    return train_stories, val_stories, df


def get_dataset(dataset_name: str, vocab_size: int) -> Tuple[List[str], List[str], pd.DataFrame]:
    if dataset_name == "tiny_stories":
        train_lst, val_lst, df = load_tiny_stories_gpt4_dataset()
    elif dataset_name == "simple_stories":
        train_lst, val_lst, df = download_simple_stories_dataset()
    elif dataset_name == "little_stories":
        train_lst, val_lst, df = download_little_stories_dataset()
    elif dataset_name == "preprocessed_stories":
        train_lst, val_lst, df = load_preprocessed_stories(vocab_size=vocab_size)
    elif dataset_name == "camstories_10k":
        train_lst, val_lst, df = load_camstories_10k_dataset()
    elif dataset_name == "test":
        train_lst = ["hello world", "'Hi!', he said. How are you?"]
        val_lst = ["This is a test validation string"]
        df = pd.DataFrame()
    else:
        raise ValueError("No valid dataset_name was given")
    return train_lst, val_lst, df


def get_byte_pair_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")
    special_tokens = {
        "unk_token": "<|unknown|>",
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)
    log.info("Original vocabulary size of BytePair tokenizer: ", vocab=tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=tokenizer.special_tokens_map)
    return tokenizer


def turn_list_of_stories_into_string(train_lst: List[str], val_lst: List[str]) -> Tuple[str, str]:
    train_str = '<|endoftext|>'.join(train_lst)
    val_str = '<|endoftext|>'.join(val_lst)
    return train_str, val_str


def get_word_piece_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-35M")
    special_tokens = {
        "unk_token": "<|unknown|>",
        "pad_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
    }
    tokenizer.add_special_tokens(special_tokens)
    log.info("Original vocabulary size of SimpleStories tokenizer: ", vocab=tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=tokenizer.special_tokens_map)
    return tokenizer


def get_tokenizer(tokenizer_name: str, dataset_name: str, vocab_size: int = 0, built_vocab=True) -> PreTrainedTokenizerFast:
    if tokenizer_name == "byte_pair":
        tokenizer = get_byte_pair_tokenizer()

    elif tokenizer_name == "word_level":
        train_lst, val_lst, _ = get_dataset(dataset_name, vocab_size)
        train_str,  val_str = turn_list_of_stories_into_string(train_lst, val_lst)
        stories = train_str + val_str 
        vocab = get_vocab(stories, dataset_name, vocab_size, built_vocab=built_vocab)
        tokenizer = get_word_level_tokenizer(vocab)

    elif tokenizer_name == "word_piece":
        tokenizer = get_word_piece_tokenizer()
    
    else: 
        raise ValueError(f"Currently not supporting {tokenizer_name} tokenizer")

    return tokenizer



