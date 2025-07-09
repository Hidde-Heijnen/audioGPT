# %%
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
from transformers import AutoTokenizer
import structlog
import torch


# os.environ["TOKENIZERS_PARALLELISM"] = "false"


log = structlog.get_logger()

# Define shared special tokens
SPECIAL_TOKENS = {
    "unk_token": "<|unknown|>",
    "pad_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
}


class ConsistentTokenizerWrapper(PreTrainedTokenizerFast):
    """
    Wrapper that ensures all tokenizers consistently add both BOS and EOS tokens
    when add_special_tokens=True, regardless of the underlying tokenizer implementation.
    """
    
    def __init__(self, base_tokenizer):
        # Don't call super().__init__() to avoid conflicts
        self.__dict__.update(base_tokenizer.__dict__)
        self._base_tokenizer = base_tokenizer
    
    def __call__(self, text, add_special_tokens=False, return_tensors=None, padding=False, 
                 truncation=False, max_length=None, **kwargs):
        """Override __call__ to ensure consistent BOS/EOS token addition."""
        
        if add_special_tokens:
            # First tokenize without special tokens
            result = self._base_tokenizer(
                text, 
                add_special_tokens=False, 
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
            
            # Manually add BOS and EOS tokens
            input_ids = result['input_ids']
            
            if return_tensors == 'pt':
                # Handle torch tensor case
                bos_id = torch.tensor([self.bos_token_id], dtype=input_ids.dtype, device=input_ids.device)
                eos_id = torch.tensor([self.eos_token_id], dtype=input_ids.dtype, device=input_ids.device)
                
                # Add BOS at beginning and EOS at end
                input_ids = torch.cat([bos_id, input_ids.squeeze(0), eos_id]).unsqueeze(0)
                result['input_ids'] = input_ids
            else:
                # Handle list case
                if isinstance(input_ids[0], list):
                    # Batch case
                    for i in range(len(input_ids)):
                        input_ids[i] = [self.bos_token_id] + input_ids[i] + [self.eos_token_id]
                else:
                    # Single sequence case
                    input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
                result['input_ids'] = input_ids
                
            return result
        else:
            # Use base tokenizer as-is when not adding special tokens
            return self._base_tokenizer(
                text, 
                add_special_tokens=False, 
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                **kwargs
            )
    
    def __getattr__(self, name):
        """Delegate attribute access to the base tokenizer."""
        return getattr(self._base_tokenizer, name)


def preprocess_text(stories: List[str]) -> List[str]: # Done for new text that is not from preprocessed dataset before tokenizing
    return [contractions.fix(story) for story in stories]


def get_regex_pattern(seperate_possesive: bool = False) -> str:
    """Return regex pattern for pre-tokenization.

    Args:
        seperate_possesive (bool): If True, the pattern will isolate the possessive
            "'s" (and standalone trailing "'") as separate tokens. When False, the
            possessive stays attached to the preceding word (default behaviour).
    """
    # Base pattern that keeps possessive 's attached
    base_pattern = r"<\|[^|]+?\|>|\p{Emoji}|\b\w+(?:'\w+)?\b|[^\s\w]"

    # Extended pattern that isolates possessive `'s` but allows other contractions.
    # We use a negative look-ahead to prevent \b\w+'s\b being matched by the
    # generic word pattern.
    if seperate_possesive:
        return r"<\|[^|]+?\|>|\p{Emoji}|'s\b|\b\w+(?:'(?!s\b)\w+)?\b|[^\s\w]"

    return base_pattern


# Define normalization rules in a pickle-able format
NORMALIZATION_RULES = [
    # (r"('s\b)", r" \1"),  # Isolate possessive 's
    (r"(?<=\w)'(?=\s|$)", " '"),  # Isolate trailing '
    (r"[\n\r\t\xa0\u2028\u2029]", " "),  # Standardize whitespace
    (r"\s+", " ")  # Collapse multiple spaces
]

def apply_normalization_rules(text: str) -> str:
    """Applies a list of regex rules to a string for scripts."""
    text = text.lower()
    for pattern, replacement in NORMALIZATION_RULES:
        text = re.sub(pattern, replacement, text)
    return text


def get_normalizer() -> NormalizerSequence:
    """Returns the normalizer sequence for the word-level tokenizer."""
    normalizers = [
        NFKC(),
        BertNormalizer(
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=None,
            lowercase=True
        )
    ]
    for pattern, replacement in NORMALIZATION_RULES:
        normalizers.append(Replace(Regex(pattern), replacement))
    return NormalizerSequence(normalizers)


def get_word_level_tokenizer(vocab: Dict[str, int], seperate_possesive: bool = False) -> PreTrainedTokenizerFast:
    """Create a HuggingFace *word-level* tokenizer.

    If ``seperate_possesive`` is ``True`` the tokenizer will separate the
    possessive token "'s" (and a trailing apostrophe) from the preceding word.
    Otherwise the default behaviour keeps them attached.
    """

    log.warn("TODO: preprocessing with contraction-expansion")

    pattern = get_regex_pattern(seperate_possesive=seperate_possesive)

    tokenizer = Tokenizer(WordLevel(vocab, unk_token=SPECIAL_TOKENS["unk_token"]))
    tokenizer.normalizer = get_normalizer()

    # Build pre-tokenizer sequence. When we need possessive isolation we first
    # split on the specific pattern `'s\b`, and then apply the generic split
    # pattern.  This ensures we end up with exactly two tokens: the base word
    # and the "'s" suffix (not three separate tokens).

    pretokenizers = [WhitespaceSplit()]

    if seperate_possesive:
        pretokenizers.append(
            Split(Regex(r"'s\b"), behavior="isolated", invert=False)
        )

    pretokenizers.append(
        Split(
            pattern=Regex(pattern),  # generic split
            behavior="isolated",
            invert=False,
        )
    )

    tokenizer.pre_tokenizer = PreTokenizerSequence(pretokenizers)

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    # Add shared special tokens
    hf_tokenizer.add_special_tokens(SPECIAL_TOKENS)
    log.info("Original vocabulary size of Word Level tokenizer: ", vocab=hf_tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=hf_tokenizer.special_tokens_map)
    
    # Wrap with consistent behavior
    return ConsistentTokenizerWrapper(hf_tokenizer)


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


def get_vocab(
    text: str,
    dataset_name: str,
    vocab_size: int,
    tokenizer_name: str,
    built_vocab: bool = True,
    parquet_vocab: bool = True,
) -> Dict[str, int]:
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    vocab_size_name = vocab_size if vocab_size != 0 else "full"

    # Determine if possessive splitting is required based on tokenizer variant
    seperate_possesive = tokenizer_name == "word_level_pmod"

    # Keep vocabularies for possessive-splitting separate to avoid collisions
    ds_name = f"{dataset_name}_poss" if seperate_possesive else dataset_name
    vocab_file = None
    if parquet_vocab:
        vocab_file = os.path.join(
            base_dir, f"{ds_name}_{vocab_size_name}_vocab.parquet"
        )
    else:
        vocab_file = os.path.join(
            base_dir, "word_level_tokenizer", f"{ds_name}_{vocab_size_name}_vocab.csv"
        )
    log.info("Getting the vocab from", path=vocab_file)

    pattern = get_regex_pattern(seperate_possesive=seperate_possesive)
    log.info("Getting vocab for word-level tokenizer")

    if os.path.exists(vocab_file):
        log.info(f"Loading existing vocab from: {vocab_file}")
        vocab = load_vocab_from_csv(vocab_file)
    elif built_vocab:
        log.info(f"No vocab found. Creating and saving to: {vocab_file}")
        vocab, _ = build_vocab_from_data(text, pattern, vocab_size)
        if SPECIAL_TOKENS["unk_token"] not in vocab:
            vocab[SPECIAL_TOKENS["unk_token"]] = len(vocab)
        save_vocab_to_csv(vocab, vocab_file)
    else: 
        raise ValueError("No vocab found.")
    
    if SPECIAL_TOKENS["unk_token"] not in vocab:
        vocab[SPECIAL_TOKENS["unk_token"]] = len(vocab)
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
        stories = [s.strip() for s in text.split(SPECIAL_TOKENS["eos_token"]) if s.strip()]
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


def load_camstories_dataset(vocab_size: int = 0) -> Tuple[List[str], List[str], pd.DataFrame]:
    base_dir = os.path.dirname(__file__)  # Current directory where tokenizer.py is located
    # Load supported vocab-specific file if available
    supported_sizes = [10000]
    if vocab_size in supported_sizes:
        file_name = f"camstories_{vocab_size}.parquet"
    else:
        file_name = "camstories.parquet"
    file_path = os.path.join(base_dir, "data", "camstories", file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"camstories dataset for vocab size {vocab_size} not found at: {file_path}")
    df = pd.read_parquet(file_path)
    # Drop origin column if present
    if "origin" in df.columns:
        df = df.drop(columns=["origin"])
    train_lst = df[df["split"] == "train"]["story"].tolist()
    val_lst = df[df["split"] == "val"]["story"].tolist()
    return train_lst, val_lst, df


def get_dataset(dataset_name: str, vocab_size: int) -> Tuple[List[str], List[str], pd.DataFrame]:
    if dataset_name == "tiny_stories":
        train_lst, val_lst, df = load_tiny_stories_gpt4_dataset()
    elif dataset_name == "simple_stories":
        train_lst, val_lst, df = download_simple_stories_dataset()
    elif dataset_name == "little_stories":
        train_lst, val_lst, df = download_little_stories_dataset()
    elif dataset_name == "camstories":
        train_lst, val_lst, df = load_camstories_dataset(vocab_size=vocab_size)
    elif dataset_name == "test":
        train_lst = ["hello world", "'Hi!', he said. How are you?"]
        val_lst = ["This is a test validation string"]
        df = pd.DataFrame()
    else:
        raise ValueError("No valid dataset_name was given")
    return train_lst, val_lst, df


def get_byte_pair_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories")
    # Add shared special tokens
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    log.info("Original vocabulary size of BytePair tokenizer: ", vocab=tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=tokenizer.special_tokens_map)
    
    # Wrap with consistent behavior
    return ConsistentTokenizerWrapper(tokenizer)


def turn_list_of_stories_into_string(train_lst: List[str], val_lst: List[str]) -> Tuple[str, str]:
    train_str = SPECIAL_TOKENS["eos_token"].join(train_lst)
    val_str = SPECIAL_TOKENS["eos_token"].join(val_lst)
    return train_str, val_str


def get_word_piece_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-35M")
    # Add shared special tokens
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    log.info("Original vocabulary size of SimpleStories tokenizer: ", vocab=tokenizer.vocab_size)
    log.info("Special tokens map:", special_tokens_map=tokenizer.special_tokens_map)
    
    # Wrap with consistent behavior
    return ConsistentTokenizerWrapper(tokenizer)


def get_huggingface_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Create a HuggingFace tokenizer from any model name.
    
    Args:
        model_name (str): The HuggingFace model name/path to load the tokenizer from.
    
    Returns:
        PreTrainedTokenizerFast: The loaded tokenizer with special tokens added.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add shared special tokens
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        log.info(f"Loaded HuggingFace tokenizer from {model_name}")
        log.info("Original vocabulary size:", vocab=tokenizer.vocab_size)
        log.info("Special tokens map:", special_tokens_map=tokenizer.special_tokens_map)
        
        # Wrap with consistent behavior
        return ConsistentTokenizerWrapper(tokenizer)
    except Exception as e:
        log.error(f"Failed to load tokenizer from {model_name}: {e}")
        raise


def get_tokenizer(tokenizer_name: str, dataset_name: str, vocab_size: int = 0, built_vocab=True, parquet_vocab=True, model_name: str = None) -> PreTrainedTokenizerFast:
    """
    Get a tokenizer for a given dataset and vocabulary size.
    
    Args:
        tokenizer_name (str): Type of tokenizer to use. Options:
            - "word_level": Custom word-level tokenizer
            - "word_level_pmod": Custom word-level tokenizer with possessive modification
            - "byte_pair": Byte-pair encoding tokenizer (TinyStories)
            - "word_piece": Word-piece tokenizer (SimpleStories)
            - "huggingface": Any HuggingFace tokenizer (requires model_name)
        dataset_name (str): Name of the dataset
        vocab_size (int): Vocabulary size (0 for full vocabulary)
        built_vocab (bool): Whether to use pre-built vocabulary
        parquet_vocab (bool): Whether to use parquet format for vocabulary
        model_name (str, optional): HuggingFace model name when using "huggingface" tokenizer
        
    Returns:
        PreTrainedTokenizerFast: The configured tokenizer
        
    Examples:
        >>> # Use GPT-Neo tokenizer
        >>> tokenizer = get_tokenizer("huggingface", "camstories", model_name="EleutherAI/gpt-neo-125M")
        >>> 
        >>> # Use custom word-level tokenizer
        >>> tokenizer = get_tokenizer("word_level", "camstories", vocab_size=5000)
        >>>
        >>> # Use DialoGPT tokenizer
        >>> tokenizer = get_tokenizer("huggingface", "camstories", model_name="microsoft/DialoGPT-medium")
    """

    if tokenizer_name == "byte_pair":
        tokenizer = get_byte_pair_tokenizer()        

    elif tokenizer_name in ("word_level", "word_level_pmod"):

        if not built_vocab:
            train_lst, val_lst, _ = get_dataset(dataset_name, vocab_size)
            train_str, val_str = turn_list_of_stories_into_string(train_lst, val_lst)
            stories = train_str + val_str

            vocab = get_vocab(stories, dataset_name, vocab_size, tokenizer_name, built_vocab=built_vocab)
        else:
            vocab = get_vocab_from_file(dataset_name, vocab_size, tokenizer_name, parquet_vocab=parquet_vocab)

        seperate_possesive = tokenizer_name == "word_level_pmod"
        tokenizer = get_word_level_tokenizer(vocab, seperate_possesive=seperate_possesive)

    elif tokenizer_name == "word_piece":
        tokenizer = get_word_piece_tokenizer()
    
    elif tokenizer_name == "huggingface":
        if model_name is None:
            raise ValueError("model_name must be provided when using 'huggingface' tokenizer")
        tokenizer = get_huggingface_tokenizer(model_name)
    
    else: 
        raise ValueError(f"Currently not supporting {tokenizer_name} tokenizer")

    return tokenizer

def get_vocab_from_file(dataset_name: str, vocab_size: int, tokenizer_name: str, parquet_vocab: bool) -> Dict[str, int]:
    """Load an already pre-computed vocabulary from disk.

    The project ships pre-built vocabularies for the *camstories* datasets
    (and potentially others).  These vocabularies are stored either as
    parquet or csv files following the naming scheme

        <dataset_name>_<vocab_size>{_pmod}_vocab.(parquet|csv)
        <dataset_name>_full{_pmod}_vocab.(parquet|csv)

    where the optional ``_pmod`` suffix is used for the possessive-modified
    tokenizer variant (``word_level_pmod``).

    Parameters
    ----------
    dataset_name: str
        Base dataset identifier (e.g. "camstories").
    vocab_size: int
        Target vocabulary size.  Use ``0`` for the *full* vocabulary.
    tokenizer_name: str
        Either ``word_level`` or ``word_level_pmod``.  The latter triggers
        the ``_pmod`` suffix when looking up the vocab file.
    parquet_vocab: bool
        If ``True`` expects a ``.parquet`` file, otherwise a ``.csv`` file.

    Returns
    -------
    Dict[str, int]
        Mapping from *token* -> *index* as required by
        ``tokenizers.models.WordLevel``.
    """

    # Determine possessive splitting variant
    seperate_possesive = tokenizer_name == "word_level_pmod"

    # Build the expected file name
    vocab_size_name = vocab_size if vocab_size != 0 else "full"
    pmod_suffix = "_pmod" if seperate_possesive else ""
    file_ext = "parquet" if parquet_vocab else "csv"

    file_name = f"{dataset_name}_{vocab_size_name}{pmod_suffix}_vocab.{file_ext}"

    # Construct full path – vocabularies are stored under data/<dataset_name>
    base_dir = os.path.join(os.path.dirname(__file__), "data", dataset_name)
    vocab_path = os.path.join(base_dir, file_name)

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

    log.info("Loading vocabulary from existing file", path=vocab_path)

    if parquet_vocab:
        # Read only the necessary columns to keep memory usage low
        df = pd.read_parquet(vocab_path, columns=["token", "index"])
        vocab = {row.token: int(row.index) for row in df.itertuples(index=False)}
    else:
        vocab = load_vocab_from_csv(vocab_path)

    # Ensure all special tokens are present – should already be in parquet files but
    # we enforce it defensively.
    for special_token in SPECIAL_TOKENS.values():
        if special_token not in vocab:
            vocab[special_token] = max(vocab.values()) + 1

    return vocab