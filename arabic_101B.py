"""
Arabic 101 Billion Words dataset (for srs pretraining)
https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python arabic_words.py
Will save shards to the local directory "arabic_101B".
"""

import os
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "data/arabic_101B"
shard_size = int(1e8) # 100M tokens per shard

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
ds = load_dataset("ClusterlabAi/101_billion_arabic_words_dataset", split="train")

# init the tokenizer - using Aranizer instead of tiktoken
tokenizer = AutoTokenizer.from_pretrained("riotu-lab/Aranizer-PBE-86k")

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    # Use the Aranizer tokenizer to tokenize the text
    tokens = tokenizer.tokenize(doc["text"])
    
    # Convert tokens to token IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Add the special token at the beginning (using the tokenizer's EOS token as delimiter)
    # Note: You might want to use a different special token depending on the tokenizer's setup
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    if eos_token_id is None:
        # If no EOS token is defined, we'll just use 0 as delimiter (not ideal but works)
        eos_token_id = 0
    
    tokens_with_special = [eos_token_id] + token_ids
    tokens_np = np.array(tokens_with_special)
    
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, ds, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        if token_count + len(tokens) < shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"arabic_{split}_{shard_index:06d}")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"arabic_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])