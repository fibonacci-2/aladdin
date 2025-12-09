"""
Arabic 101 Billion Words dataset (for srs pretraining)
https://huggingface.co/datasets/ClusterlabAi/101_billion_arabic_words_dataset
Downloads and tokenizes the data and saves data shards to disk.
"""

import os
import multiprocessing as mp
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------------------
local_dir = "data/fineweb"
shard_size = int(1e8) # 100M tokens per shard

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

print(f"Data will be saved to: {DATA_CACHE_DIR}")

# download the dataset
print("Loading dataset...")
# ds = load_dataset("ClusterlabAi/101_billion_arabic_words_dataset", split="train")
ds = load_dataset("Omartificial-Intelligence-Space/FineWeb2-MSA", split="train")  # for testing
# init the tokenizer - using Aranizer instead of tiktoken
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("riotu-lab/Aranizer-PBE-64k")

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    # Use encode_plus for better handling of special tokens
    encoded = tokenizer.encode_plus(
        doc["text"],
        add_special_tokens=True,
        return_attention_mask=False,
        return_tensors=None
    )
    
    tokens_np = np.array(encoded['input_ids'], dtype=np.uint32)
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# tokenie all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = 24
print(f"Using {nprocs} processes for tokenization")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard - using uint32 for large vocabularies
    all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
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
        filename = os.path.join(DATA_CACHE_DIR, f"tweets_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print("Tokenization completed!")

