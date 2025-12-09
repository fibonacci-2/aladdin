import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

# ------------------------------------------
local_dir = "data/arabic-reddit"
shard_size = int(1e8)  # 100M tokens per shard

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

print(f"Data will be saved to: {DATA_CACHE_DIR}")

# load the cleaned reddit data
print("Loading cleaned Reddit data...")
df = pd.read_csv("/workspace/data/cleaned_reddit_data.csv")  # from previous script
print(f"Loaded {len(df)} posts/comments")

# init the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("riotu-lab/Aranizer-PBE-64k")

print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

def tokenize_doc(doc):
    """Tokenizes a single document and returns a numpy array of uint32 tokens"""
    try:
        encoded = tokenizer.encode_plus(
            doc["selftext"],
            add_special_tokens=True,
            return_attention_mask=False,
            return_tensors=None
        )
        tokens_np = np.array(encoded['input_ids'], dtype=np.uint32)
        return tokens_np
    except Exception as e:
        return np.array([], dtype=np.uint32)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# Convert DataFrame to list of dicts for multiprocessing
documents = df.to_dict('records')

print(f"Tokenizing {len(documents)} documents...")

# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count()//2)
print(f"Using {nprocs} processes for tokenization")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint32)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize_doc, documents, chunksize=16):
        if len(tokens) == 0:
            continue
            
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
            filename = os.path.join(DATA_CACHE_DIR, f"reddit_{split}_{shard_index:06d}")
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
        filename = os.path.join(DATA_CACHE_DIR, f"reddit_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])

print("Tokenization completed!")
print(f"Created {shard_index + 1} shards in {DATA_CACHE_DIR}")
