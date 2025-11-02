import os
import shutil
import numpy as np
from pathlib import Path
import random

# Source directories
source_dir1 = "Arabic-Tweets"
source_dir2 = "arabic_101B"

# Output directory
output_dir = "combined_shuffled"
os.makedirs(output_dir, exist_ok=True)

# Get all train and val shards from both datasets
def get_shards(directory, split):
    pattern = f"*_{split}_*.npy"
    return sorted(Path(directory).glob(pattern))

train_shards_1 = get_shards(source_dir1, "train")
train_shards_2 = get_shards(source_dir2, "train")
val_shards_1 = get_shards(source_dir1, "val")
val_shards_2 = get_shards(source_dir2, "val")

print(f"Arabic-Tweets: {len(train_shards_1)} train, {len(val_shards_1)} val")
print(f"arabic_101B: {len(train_shards_2)} train, {len(val_shards_2)} val")

# Balance by sampling from larger dataset to match smaller one
def balance_shards(shards1, shards2):
    """Balance two shard lists by sampling the larger one to match the smaller"""
    len1, len2 = len(shards1), len(shards2)
    
    if len1 < len2:
        # Sample shards2 to match len1
        random.seed(42)
        balanced_shards2 = random.sample(list(shards2), len1)
        balanced_shards1 = list(shards1)
    elif len2 < len1:
        # Sample shards1 to match len2
        random.seed(42)
        balanced_shards1 = random.sample(list(shards1), len2)
        balanced_shards2 = list(shards2)
    else:
        # Already balanced
        balanced_shards1 = list(shards1)
        balanced_shards2 = list(shards2)
    
    return balanced_shards1, balanced_shards2

# Balance train and val shards
balanced_train_1, balanced_train_2 = balance_shards(train_shards_1, train_shards_2)
balanced_val_1, balanced_val_2 = balance_shards(val_shards_1, val_shards_2)

print(f"\nBalanced:")
print(f"Arabic-Tweets: {len(balanced_train_1)} train, {len(balanced_val_1)} val")
print(f"arabic_101B: {len(balanced_train_2)} train, {len(balanced_val_2)} val")

# Combine and shuffle
all_train_shards = balanced_train_1 + balanced_train_2
all_val_shards = balanced_val_1 + balanced_val_2

random.seed(42)
random.shuffle(all_train_shards)
random.shuffle(all_val_shards)

print(f"\nTotal: {len(all_train_shards)} train, {len(all_val_shards)} val shards")

# Copy and rename shards
def copy_shards(shard_list, split):
    for idx, shard_path in enumerate(shard_list):
        new_name = f"combined_{split}_{idx:06d}.npy"
        dest_path = os.path.join(output_dir, new_name)
        shutil.copy2(shard_path, dest_path)
        if idx % 50 == 0:
            print(f"Copied {idx+1}/{len(shard_list)} {split} shards")
    print(f"Finished copying {len(shard_list)} {split} shards")

print("\nCopying train shards...")
copy_shards(all_train_shards, "train")

print("\nCopying val shards...")
copy_shards(all_val_shards, "val")

print(f"\nDone! Combined dataset saved to: {output_dir}")
print(f"Total size: {sum(f.stat().st_size for f in Path(output_dir).glob('*.npy')) / (1024**3):.2f} GB")