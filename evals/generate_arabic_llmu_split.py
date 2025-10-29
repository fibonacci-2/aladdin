import pandas as pd
import numpy as np

# Configuration
input_file = "ArabicMMLU.csv"  # Change this to your CSV file path
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
random_seed = 42

# Read the CSV
df = pd.read_csv(input_file)

# Shuffle the data
df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Calculate split sizes
total_size = len(df_shuffled)
train_size = int(total_size * train_ratio)
val_size = int(total_size * val_ratio)

# Split the data
train_df = df_shuffled[:train_size]
val_df = df_shuffled[train_size:train_size + val_size]
test_df = df_shuffled[train_size + val_size:]

# Save the splits
base_name = input_file.replace('.csv', '')
train_df.to_csv(f"{base_name}_train.csv", index=False)
val_df.to_csv(f"{base_name}_val.csv", index=False)
test_df.to_csv(f"{base_name}_test.csv", index=False)

# Print results
print(f"Original dataset: {total_size} rows")
print(f"Train: {len(train_df)} rows")
print(f"Validation: {len(val_df)} rows") 
print(f"Test: {len(test_df)} rows")
print("Splitting completed!")
