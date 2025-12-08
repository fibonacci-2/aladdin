import os

# Path to the folder containing the files
folder_path = 'fineweb-twitter-reddit/'

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    # Check if the file starts with 'arabic_train_'
    if filename.startswith('arabic_val_') and filename.endswith('.npy'):
        # Generate the new filename by replacing 'arabic_train_' with 'english_train_'
        new_filename = filename.replace('arabic_val_', 'fineweb_val_')
        
        # Get the full path for both old and new filenames
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f'Renamed: {filename} -> {new_filename}')

