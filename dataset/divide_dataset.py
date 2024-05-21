import os
import shutil
import random

def split_dataset(source_dir, dest_dir1, dest_dir2, split_ratio=0.8):
    # Create destination directories if they don't exist
    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)
    
    # Traverse through the source directory
    for root, dirs, files in os.walk(source_dir):
        # Calculate the split index for files in this directory
        split_index = int(len(files) * split_ratio)
        
        # Split the files
        train_files = files[:split_index]
        test_files = files[split_index:]
        
        # Move files to destination directories
        for file in train_files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir1, os.path.relpath(source_file_path, source_dir))
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy(source_file_path, dest_file_path)
        
        for file in test_files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_dir2, os.path.relpath(source_file_path, source_dir))
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy(source_file_path, dest_file_path)
    
    print("Dataset split successfully.")

# Example usage
source_directory = "./dataset"
destination_directory1 = "set_1"
destination_directory2 = "set_2"

split_dataset(source_directory, destination_directory1, destination_directory2, split_ratio=0.5)
