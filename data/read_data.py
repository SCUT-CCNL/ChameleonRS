# -*- coding: utf-8 -*-
import os


def list_files_to_txt(folder_path, output_file):
    '''Lists filenames (without extensions) in a folder and saves them to a `.txt` file.

    Args:
        folder_path (str): Path to the input directory (e.g., 'Potsdam/IRRG/img_dir/train').
        output_file (str): Path to the output `.txt` file (e.g., 'Potsdam/IRRG/train.txt').
    
    Notes:
        - Skips subdirectories, only processes files.
        - Automatically removes file extensions.
        - Creates parent directories if they don't exist.
    '''
    # Step 1: Get all filenames in the folder
    file_names = os.listdir(folder_path)
    
    # Step 2: Filter out subdirectories and extract names without extensions
    file_names_without_extension = [
        os.path.splitext(name)[0]  # Remove file extension
        for name in file_names
        if os.path.isfile(os.path.join(folder_path, name))  # Skip subfolders
    ]
    
    # Step 3: Early exit if no files found
    if not file_names_without_extension:
        print(f"?? Warning: No files found in '{folder_path}'!")
        return
    
    # Step 4: Create parent directories (if needed) and write the results
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)  # Ensure output dir exists
    with open(output_file, "w") as f:
        for name in file_names_without_extension:
            f.write(name + "\n")  # Write one filename per line
    
    print(f"Successfully saved to '{output_file}'")


if __name__ == "__main__":
    # Example: Process multiple paths in bulk
    datasets = [
        # Format: (input_directory, output_txt_path)
        ("Potsdam/IRRG/img_dir/train", "Potsdam/IRRG/train.txt"),
        ("Potsdam/RGB/img_dir/train", "Potsdam/RGB/train.txt"),
        ("Vaihingen/img_dir/val", "Vaihingen/val.txt"),
        ("Vaihingen/img_dir/test", "Vaihingen/test.txt"),
        ("Vaihingen/img_dir/train", "Vaihingen/train.txt"),  # Add more as needed
    ]
    
    # Execute for all defined paths
    for folder_path, output_txt in datasets:
        list_files_to_txt(folder_path, output_txt)
