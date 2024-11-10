import os
import shutil
import subprocess

def move_directories(source_dir, dest_dir):
 
    # List all directories in source_dir, sorted alphabetically
    try:
        directories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        directories.sort()  # Sort directories alphabetically
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    except PermissionError:
        print(f"Error: Permission denied to access '{source_dir}'.")
        return

    # Check if we have more than 100 directories to move
    if len(directories) <= 100:
        print(f"There are {len(directories)} directories in '{source_dir}', no directories will be moved.")
        return

    # Directories to move (everything after the 100th one)
    directories_to_move = directories[100:]

    # Move directories to the destination
    for dir_name in directories_to_move:
        source_path = os.path.join(source_dir, dir_name)
        dest_path = os.path.join(dest_dir, dir_name)

        try:
            shutil.move(source_path, dest_path)
            print(f"Moved: {dir_name}")
        except Exception as e:
            print(f"Error moving '{dir_name}': {e}")

    print(f"Completed moving {len(directories_to_move)} directories to '{dest_dir}'.")

# Example usage:
source_directory = 'val'
destination_directory = 'train'

move_directories(source_directory, destination_directory)