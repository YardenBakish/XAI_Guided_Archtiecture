import os
import re


def get_file_with_largest_number(directory):
    largest_number = -1
    largest_file = None

    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern 'file_<number>'
        match = re.match(r'checkpoint_(\d+)', filename)
        if match:
            # Extract the number part from the filename and convert it to an integer
            number = int(match.group(1))
            # Update the largest file if the current number is larger
            if number > largest_number:
                largest_number = number
                largest_file = filename

    if largest_file:
        return os.path.join(directory, largest_file)
    else:
        print("No suitable file found")

        return False  # No file found that matches the pattern


def create_directory_if_not_exists(directory):
    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
        print(f"Directory '{directory}' has been created.")
    else:
        print(f"Directory '{directory}' already exists.")

def is_valid_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory) and os.path.isdir(directory):
        # Check if the directory is not empty
        if os.listdir(directory):
            last_checkpoint_path =  get_file_with_largest_number(directory)
            return last_checkpoint_path
        else:
            print("Directory exists but is empty.")
            return False
    else:
        print("Directory does not exist.")
        return False