import json

def load_json(filename):
    """Load JSON file as a dictionary."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_json(filename, data):
    """Save dictionary to JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def update_json(filename, new_dict):
    """Update or add a key-value pair to the JSON file."""
    data = load_json(filename)
    data.update(new_dict)
    print(data)
    save_json(filename, data)


