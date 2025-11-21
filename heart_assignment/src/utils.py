import os, joblib, json

def save_model(model, path):
    """Save a sklearn model using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path):
    """Load a sklearn model saved with joblib."""
    return joblib.load(path)

def save_json(obj, path):
    """Save a Python object as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def load_json(path):
    """Load a JSON file and return Python object."""
    with open(path, "r") as f:
        return json.load(f)
