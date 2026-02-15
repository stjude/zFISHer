import json
from pathlib import Path
import numpy as np

# Private global state
_SESSION_DATA = {
    "output_dir": None,
    "r1_path": None,
    "r2_path": None,
    "shift": None,
    "processed_files": {} 
}
_is_loading = False

def set_loading(state: bool):
    """Sets the session loading state."""
    global _is_loading
    _is_loading = state

def is_loading():
    """Checks if a session is currently being loaded."""
    return _is_loading

def get_data(key=None, default=None):
    """Retrieves a value from the session data."""
    if key:
        return _SESSION_DATA.get(key, default)
    return _SESSION_DATA

def update_data(key, value):
    """Updates a value in the session data and saves the session."""
    _SESSION_DATA[key] = value
    save_session()

def set_processed_file(layer_name, path, layer_type: str, metadata: dict = None):
    """Registers a processed file path and its type, then saves the session."""
    if "processed_files" not in _SESSION_DATA:
        _SESSION_DATA["processed_files"] = {}
    
    file_info = {
        'path': str(path),
        'type': layer_type, # e.g., 'points', 'labels', 'image', 'vectors', 'report'
    }
    if metadata:
        file_info.update(metadata)

    _SESSION_DATA["processed_files"][layer_name] = file_info
    save_session()

def clear_session():
    """Resets the session data to defaults."""
    global _SESSION_DATA
    _SESSION_DATA.update({
        "output_dir": None,
        "r1_path": None,
        "r2_path": None,
        "shift": None,
        "processed_files": {}
    })

def initialize_new_session(output_dir, r1_path, r2_path):
    """
    Clears the current session, sets up a new one, and creates directories.
    Returns True on success, False if a session already exists in the directory.
    """
    output_dir = Path(output_dir)
    session_file = output_dir / "zfisher_session.json"
    if session_file.exists():
        return False

    # Create all necessary subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    (output_dir / "aligned").mkdir(exist_ok=True)
    (output_dir / "captures").mkdir(exist_ok=True)
    (output_dir / "input").mkdir(exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)

    # Reset and update session state
    clear_session()
    update_data("output_dir", str(output_dir))
    update_data("r1_path", str(r1_path))
    update_data("r2_path", str(r2_path))
    save_session()
    return True

def save_session():
    """Saves the current session state to JSON."""
    out_dir = _SESSION_DATA.get("output_dir")
    if not out_dir: 
        return
        
    try:
        out_path = Path(out_dir) / "zfisher_session.json"
        with open(out_path, 'w') as f:
            json.dump(_SESSION_DATA, f, indent=4, default=str)
        print(f"Session saved: {out_path}")
    except Exception as e:
        print(f"Failed to save session: {e}")

def load_session_file(path):
    """Loads session data from a file."""
    with open(path, 'r') as f:
        data = json.load(f)
    _SESSION_DATA.update(data)
    return _SESSION_DATA