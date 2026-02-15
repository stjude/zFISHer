import json
from pathlib import Path
import numpy as np

from .. import constants

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
    """
    Sets the global session loading state.

    This is used to prevent certain UI events (like auto-selecting layers)
    from firing while a session is being loaded from a file.

    Parameters
    ----------
    state : bool
        True if a session is being loaded, False otherwise.
    """
    global _is_loading
    _is_loading = state

def is_loading():
    """
    Checks if a session is currently being loaded.

    Returns
    -------
    bool
        True if a session is being loaded.
    """
    return _is_loading

def get_data(key=None, default=None):
    """
    Retrieves a value from the session data dictionary.

    Parameters
    ----------
    key : str, optional
        The key of the data to retrieve. If None, returns the entire
        session data dictionary.
    default : any, optional
        The default value to return if the key is not found.

    Returns
    -------
    any
        The requested value or the entire session dictionary.
    """
    if key:
        return _SESSION_DATA.get(key, default)
    return _SESSION_DATA

def update_data(key, value):
    """
    Updates a key-value pair in the session data and saves the session.

    Parameters
    ----------
    key : str
        The key of the data to update.
    value : any
        The new value to set.
    """
    _SESSION_DATA[key] = value
    save_session()

def set_processed_file(layer_name, path, layer_type: str, metadata: dict = None):
    """
    Registers a processed file's path and metadata, then saves the session.

    Parameters
    ----------
    layer_name : str
        The name of the layer associated with the file.
    path : str or Path
        The path to the saved file.
    layer_type : str
        The type of layer (e.g., 'points', 'labels', 'image').
    metadata : dict, optional
        An optional dictionary for additional metadata (e.g., subtype).
    """
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
    """
    Resets the in-memory session data to its default, empty state.
    """
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
    Initializes a new session.

    This function clears any existing session data, creates the required
    directory structure, and saves a new session file.

    Parameters
    ----------
    output_dir : str or Path
        The root directory for the new session.
    r1_path, r2_path : str or Path
        Paths to the input files for Round 1 and Round 2.

    Returns
    -------
    bool
        True on success, False if a session file already exists in the
        target directory.
    """
    output_dir = Path(output_dir)
    session_file = output_dir / constants.SESSION_FILENAME
    if session_file.exists():
        return False

    # Create all necessary subdirectories
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / constants.SEGMENTATION_DIR).mkdir(exist_ok=True)
    (output_dir / constants.ALIGNED_DIR).mkdir(exist_ok=True)
    (output_dir / constants.CAPTURES_DIR).mkdir(exist_ok=True)
    (output_dir / constants.INPUT_DIR).mkdir(exist_ok=True)
    (output_dir / constants.REPORTS_DIR).mkdir(exist_ok=True)

    # Reset and update session state
    clear_session()
    update_data("output_dir", str(output_dir))
    update_data("r1_path", str(r1_path))
    update_data("r2_path", str(r2_path))
    # No need to call save_session() here, as update_data does it automatically
    return True

def save_session():
    """
    Saves the current in-memory session state to a JSON file.
    """
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
    """
    Loads session data from a JSON file into memory.

    Parameters
    ----------
    path : str or Path
        The path to the `zfisher_session.json` file.

    Returns
    -------
    dict
        The loaded session data.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    _SESSION_DATA.update(data)
    return _SESSION_DATA