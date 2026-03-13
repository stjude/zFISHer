import json
import logging
import threading
from pathlib import Path
import numpy as np

from .. import constants
from .log_config import attach_session_log, detach_session_log

logger = logging.getLogger(__name__)

# Thread-safety lock for all session state access
_lock = threading.Lock()

# Private global state
_SESSION_DATA = {
    "output_dir": None,
    "r1_path": None,
    "r2_path": None,
    "shift": None,
    "processed_files": {},
    "colocalization_rules": []
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
    with _lock:
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
    with _lock:
        _SESSION_DATA[key] = value
        _save_session_unlocked()

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
    with _lock:
        if "processed_files" not in _SESSION_DATA:
            _SESSION_DATA["processed_files"] = {}

        file_info = {
            'path': str(path),
            'type': layer_type, # e.g., 'points', 'labels', 'image', 'vectors', 'report'
        }
        if metadata:
            file_info.update(metadata)

        _SESSION_DATA["processed_files"][layer_name] = file_info
        _save_session_unlocked()
    logger.info("Saved processed file: %s (%s) -> %s", layer_name, layer_type, path)

def remove_processed_file(layer_name):
    """
    Removes a layer's entry from the session's processed_files registry and saves.

    Parameters
    ----------
    layer_name : str
        The name of the layer to remove.
    """
    with _lock:
        _SESSION_DATA.get("processed_files", {}).pop(layer_name, None)
        _save_session_unlocked()

def get_nuclear_channel():
    """
    Returns the resolved nuclear channel name for the current session.

    Falls back to ``constants.DAPI_CHANNEL_NAME`` when no nuclear channel
    has been stored yet (e.g. sessions created before this feature existed).
    """
    return get_data("nuclear_channel", constants.DAPI_CHANNEL_NAME)


def clear_session():
    """
    Resets the in-memory session data to its default, empty state.
    """
    global _SESSION_DATA
    detach_session_log()
    with _lock:
        _SESSION_DATA.clear()
        _SESSION_DATA.update({
            "output_dir": None,
            "r1_path": None,
            "r2_path": None,
            "shift": None,
            "processed_files": {},
            "colocalization_rules": []
        })

def initialize_new_session(output_dir, r1_path, r2_path, progress_callback=None):
    """
    Core Logic: Initializes directories and prepares files for processing.
    This can be called by the Widget OR a headless script.
    """
    output_dir = Path(output_dir)
    session_file = output_dir / constants.SESSION_FILENAME
    if session_file.exists():
        return False

    # 1. Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in [constants.SEGMENTATION_DIR, constants.ALIGNED_DIR,
                   constants.CAPTURES_DIR, constants.INPUT_DIR, constants.REPORTS_DIR,
                   constants.LOGS_DIR]:
        (output_dir / folder).mkdir(exist_ok=True)

    # 2. Reset global state
    clear_session()
    attach_session_log(output_dir, session_filename=constants.SESSION_FILENAME)
    update_data("output_dir", str(output_dir))
    update_data("r1_path", str(r1_path))
    update_data("r2_path", str(r2_path))

    logger.info("New session created: output_dir=%s, r1=%s, r2=%s", output_dir, r1_path, r2_path)

    # 3. Headless-Ready Conversion
    # We move the conversion logic here so it happens even without napari
    from .io import load_image_session, convert_nd2_to_ome
    
    for prefix, path in [("R1", r1_path), ("R2", r2_path)]:
        if Path(path).suffix.lower() == '.nd2':
            if progress_callback:
                progress_callback(0, f"Converting {prefix} ND2...")
            img_session = load_image_session(Path(path))
            convert_nd2_to_ome(img_session, output_dir / constants.INPUT_DIR, prefix)

    return True

def _save_session_unlocked():
    """Internal save — caller must already hold _lock."""
    out_dir = _SESSION_DATA.get("output_dir")
    if not out_dir:
        return

    try:
        filename = _SESSION_DATA.get("session_filename", constants.SESSION_FILENAME)
        out_path = Path(out_dir) / filename
        with open(out_path, 'w') as f:
            json.dump(_SESSION_DATA, f, indent=4, default=str)
    except Exception as e:
        logger.error("Failed to save session: %s", e)

def save_session():
    """
    Saves the current in-memory session state to a JSON file.
    """
    with _lock:
        _save_session_unlocked()

def load_session_file(path):
    """
    Loads session data from a JSON file into memory.

    Creates a new session file (``zfisher_session_2.json``, ``_3``, etc.)
    so the original is never overwritten. All subsequent saves go to the
    new file.

    Parameters
    ----------
    path : str or Path
        The path to a ``zfisher_session*.json`` file.

    Returns
    -------
    dict
        The loaded session data.
    """
    path = Path(path)
    with open(path, 'r') as f:
        data = json.load(f)

    # Find the next available session filename in the output directory
    # Use output_dir from the loaded data (not path.parent) since that's
    # where _save_session_unlocked will actually write the file.
    out_dir = Path(data.get("output_dir", str(path.parent)))
    n = 2
    new_name = f"zfisher_session_{n}.json"
    while (out_dir / new_name).exists():
        n += 1
        new_name = f"zfisher_session_{n}.json"
    logger.info("Creating new session file: %s", new_name)

    with _lock:
        _SESSION_DATA.update(data)
        _SESSION_DATA["session_filename"] = new_name
        # Save immediately to create the new session file
        _save_session_unlocked()

    attach_session_log(out_dir, session_filename=new_name)
    logger.info("Session loaded from %s → new session file: %s", path.name, new_name)

    return _SESSION_DATA