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
    "colocalization_rules": [],
    "tri_colocalization_rules": [],
    "puncta_params": {},
}
_is_loading = False


def _sessions_dir(out_dir):
    """Directory where session JSONs are written — a ``sessions/`` subfolder of
    the output directory."""
    return Path(out_dir) / constants.SESSIONS_DIR


def _session_save_path(out_dir, filename):
    """Canonical write path for a session JSON, creating ``sessions/`` if needed."""
    d = _sessions_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def _session_exists(out_dir, filename):
    """True if a session named ``filename`` exists in either the canonical
    ``sessions/`` subfolder or the legacy output-dir root (so numbering and the
    overwrite guard account for both old and new layouts)."""
    return (_sessions_dir(out_dir) / filename).exists() or (Path(out_dir) / filename).exists()


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
            "colocalization_rules": [],
            "tri_colocalization_rules": [],
            "puncta_params": {},
        })

def initialize_new_session(output_dir, r1_path, r2_path, progress_callback=None):
    """
    Core Logic: Initializes directories and prepares files for processing.
    This can be called by the Widget OR a headless script.
    """
    output_dir = Path(output_dir)
    # Refuse to clobber an existing session in either the new sessions/ subfolder
    # or the legacy output-dir root.
    if _session_exists(output_dir, constants.SESSION_FILENAME):
        return False

    # 1. Create directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    for folder in [constants.SEGMENTATION_DIR, constants.ALIGNED_DIR,
                   constants.CAPTURES_DIR, constants.INPUT_DIR, constants.REPORTS_DIR,
                   constants.LOGS_DIR, constants.SESSIONS_DIR]:
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
        out_path = _session_save_path(out_dir, filename)
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

    When loading an existing session, creates a new numbered session file
    (e.g., ``session_2.json``) to avoid overwriting the original. This only
    applies when loading -- new sessions always start as ``session_1.json``.

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

    # When loading an existing session, create a new session JSON file (session_2, _3, etc.)
    # instead of overwriting the original. This preserves audit history and prevents
    # accidental loss of a saved state.
    # Resolve the output-dir ROOT. Prefer the stored value; otherwise derive it
    # from the file location — a file under <root>/sessions/ implies the root is
    # its grandparent (the file may live in the legacy root or the new subfolder).
    stored = data.get("output_dir")
    if stored:
        out_dir = Path(stored)
    elif path.parent.name == constants.SESSIONS_DIR:
        out_dir = path.parent.parent
    else:
        out_dir = path.parent

    # Find the next free number across BOTH layouts so new files (written to
    # sessions/) never collide or duplicate a number already used in the root.
    n = 2
    while _session_exists(out_dir, f"zfisher_session_{n}.json"):
        n += 1
    new_name = f"zfisher_session_{n}.json"
    logger.info("Creating new session file: %s", new_name)

    with _lock:
        _SESSION_DATA.update(data)
        _SESSION_DATA["session_filename"] = new_name
        # Ensure output_dir is set so the new file actually saves (legacy sessions
        # that predate stored output_dir fall back to the derived root). Use a
        # None-aware check: clear_session() pre-seeds output_dir=None, so
        # setdefault would not fill it.
        if not _SESSION_DATA.get("output_dir"):
            _SESSION_DATA["output_dir"] = str(out_dir)
        # Ensure keys added after the session was originally saved are present
        _SESSION_DATA.setdefault("colocalization_rules", [])
        _SESSION_DATA.setdefault("tri_colocalization_rules", [])
        _SESSION_DATA.setdefault("processed_files", {})
        _SESSION_DATA.setdefault("puncta_params", {})
        # Save immediately to create the new session file
        _save_session_unlocked()

    attach_session_log(out_dir, session_filename=new_name)
    logger.info("Session loaded from %s → new session file: %s", path.name, new_name)

    return _SESSION_DATA