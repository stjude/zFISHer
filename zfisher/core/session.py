import json
import logging
import re
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


def _output_subdirs():
    """The standard top-level subfolders of an output directory."""
    return (constants.REPORTS_DIR, constants.SEGMENTATION_DIR, constants.ALIGNED_DIR,
            constants.CAPTURES_DIR, constants.INPUT_DIR, constants.LOGS_DIR,
            constants.SESSIONS_DIR)


def _path_components(path_str):
    """Split a path string on BOTH separators, so a Windows path parses on POSIX
    and vice-versa (``Path`` only understands the runtime flavour). Drops empty
    segments and ``.``."""
    return [c for c in re.split(r"[\\/]+", str(path_str)) if c not in ("", ".")]


def _path_exists(path_str):
    """Existence check that never raises on a malformed/foreign path string."""
    try:
        return bool(path_str) and Path(path_str).exists()
    except OSError:
        return False


def _reanchor_path(path_str, stored_root, real_root):
    """Re-point a path that was saved on another machine onto ``real_root``.

    Makes an output folder portable: a session JSON copied/downloaded from a
    different machine carries that machine's absolute paths, which don't exist
    locally. We re-anchor each one onto the directory where the session file
    actually lives now. Preference order:

    1. If the file already exists as-is, keep it (same machine / same path).
    2. Re-base it from the session's stored output root onto ``real_root`` —
       matching the root case-insensitively and across path separators, so it
       works between Windows/macOS/Linux and across drive-letter casing.
    3. Else re-anchor from the LAST known output subfolder in the path
       (``reports/``, ``segmentation/``, …) — but only if the result actually
       exists, so we never fabricate a path that points at an unrelated file.

    Falls back to the original string if none apply (e.g. an external input file
    that lives outside the output folder).
    """
    if not path_str:
        return path_str
    try:
        if Path(path_str).exists():
            return path_str
    except OSError:
        pass

    parts = _path_components(path_str)

    # 2) Re-base against the stored output root (OS- and case-insensitive).
    if stored_root:
        root = _path_components(stored_root)
        if root and len(parts) > len(root) and \
                [c.lower() for c in parts[:len(root)]] == [c.lower() for c in root]:
            return str(Path(real_root).joinpath(*parts[len(root):]))

    # 3) Re-anchor from the LAST recognised output subfolder, if it resolves.
    known = {s.lower() for s in _output_subdirs()}
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in known:
            candidate = Path(real_root).joinpath(*parts[i:])
            if candidate.exists():
                return str(candidate)
            break  # only the last subfolder occurrence is a plausible anchor

    return path_str


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
    #
    # The output-dir ROOT is wherever this session file ACTUALLY lives now — a
    # file under <root>/sessions/ implies the root is its grandparent; a legacy
    # file in the root implies the root is its parent. We trust the file's real
    # location over the absolute output_dir baked into the JSON, so a folder
    # copied or downloaded from another machine still loads.
    if path.parent.name == constants.SESSIONS_DIR:
        out_dir = path.parent.parent
    else:
        out_dir = path.parent

    # Re-anchor every stored path from the originating machine's output root onto
    # the real one, so relocated/downloaded folders resolve their processed files.
    stored_out_dir = data.get("output_dir")
    data["output_dir"] = str(out_dir)
    for key in ("r1_path", "r2_path"):
        if data.get(key):
            data[key] = _reanchor_path(data[key], stored_out_dir, out_dir)
    for name, info in data.get("processed_files", {}).items():
        if isinstance(info, dict) and info.get("path"):
            resolved = _reanchor_path(info["path"], stored_out_dir, out_dir)
            # Recover from a backup left by a removal/crash (or carried in a
            # download) when the canonical file is missing.
            if not _path_exists(resolved) and _path_exists(resolved + ".bak"):
                logger.info("Session load: recovering '%s' from backup %s.bak", name, resolved)
                resolved = resolved + ".bak"
            info["path"] = resolved
            if not _path_exists(resolved):
                logger.warning(
                    "Session load: file for layer '%s' not found after re-anchor "
                    "(partial download?): %s", name, info["path"])

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
        # output_dir was already re-anchored to the real location above.
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