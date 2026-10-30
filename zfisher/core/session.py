import contextvars
import copy
import json
import logging
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .. import constants
from .log_config import attach_session_log, detach_session_log
from .timing import timed_stage

logger = logging.getLogger(__name__)

# The keys every session starts with.
_DEFAULTS = {
    "output_dir": None,
    "r1_path": None,
    "r2_path": None,
    "shift": None,
    "processed_files": {},
    "colocalization_rules": [],
    "tri_colocalization_rules": [],
    "puncta_params": {},
}


def _fresh():
    return copy.deepcopy(_DEFAULTS)


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


def _reanchor_path(path_str, stored_root, real_root, keep_existing_original=False):
    """Re-point a path saved under the session's stored output root onto
    ``real_root``, the folder the session file actually lives in now.

    Makes an output folder portable: a session JSON copied/downloaded from a
    different machine carries that machine's absolute paths, which don't exist
    locally, and a folder copied on the same machine carries paths into the
    folder it was copied from. Preference order:

    1. A path under the stored output root is re-based onto ``real_root`` —
       matching the root case-insensitively and across path separators, so it
       works between Windows/macOS/Linux and across drive-letter casing. This
       holds even when the old path still exists: a copy must never read or
       write the files of the folder it was copied from. With
       ``keep_existing_original`` (for raw inputs, which are only read), the
       old path is kept when the re-based one is missing and the old one exists.
    2. A path outside the stored root that exists as-is is kept (an input file
       that lives outside the output folder).
    3. Else re-anchor from the LAST known output subfolder in the path
       (``reports/``, ``segmentation/``, …) — but only if the result actually
       exists, so we never fabricate a path that points at an unrelated file.

    Falls back to the original string if none apply.
    """
    if not path_str:
        return path_str

    parts = _path_components(path_str)

    # 1) Re-base against the stored output root (OS- and case-insensitive).
    if stored_root:
        root = _path_components(stored_root)
        if root and len(parts) > len(root) and \
                [c.lower() for c in parts[:len(root)]] == [c.lower() for c in root]:
            rebased = str(Path(real_root).joinpath(*parts[len(root):]))
            if keep_existing_original and not _path_exists(rebased) and _path_exists(path_str):
                return path_str
            return rebased

    # 2) Keep a path outside the output folder that exists as-is.
    if _path_exists(path_str):
        return path_str

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

class Session:
    """One analysis session: its settings and registry of files, the lock that
    guards them, and the session JSON they are saved to.

    A change saves the file at once, as zFISHer always has, unless it is made
    inside ``batch_writes()``, which saves once at the end. ``dirty`` says
    whether there are changes the file does not have yet.

    The module-level functions (``get_data``, ``update_data``, ...) act on one
    default instance, ``current()``; code that runs several sessions in one
    process, or should not touch the GUI's, makes its own ``Session``.
    """

    def __init__(self, data=None):
        self._data = _fresh() if data is None else data
        self._lock = threading.RLock()
        self._dirty = False
        self._deferred = 0
        self._read_only = False

    @property
    def read_only(self):
        """True for a session opened with ``load_into(path, read_only=True)``:
        it can be read, and any change raises ``PermissionError``."""
        return self._read_only

    # --- reading -----------------------------------------------------------
    def get(self, key, default=None):
        """The value of ``key``, or ``default``."""
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self):
        """A deep copy of every key: safe to keep or change."""
        with self._lock:
            return copy.deepcopy(self._data)

    @property
    def output_dir(self):
        """The session's output folder as a ``Path``, or None before one is set."""
        value = self.get("output_dir")
        return Path(value) if value else None

    @property
    def lock(self):
        """The session's (re-entrant) lock: hold it around a read, a decision
        and a write that must not interleave with another thread's."""
        return self._lock

    @property
    def dirty(self):
        """True while changes are waiting for ``save()``."""
        return self._dirty

    def nuclear_channel(self):
        """The nuclear channel's name; ``constants.DAPI_CHANNEL_NAME`` for
        sessions made before it was stored."""
        return self.get("nuclear_channel", constants.DAPI_CHANNEL_NAME)

    # --- changing ----------------------------------------------------------
    def set(self, key, value):
        """Set ``key`` to ``value`` and save (see ``batch_writes``)."""
        with self._lock:
            self._data[key] = value
            self._changed()

    def update_nested(self, key, subkey, value):
        """Set ``self[key][subkey] = value``, the dict under ``key`` created if
        missing: the read-change-write of a dict-valued key in one step under
        the lock, so two threads cannot lose each other's change."""
        with self._lock:
            inner = self._data.get(key)
            if not isinstance(inner, dict):
                inner = {}
            inner = dict(inner)
            inner[subkey] = value
            self._data[key] = inner
            self._changed()

    def set_processed_file(self, layer_name, path, layer_type, metadata=None):
        """Register the file of ``layer_name`` (its path, layer type and any
        ``metadata``) and save."""
        with self._lock:
            files = self._data.setdefault("processed_files", {})
            file_info = {'path': str(path), 'type': layer_type}
            if metadata:
                file_info.update(metadata)
            files[layer_name] = file_info
            self._changed()

    def remove_processed_file(self, layer_name):
        """Drop ``layer_name`` from the registry and save."""
        with self._lock:
            self._data.get("processed_files", {}).pop(layer_name, None)
            self._changed()

    def reset(self):
        """Back to an empty session, in memory only."""
        with self._lock:
            self._data.clear()
            self._data.update(_fresh())
            self._dirty = False
            self._read_only = False

    # --- saving ------------------------------------------------------------
    def _changed(self):
        if self._read_only:
            raise PermissionError("This session was opened read-only; open it with Load Session to change it.")
        self._dirty = True
        if not self._deferred:
            self.save()

    @contextmanager
    def batch_writes(self):
        """Inside this block changes are kept in memory; the file is written
        once when the outermost block ends, also when it ends with an error."""
        with self._lock:
            self._deferred += 1
        failed = False
        try:
            yield self
        except BaseException:
            failed = True
            raise
        finally:
            with self._lock:
                self._deferred -= 1
                if not self._deferred and self._dirty:
                    try:
                        self.save()
                    except SessionSaveError:
                        if not failed:
                            raise
                        logger.exception("Session not saved after an error inside batch_writes")

    def save(self):
        """Write the session JSON now (atomically; see ``_write_text_atomic``).
        Does nothing before an output folder is set. Raises
        ``SessionSaveError``, leaving the previous file whole."""
        with self._lock:
            out_dir = self._data.get("output_dir")
            if not out_dir:
                return
            out_path = None
            try:
                # The version that last wrote this file; created_with_zfisher keeps the first.
                from .provenance import software_version
                self._data["zfisher_version"] = software_version()
                filename = self._data.get("session_filename", constants.SESSION_FILENAME)
                out_path = _session_save_path(out_dir, filename)
                # Serialised first, so a value that cannot be written never touches the file.
                text = json.dumps(self._data, indent=4, default=str)
                _write_text_atomic(out_path, text)
            except Exception as e:
                where = out_path or Path(out_dir) / constants.SESSIONS_DIR
                logger.error("Failed to save session to %s: %s", where, e)
                raise SessionSaveError(f"Could not save the session file {where}: {e}") from e
            self._dirty = False

    def load_into(self, path, read_only=False):
        """Replace this session with the session file at ``path``, saved under
        the next free number (``zfisher_session_<n>.json``) so the file read
        is never overwritten. Returns the session's data. See
        ``load_session_file`` for how paths are re-anchored.

        With ``read_only`` nothing is written, now or later: the session keeps
        the name of the file it was read from, and changes raise."""
        data, new_name = _read_session_file(Path(path))
        from .params import SCHEMA_VERSION
        if read_only:
            new_name = Path(path).name
        with self._lock:
            # Nothing of the previous session may survive into this one: a file
            # from before a key existed (a saved registration, the canvas scale,
            # the nuclear channel) would otherwise inherit the value of whatever
            # session was open before it.
            self._data.clear()
            self._data.update(data)
            self._data["session_filename"] = new_name
            # Ensure keys added after the session was originally saved are present
            self._data.setdefault("colocalization_rules", [])
            self._data.setdefault("tri_colocalization_rules", [])
            self._data.setdefault("processed_files", {})
            self._data.setdefault("puncta_params", {})
            # Sessions from before the field are format 1, the format this writes.
            self._data.setdefault("schema_version", SCHEMA_VERSION)
            self._read_only = bool(read_only)
            self._dirty = False
            if not read_only:
                # Save immediately to create the new session file
                logger.info("Creating new session file: %s", new_name)
                self.save()
            return self._data


# --- The default session and the module-level functions on it -------------------

_default = Session()
# Older names for the default session's state, kept for code that used them.
_SESSION_DATA = _default._data
_lock = _default._lock
_bound = contextvars.ContextVar("zfisher_session", default=None)


def current():
    """The session the module-level functions act on: the one bound with
    ``use()`` in this thread or task, else the default session (the GUI's)."""
    bound = _bound.get()
    return bound if bound is not None else _default


@contextmanager
def use(session):
    """Run a block on ``session``: every module-level function, and so every
    core function that uses them, acts on it instead of the default session.
    Bound per thread and per task; a thread started inside the block sees it
    only if it is given the block's context (``contextvars.copy_context``)."""
    token = _bound.set(session)
    try:
        yield session
    finally:
        _bound.reset(token)


def get_data(key=None, default=None):
    """
    Retrieves a value from the session data dictionary.

    Parameters
    ----------
    key : str, optional
        The key of the data to retrieve. If None, returns the entire
        session data dictionary (live: changing it changes the session; use
        ``current().snapshot()`` for a copy).
    default : any, optional
        The default value to return if the key is not found.

    Returns
    -------
    any
        The requested value or the entire session dictionary.
    """
    if key:
        return current().get(key, default)
    return current()._data


def update_data(key, value):
    """Updates a key-value pair in the session data and saves the session."""
    current().set(key, value)


def update_nested(key, subkey, value):
    """Set ``session[key][subkey] = value`` as one step under the session's
    lock (see ``Session.update_nested``), through ``get_data`` and
    ``update_data``."""
    with current().lock:
        inner = get_data(key, default={})
        inner = dict(inner) if isinstance(inner, dict) else {}
        inner[subkey] = value
        update_data(key, inner)


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
    current().set_processed_file(layer_name, path, layer_type, metadata)
    logger.info("Saved processed file: %s (%s) -> %s", layer_name, layer_type, path)


def remove_processed_file(layer_name):
    """Removes a layer's entry from the session's processed_files registry and saves."""
    current().remove_processed_file(layer_name)


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
    detach_session_log()
    current().reset()

@timed_stage("session_init")
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
    from .provenance import software_version
    from .params import SCHEMA_VERSION
    update_data("schema_version", SCHEMA_VERSION)
    update_data("created_with_zfisher", software_version())
    # What each round's image is, so a file pointed to later can be checked.
    from .io import record_raw_identity
    record_raw_identity("R1", r1_path)
    record_raw_identity("R2", r2_path)

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

class SessionSaveError(OSError):
    """The session file could not be written. The previous file is left as it
    was; the change is kept in memory and goes out with the next save."""


_REPLACE_ATTEMPTS = 5      # os.replace can fail briefly on Windows while another
_REPLACE_WAIT_S = 0.05     # program (a virus scanner, a sync client) has the file open


def _write_text_atomic(path, text):
    """Write ``text`` to ``path`` so that a reader, or a crash, only ever sees
    the old file or the new one: write a temporary file beside it, flush it to
    disk, then swap it into place."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(_REPLACE_WAIT_S)
    finally:
        tmp.unlink(missing_ok=True)


def save_session():
    """
    Saves the current in-memory session state to a JSON file.
    """
    current().save()


def _read_session_file(path):
    """Read and check the session JSON at ``path``, re-anchor its paths onto
    the folder it is in now, and pick the name it will be saved under.
    Returns ``(data, new_name)``."""
    with open(path, 'r') as f:
        data = json.load(f)
    from .params import SCHEMA_VERSION
    made_with = data.get("schema_version", 1)
    if not isinstance(made_with, int) or made_with > SCHEMA_VERSION:
        raise ValueError(
            f"{path.name} was written by a newer zFISHer (session format {made_with}; this zFISHer reads "
            f"up to {SCHEMA_VERSION}, {data.get('zfisher_version', 'unknown version')}). Open it with that "
            f"version or later.")

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

    # Re-anchor every stored path from the stored output root onto the real one,
    # so relocated/downloaded folders resolve their processed files and a copied
    # folder uses its own files, not those of the folder it was copied from.
    stored_out_dir = data.get("output_dir")
    data["output_dir"] = str(out_dir)
    for key in ("r1_path", "r2_path"):
        if data.get(key):
            # Raw inputs are only read, so one the copy lacks is read from where it was.
            data[key] = _reanchor_path(data[key], stored_out_dir, out_dir, keep_existing_original=True)
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
    return data, new_name


def load_session_file(path):
    """
    Loads session data from a JSON file into memory, replacing the session
    that was there: nothing of the previous one is kept.

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
    data = current().load_into(path)
    new_name = data["session_filename"]
    attach_session_log(Path(data["output_dir"]), session_filename=new_name)
    logger.info("Session loaded from %s → new session file: %s", path.name, new_name)
    return data
