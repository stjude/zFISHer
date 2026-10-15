"""A Python stack trace for native crashes.

A segfault, abort or access violation inside Qt, OpenGL or a compiled library
ends the process without a Python traceback, and the session log simply
stops. ``enable`` points Python's ``faulthandler`` at ``crash_<time>.log`` in a
folder, so such a crash leaves the Python stack of every thread, which names
the zFISHer and napari code that was running.

zFISHer enables it at start-up in the user's log folder and moves it into the
session's ``logs/`` folder when a session log starts; it ends with the session
log. A run that ends normally leaves no crash file: an empty one is removed at
exit, and when the log moves or ends.
"""
import atexit
import datetime
import faulthandler
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_file = None          # the open crash log, or None


def user_log_dir():
    """Where zFISHer keeps logs before a session is open."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "zFISHer"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "zFISHer" / "Logs"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "zfisher" / "logs"


def current_path():
    """The crash log faulthandler writes to now, or None."""
    return Path(_file.name) if _file is not None else None


def _close():
    """Close the crash log, and remove it if nothing was written to it."""
    global _file
    if _file is None:
        return
    path = Path(_file.name)
    try:
        _file.close()
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
    except OSError as exc:
        logger.warning("Could not tidy the crash log %s: %s", path, exc)
    _file = None


def _remove_stale(directory):
    """Remove empty crash logs of processes no longer running, left by a run
    that ended without Python's exit handlers (killed, or ``os._exit``). A log
    with something in it is kept: it is the evidence."""
    import psutil
    for path in directory.glob("crash_*_*.log"):
        try:
            pid = int(path.stem.rsplit("_", 1)[1])
            if path.stat().st_size == 0 and pid != os.getpid() and not psutil.pid_exists(pid):
                path.unlink()
        except (ValueError, OSError):
            continue


def enable(directory):
    """Write the Python stacks of a native crash to a new
    ``<directory>/crash_<time>.log``, replacing the crash log in use (removed
    if empty). Returns its path, or None if the folder cannot be written."""
    global _file
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        new = open(directory / f"crash_{stamp}_{os.getpid()}.log", "w", encoding="utf-8")
    except OSError as exc:
        logger.warning("No crash log in %s: %s", directory, exc)
        return None
    faulthandler.enable(file=new, all_threads=True)
    _close()
    _file = new
    _remove_stale(directory)
    logger.info("Native crashes are logged to %s", new.name)
    return Path(new.name)


def disable():
    """Stop writing to the crash log (removed if empty); faulthandler goes
    back to standard error when there is one."""
    _close()
    if sys.__stderr__ is not None:
        faulthandler.enable(file=sys.__stderr__, all_threads=True)
    else:
        faulthandler.disable()


atexit.register(_close)
