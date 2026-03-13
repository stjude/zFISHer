"""
Session-scoped file logging for zFISHer.

When a session is created or loaded, `attach_session_log` adds a
`FileHandler` to the root logger so that every module's log output
(plus redirected stdout/stderr) is captured in a timestamped log file
inside the session's ``logs/`` directory.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from .. import constants

logger = logging.getLogger(__name__)

# Module-level state
_file_handler = None  # type: logging.FileHandler | None
_stdout_redirector = None
_stderr_redirector = None
_original_stdout = None
_original_stderr = None

_LOG_FORMAT = "%(asctime)s | %(name)s [%(levelname)s] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _StreamToLogger:
    """Redirect a stream (stdout/stderr) to a logger at a given level,
    while still writing to the original stream so the console works."""

    def __init__(self, log: logging.Logger, level: int, original_stream):
        self._log = log
        self._level = level
        self._original = original_stream
        self._buffer = ""

    def write(self, message):
        # Always write to the original stream (console)
        if self._original:
            self._original.write(message)
        # Buffer and log complete lines
        if message and message.strip():
            self._log.log(self._level, message.rstrip())

    def flush(self):
        if self._original:
            self._original.flush()

    def isatty(self):
        return False


def attach_session_log(output_dir, session_filename=None):
    """Start logging to a file in ``<output_dir>/logs/``.

    Safe to call multiple times — detaches any previous handler first.
    """
    global _file_handler, _stdout_redirector, _stderr_redirector
    global _original_stdout, _original_stderr

    detach_session_log()

    output_dir = Path(output_dir)
    logs_dir = output_dir / constants.LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = logs_dir / f"zfisher_{timestamp}.log"

    _file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))

    root = logging.getLogger()
    root.addHandler(_file_handler)
    # Only open up DEBUG on our own loggers — leave root level alone so
    # napari's notification handler doesn't see our INFO/DEBUG messages.
    logging.getLogger("zfisher").setLevel(logging.DEBUG)

    # Redirect stdout/stderr so print() output is also captured.
    # Give each its own explicit level so they emit even though root
    # may be at WARNING.
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    _stdout_logger = logging.getLogger("stdout")
    _stdout_logger.setLevel(logging.DEBUG)
    _stderr_logger = logging.getLogger("stderr")
    _stderr_logger.setLevel(logging.DEBUG)
    _stdout_redirector = _StreamToLogger(_stdout_logger, logging.INFO, _original_stdout)
    _stderr_redirector = _StreamToLogger(_stderr_logger, logging.DEBUG, _original_stderr)
    sys.stdout = _stdout_redirector
    sys.stderr = _stderr_redirector

    logger.info("=== zFISHer session log started: %s ===", log_path.name)
    logger.info("Output directory: %s", output_dir)
    if session_filename:
        logger.info("Session file: %s", session_filename)


def detach_session_log():
    """Remove the file handler and restore stdout/stderr."""
    global _file_handler, _stdout_redirector, _stderr_redirector
    global _original_stdout, _original_stderr

    if _file_handler:
        logger.info("=== zFISHer session log ended ===")
        root = logging.getLogger()
        root.removeHandler(_file_handler)
        _file_handler.close()
        _file_handler = None

    if _original_stdout:
        sys.stdout = _original_stdout
        _original_stdout = None
        _stdout_redirector = None
    if _original_stderr:
        sys.stderr = _original_stderr
        _original_stderr = None
        _stderr_redirector = None
