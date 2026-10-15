"""Native crashes and the errors napari catches leave evidence.

A segfault in Qt or OpenGL used to end zFISHer with no traceback, and napari
turned exceptions in its event callbacks into notifications that zFISHer
blocks, so they vanished. Now ``crash_log`` points ``faulthandler`` at a
``crash_<time>.log`` (the user's log folder, then the session's ``logs/``),
which a normal run leaves behind empty and removes, and
``ui.diagnostics.log_caught_errors`` logs each error napari catches.

Run:  python -m pytest tests/test_crash_diagnostics.py -q
"""
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

REPO = Path(__file__).resolve().parents[1]


def _child(code, tmp_path):
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                          timeout=120, env=dict(os.environ, PYTHONPATH=str(REPO)))


def test_a_native_crash_leaves_the_python_stack(tmp_path):
    result = _child(f"""
from zfisher.core import crash_log
crash_log.enable({str(tmp_path)!r})
def paint_the_overlay():
    import ctypes
    ctypes.string_at(0)          # a segmentation fault, as in a native Qt or GL crash
paint_the_overlay()
""", tmp_path)
    assert result.returncode != 0
    logs = list(tmp_path.glob("crash_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text()
    assert "Fatal Python error" in text and "paint_the_overlay" in text


def test_a_normal_run_leaves_no_crash_log(tmp_path):
    result = _child(f"from zfisher.core import crash_log; crash_log.enable({str(tmp_path)!r}); print('done')",
                    tmp_path)
    assert result.returncode == 0 and "done" in result.stdout
    assert not list(tmp_path.glob("crash_*.log"))


def test_the_crash_log_moves_and_the_empty_one_is_removed(tmp_path):
    from zfisher.core import crash_log
    try:
        first = crash_log.enable(tmp_path / "user")
        second = crash_log.enable(tmp_path / "session" / "logs")
        assert not first.exists() and second.exists() and crash_log.current_path() == second
    finally:
        crash_log.disable()
    assert not second.exists() and crash_log.current_path() is None


def test_a_session_log_takes_the_crash_log_over(tmp_path):
    from zfisher.core import crash_log, log_config
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    try:
        log_config.attach_session_log(tmp_path)
        assert crash_log.current_path().parent == tmp_path / "logs"
        log_config.detach_session_log()
        assert crash_log.current_path() is None                  # back to standard error
        assert not list((tmp_path / "logs").glob("crash_*.log"))
    finally:
        log_config.detach_session_log()
        crash_log.disable()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def test_errors_napari_catches_are_logged(caplog):
    from zfisher.ui.diagnostics import log_caught_errors
    sent = []
    manager = types.SimpleNamespace(dispatch=sent.append)
    log_caught_errors(manager)
    log_caught_errors(manager)                           # once only
    try:
        raise ValueError("cannot convert float NaN to integer")
    except ValueError as exc:
        error = exc
    with caplog.at_level(logging.WARNING, logger="zfisher"):
        manager.dispatch(types.SimpleNamespace(exception=error, message=str(error)))
        manager.dispatch(types.SimpleNamespace(exception=None, message="a napari warning"))
    assert len(sent) == 2                                # still passed on, to the blocked toasts
    first, second = caplog.records[-2:]
    assert first.levelno == logging.ERROR and "cannot convert float NaN" in first.getMessage()
    assert first.exc_info[1] is error
    assert second.levelno == logging.WARNING and "a napari warning" in second.getMessage()


def test_napari_s_own_error_path_reaches_the_log(caplog, monkeypatch):
    pytest.importorskip("napari")
    from napari.utils.notifications import notification_manager
    from zfisher.ui.diagnostics import log_caught_errors
    monkeypatch.setattr(notification_manager, "dispatch", notification_manager.dispatch)
    monkeypatch.setattr(notification_manager, "catch_error", True)
    log_caught_errors(notification_manager)
    with notification_manager.notification_ready.blocker():   # as zFISHer runs it
        with caplog.at_level(logging.ERROR, logger="zfisher"):
            try:
                raise OverflowError("cannot convert float infinity to integer")
            except OverflowError as exc:
                notification_manager.receive_error(type(exc), exc, exc.__traceback__)
    assert any("cannot convert float infinity" in r.getMessage() for r in caplog.records)


def test_empty_logs_of_finished_processes_are_removed_and_full_ones_kept(tmp_path):
    from zfisher.core import crash_log
    gone = 999_999_999                                   # no such process
    (tmp_path / f"crash_2026-10-01_000000_{gone}.log").write_text("")
    (tmp_path / f"crash_2026-10-01_000001_{gone}.log").write_text("Fatal Python error: Segmentation fault")
    try:
        current = crash_log.enable(tmp_path)
        names = sorted(p.name for p in tmp_path.glob("crash_*.log"))
        assert names == sorted([f"crash_2026-10-01_000001_{gone}.log", current.name])
    finally:
        crash_log.disable()
