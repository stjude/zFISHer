"""Session files are written whole or not at all, and a failed save is reported.

The session JSON was written in place, so a crash, a full disk or a killed
process during the write left a truncated file, the only copy of the session's
registry. It is now written to a temporary file beside it and swapped into
place, and a save that fails raises ``SessionSaveError`` instead of only
logging, leaving the previous file as it was.

Core only.

Run:  python -m pytest tests/test_session_atomic_save.py -q
"""
import json
import logging

import pytest

from zfisher import constants
from zfisher.core import session


@pytest.fixture
def saved(tmp_path):
    """An open session with one saved key; logging put back afterwards."""
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    session.clear_session()
    session.update_data("output_dir", str(tmp_path))
    session.update_data("shift", [0.0, 1.0, 2.0])
    path = tmp_path / constants.SESSIONS_DIR / constants.SESSION_FILENAME
    yield path
    session.clear_session()
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def _leftovers(path):
    return [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")]


def test_a_save_writes_the_whole_file_and_leaves_no_temporary_file(saved):
    session.update_data("puncta_params", {"FITC": {"threshold": 0.5}})
    data = json.loads(saved.read_text())
    assert data["shift"] == [0.0, 1.0, 2.0] and data["puncta_params"] == {"FITC": {"threshold": 0.5}}
    assert _leftovers(saved) == []


def test_a_write_cut_short_leaves_the_previous_file(saved, monkeypatch):
    before = saved.read_bytes()

    def crash(fd):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(session.os, "fsync", crash)

    with pytest.raises(session.SessionSaveError, match="No space left on device"):
        session.update_data("shift", [9.0, 9.0, 9.0])

    assert saved.read_bytes() == before
    assert _leftovers(saved) == []
    assert session.get_data("shift") == [9.0, 9.0, 9.0]          # kept in memory ...
    monkeypatch.undo()
    session.save_session()                                        # ... and written by the next save
    assert json.loads(saved.read_text())["shift"] == [9.0, 9.0, 9.0]


def test_a_value_that_cannot_be_written_never_touches_the_file(saved):
    before = saved.read_bytes()

    class Unprintable:
        def __str__(self):
            raise ValueError("cannot be shown")

    with pytest.raises(session.SessionSaveError, match="cannot be shown"):
        session.update_data("bad", Unprintable())
    assert saved.read_bytes() == before and _leftovers(saved) == []


def test_a_file_held_open_briefly_is_retried(saved, monkeypatch):
    real_replace, calls = session.os.replace, []

    def busy_twice(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(13, "The process cannot access the file")
        real_replace(src, dst)
    monkeypatch.setattr(session.os, "replace", busy_twice)
    monkeypatch.setattr(session, "_REPLACE_WAIT_S", 0)

    session.update_data("shift", [3.0, 3.0, 3.0])

    assert len(calls) == 3 and json.loads(saved.read_text())["shift"] == [3.0, 3.0, 3.0]


def test_a_file_that_stays_locked_is_reported(saved, monkeypatch):
    before = saved.read_bytes()

    def locked(src, dst):
        raise PermissionError(13, "The process cannot access the file")
    monkeypatch.setattr(session.os, "replace", locked)
    monkeypatch.setattr(session, "_REPLACE_WAIT_S", 0)

    with pytest.raises(session.SessionSaveError, match=constants.SESSION_FILENAME):
        session.update_data("shift", [4.0, 4.0, 4.0])
    assert saved.read_bytes() == before and _leftovers(saved) == []


def test_a_background_puncta_save_says_so_in_the_status_bar(saved, monkeypatch):
    """A puncta save runs from napari's data event, where a raised error is
    only logged; the failure goes to the status bar and the CSV is written."""
    pytest.importorskip("napari")
    import types
    import numpy as np
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from zfisher.ui import events
    viewer = types.SimpleNamespace(status="")
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    layer = napari.layers.Points(np.array([[1.0, 2.0, 3.0]]), name="R1 - FITC_puncta")

    def full(fd):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(session.os, "fsync", full)

    events.save_puncta_layer(layer)

    assert "Could not save the session file" in viewer.status and "No space left" in viewer.status
    assert constants.puncta_csv_path(saved.parent.parent / constants.REPORTS_DIR, layer.name).exists()
