"""The ``Session`` object behind the module-level session functions.

A ``Session`` owns its settings, its file registry, its lock and its JSON
file; the module functions (``get_data``, ``update_data``, ...) act on one
default instance, so existing code is unchanged, while a second session can
be run in the same process without touching the GUI's. ``batch_writes`` saves
once at the end instead of after every change.

Run:  python -m pytest tests/test_session_object.py -q
"""
import json

import pytest

from zfisher import constants
from zfisher.core import session
from zfisher.core.session import Session


@pytest.fixture
def writes(monkeypatch):
    """Counts the session files written, by path."""
    real, written = session._write_text_atomic, []

    def counting(path, text):
        written.append(path)
        real(path, text)
    monkeypatch.setattr(session, "_write_text_atomic", counting)
    return written


@pytest.fixture(autouse=True)
def clean_default():
    session.clear_session()
    yield
    session.clear_session()


def _file(folder):
    return json.loads((folder / constants.SESSIONS_DIR / constants.SESSION_FILENAME).read_text())


def test_two_sessions_do_not_share_state(tmp_path):
    a, b = Session(), Session()
    a.set("output_dir", str(tmp_path / "a"))
    b.set("output_dir", str(tmp_path / "b"))
    a.set_processed_file("R1 - DAPI_masks", tmp_path / "a" / "m.tif", "labels")
    assert b.get("processed_files") == {} and session.get_data("output_dir") is None
    assert _file(tmp_path / "a")["processed_files"] and not _file(tmp_path / "b")["processed_files"]


def test_the_module_functions_act_on_the_default_session(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(Session, "set", lambda self, k, v: calls.append((self, k, v)))
    session.update_data("shift", [1, 2, 3])
    assert calls == [(session.current(), "shift", [1, 2, 3])]
    assert session.get_data() is session.current()._data is session._SESSION_DATA


def test_batch_writes_saves_once_at_the_end(tmp_path, writes):
    s = Session()
    s.set("output_dir", str(tmp_path))
    writes.clear()
    with s.batch_writes():
        s.set("shift", [0, 1, 2])
        with s.batch_writes():                        # nested: still one save, at the outermost end
            s.set_processed_file("x", tmp_path / "x.csv", "points")
            s.update_nested("puncta_params", "R1 - FITC", {"sigma": 1.0})
        assert writes == [] and s.dirty
    assert len(writes) == 1 and not s.dirty
    assert _file(tmp_path)["puncta_params"] == {"R1 - FITC": {"sigma": 1.0}}


def test_changes_inside_a_failing_block_are_still_saved(tmp_path, writes):
    s = Session()
    s.set("output_dir", str(tmp_path))
    with pytest.raises(RuntimeError):
        with s.batch_writes():
            s.set("shift", [4, 5, 6])
            raise RuntimeError("stage failed")
    assert _file(tmp_path)["shift"] == [4, 5, 6] and not s.dirty


def test_snapshot_is_a_copy_and_reset_empties(tmp_path):
    s = Session()
    s.set("processed_files", {"a": {"path": "p"}})
    snap = s.snapshot()
    snap["processed_files"]["b"] = {}
    assert set(s.get("processed_files")) == {"a"}
    s.reset()
    assert s.get("processed_files") == {} and s.output_dir is None and not s.dirty


def test_a_second_session_loads_a_file_without_touching_the_default(tmp_path):
    first = Session()
    first.set("output_dir", str(tmp_path))
    first.set("nuclear_channel", "Hoechst")
    session.update_data("shift", [9, 9, 9])                       # the GUI's session, in memory only
    other = Session()
    data = other.load_into(tmp_path / constants.SESSIONS_DIR / constants.SESSION_FILENAME)
    assert data["session_filename"] == "zfisher_session_2.json" and other.nuclear_channel() == "Hoechst"
    assert other.output_dir == tmp_path and session.get_data("shift") == [9, 9, 9]
    assert session.get_data("nuclear_channel") is None
