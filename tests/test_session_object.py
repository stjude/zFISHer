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


def test_a_bound_session_reaches_threads_only_with_their_context(tmp_path):
    import contextvars
    import threading
    mine, seen = Session(), {}
    with session.use(mine):
        assert session.current() is mine
        session.update_data("shift", [1, 1, 1])                   # acts on mine, not the default
        plain = threading.Thread(target=lambda: seen.update(plain=session.current()))
        ctx = contextvars.copy_context()
        given = threading.Thread(target=lambda: ctx.run(lambda: seen.update(given=session.current())))
        plain.start(), given.start(), plain.join(), given.join()
    assert seen == {"plain": session.current(), "given": mine} and session.current() is not mine
    assert mine.get("shift") == [1, 1, 1] and session.get_data("shift") is None


def test_a_pipeline_run_leaves_the_open_session_and_its_log_alone(tmp_path):
    import logging
    from zfisher.core import log_config
    from synthetic_rounds import run_batch, write_rounds
    gui = tmp_path / "gui"
    session.update_data("output_dir", str(gui))
    session.update_data("nuclear_channel", "Hoechst")
    log_config.attach_session_log(gui, session_filename=constants.SESSION_FILENAME)
    before = (gui / constants.SESSIONS_DIR / constants.SESSION_FILENAME).read_bytes()
    root = tmp_path / "rounds"
    root.mkdir()
    write_rounds(root)
    levels = {n: logging.getLogger(n).level for n in ("zfisher", "stdout", "stderr")}
    try:
        run_batch(root, tmp_path / "out")
        assert session.get_data("output_dir") == str(gui) and session.get_data("nuclear_channel") == "Hoechst"
        assert (gui / constants.SESSIONS_DIR / constants.SESSION_FILENAME).read_bytes() == before
        assert log_config.attached_session_log() == (gui, constants.SESSION_FILENAME)
        run_log = "".join(p.read_text() for p in (tmp_path / "out" / constants.LOGS_DIR).glob("*.log"))
        gui_log = "".join(p.read_text() for p in (gui / constants.LOGS_DIR).glob("*.log"))
        assert "stage=pipeline_total" in run_log and "stage=pipeline_total" not in gui_log
    finally:
        log_config.detach_session_log()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
