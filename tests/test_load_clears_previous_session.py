"""Loading a session replaces the one in memory; nothing of the previous one is kept.

``session.load_session_file`` merged the file into the session dictionary, so a
key the file did not have kept the value of whatever session was open before:
a session made before 2.1.0, opened after a newer one, inherited that one's
saved registration, canvas scale and nuclear channel, and re-detection would
have warped its puncta with another dataset's transform.

Core only, no napari.

Run:  python -m pytest tests/test_load_clears_previous_session.py -q
"""
import json
import logging
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from zfisher import constants
from zfisher.core import session

PREVIOUS_ONLY = ("nuclear_channel", "canvas_scale", "canvas_offset_pixels", "registration", "shift")


@pytest.fixture
def sessions(tmp_path):
    """Session A open in memory with every kind of key; session B on disk, an
    older file with only the paths and an empty registry. Logging is put back
    afterwards: attaching a session log opens the zfisher loggers up to DEBUG."""
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    a, b = tmp_path / "a", tmp_path / "b"
    (b / constants.SESSIONS_DIR).mkdir(parents=True)

    session.clear_session()
    session.update_data("output_dir", str(a))
    session.update_data("r1_path", str(a / "R1.nd2"))
    session.update_data("r2_path", str(a / "R2.nd2"))
    session.update_data("nuclear_channel", "Hoechst")
    session.update_data("canvas_scale", [0.3, 0.1, 0.1])
    session.update_data("canvas_offset_pixels", [0, 4, -2])
    session.update_data("shift", [0.0, 3.4, -2.2])
    session.update_data("registration", {"bspline_path": str(a / constants.ALIGNED_DIR / "registration_bspline.tfm")})
    session.update_data("puncta_params", {"FITC": {"threshold": 0.5}})
    session.update_data("colocalization_rules", [{"a": "FITC", "b": "Cy5"}])
    session.set_processed_file("Aligned R1 - FITC_puncta", str(a / constants.REPORTS_DIR / "x.csv"), "points")

    b_file = b / constants.SESSIONS_DIR / constants.SESSION_FILENAME
    b_file.write_text(json.dumps({"output_dir": str(b), "r1_path": str(b / "R1.nd2"),
                                  "r2_path": str(b / "R2.nd2"), "processed_files": {}}))
    yield types.SimpleNamespace(a=a, b=b, file=b_file)
    session.clear_session()
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def test_loading_a_session_keeps_nothing_of_the_previous_one(sessions):
    session.load_session_file(sessions.file)

    data = session.get_data()
    assert data["output_dir"] == str(sessions.b)
    assert Path(data["r1_path"]).name == "R1.nd2" and str(sessions.a) not in data["r1_path"]
    for key in PREVIOUS_ONLY:
        assert key not in data, key
    assert session.get_nuclear_channel() == constants.DAPI_CHANNEL_NAME
    assert data["processed_files"] == {} and data["puncta_params"] == {}
    assert data["colocalization_rules"] == [] and data["tri_colocalization_rules"] == []


def test_the_new_session_file_lists_nothing_of_the_previous_one(sessions):
    session.load_session_file(sessions.file)

    written = sessions.b / constants.SESSIONS_DIR / session.get_data("session_filename")
    assert written.exists() and written.name != constants.SESSION_FILENAME   # the numbered copy
    on_disk = json.loads(written.read_text())
    assert on_disk["output_dir"] == str(sessions.b) and on_disk["processed_files"] == {}
    assert not any(key in on_disk for key in PREVIOUS_ONLY)


def test_what_the_file_has_is_what_the_session_has(sessions):
    sessions.file.write_text(json.dumps({"output_dir": str(sessions.b), "nuclear_channel": "DAPI-B",
                                         "canvas_scale": [0.5, 0.2, 0.2], "processed_files": {}}))

    session.load_session_file(sessions.file)

    assert session.get_nuclear_channel() == "DAPI-B"
    assert session.get_data("canvas_scale") == [0.5, 0.2, 0.2]
    assert "registration" not in session.get_data()
