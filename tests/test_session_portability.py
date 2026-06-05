"""Tests that a session folder is portable: copied/downloaded to a different
machine or path, ``load_session_file`` re-anchors the absolute paths baked in by
the originating machine onto the folder's real location, so processed files
resolve. Runs without napari/Qt.

Run:  python -m pytest tests/test_session_portability.py -q
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from zfisher import constants
from zfisher.core import session


FOREIGN_ROOT = "C:/Users/someone_else/zFISHer_Output_proj"


@pytest.fixture
def tmp_out():
    """A temp output dir. clear_session() runs before yield (clean state) and
    after (detaches the session log handler so the dir can be deleted on Windows)."""
    session.clear_session()
    d = tempfile.mkdtemp()
    try:
        yield Path(d)
    finally:
        session.clear_session()          # releases the open logs/*.log handle
        shutil.rmtree(d, ignore_errors=True)


def _make_session_folder(out, *, in_subfolder=True, stored_root=FOREIGN_ROOT,
                         include_output_dir=True, foreign_csv=None):
    """Lay out a realistic output folder under ``out`` whose session JSON points at
    ``stored_root`` (a different machine). Returns (session_file, real_csv_path)."""
    for sub in (constants.REPORTS_DIR, constants.LOGS_DIR, constants.SESSIONS_DIR):
        (out / sub).mkdir(parents=True, exist_ok=True)
    real_csv = out / constants.REPORTS_DIR / "R1_-_FITC_puncta.csv"
    real_csv.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")

    if foreign_csv is None:
        foreign_csv = f"{stored_root}/{constants.REPORTS_DIR}/R1_-_FITC_puncta.csv"
    data = {
        "processed_files": {
            "R1 - FITC_puncta": {"path": foreign_csv, "type": "points", "subtype": "puncta_csv"},
        },
        "r1_path": f"{stored_root}/input/R1.ome.tif",
        "colocalization_rules": [],
    }
    if include_output_dir:
        data["output_dir"] = stored_root

    sfile = (out / constants.SESSIONS_DIR / "zfisher_session_1.json") if in_subfolder \
        else (out / "zfisher_session_1.json")
    sfile.write_text(json.dumps(data, indent=2))
    return sfile, real_csv


def test_reanchors_from_sessions_subfolder(tmp_out):
    sfile, real_csv = _make_session_folder(tmp_out, in_subfolder=True)
    session.load_session_file(sfile)

    assert Path(session.get_data("output_dir")) == tmp_out
    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == real_csv
    assert Path(p).exists()


def test_reanchors_legacy_root_layout(tmp_out):
    sfile, real_csv = _make_session_folder(tmp_out, in_subfolder=False)
    session.load_session_file(sfile)

    assert Path(session.get_data("output_dir")) == tmp_out
    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p).exists()


def test_subfolder_fallback_when_no_stored_output_dir(tmp_out):
    # Legacy session with no output_dir and a path under an unknown prefix:
    # re-anchor from the recognised 'reports/' subfolder.
    weird = f"D:/old/totally/different/{constants.REPORTS_DIR}/R1_-_FITC_puncta.csv"
    sfile, real_csv = _make_session_folder(
        tmp_out, in_subfolder=True, include_output_dir=False, foreign_csv=weird)
    session.load_session_file(sfile)

    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == real_csv
    assert Path(p).exists()


def test_reanchor_is_cross_os_and_case_insensitive(tmp_out):
    # Foreign path uses backslashes and a stored root that differs in drive-letter
    # and folder casing — re-base must still succeed.
    for sub in (constants.REPORTS_DIR, constants.LOGS_DIR, constants.SESSIONS_DIR):
        (tmp_out / sub).mkdir(parents=True, exist_ok=True)
    real_csv = tmp_out / constants.REPORTS_DIR / "R1_-_FITC_puncta.csv"
    real_csv.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")
    data = {
        "output_dir": "C:\\Users\\Other\\zFISHer_Out",
        "processed_files": {
            "R1 - FITC_puncta": {
                "path": "c:\\users\\OTHER\\zfisher_out\\reports\\R1_-_FITC_puncta.csv",
                "type": "points",
            },
        },
    }
    sfile = tmp_out / constants.SESSIONS_DIR / "zfisher_session_1.json"
    sfile.write_text(json.dumps(data))
    session.load_session_file(sfile)

    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == real_csv
    assert Path(p).exists()


def test_subfolder_fallback_uses_last_token_not_first(tmp_out):
    # An ancestor folder named like a subdir ('input') must not win over the real
    # 'reports' subfolder nearest the file.
    weird = f"D:/proj_root/input/sub/{constants.REPORTS_DIR}/R1_-_FITC_puncta.csv"
    sfile, real_csv = _make_session_folder(
        tmp_out, in_subfolder=True, include_output_dir=False, foreign_csv=weird)
    session.load_session_file(sfile)

    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == real_csv  # real_root/reports/..., not real_root/input/...
    assert Path(p).exists()


def test_subfolder_fallback_does_not_fabricate_missing_path(tmp_out):
    # No stored output_dir, and the target file is NOT present locally: the helper
    # must not invent a real_root path that could collide with an unrelated file.
    (tmp_out / constants.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_out / constants.SESSIONS_DIR).mkdir(parents=True, exist_ok=True)
    (tmp_out / constants.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    foreign = f"D:/old/{constants.REPORTS_DIR}/absent_layer.csv"
    data = {"processed_files": {"X_puncta": {"path": foreign, "type": "points"}}}
    sfile = tmp_out / constants.SESSIONS_DIR / "zfisher_session_1.json"
    sfile.write_text(json.dumps(data))
    session.load_session_file(sfile)

    p = session.get_data("processed_files")["X_puncta"]["path"]
    # Candidate real_root/reports/absent_layer.csv does not exist → keep original,
    # never fabricate.
    assert not Path(tmp_out / constants.REPORTS_DIR / "absent_layer.csv").exists()
    assert Path(p) == Path(foreign)


def test_recovers_from_bak_when_canonical_missing(tmp_out):
    # Canonical CSV gone but a .bak backup exists (left by a removal/crash, or
    # carried in a download): load must fall back to the backup.
    for sub in (constants.REPORTS_DIR, constants.LOGS_DIR, constants.SESSIONS_DIR):
        (tmp_out / sub).mkdir(parents=True, exist_ok=True)
    canonical = tmp_out / constants.REPORTS_DIR / "R1_-_FITC_puncta.csv"
    bak = tmp_out / constants.REPORTS_DIR / "R1_-_FITC_puncta.csv.bak"
    bak.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")  # only the backup exists
    data = {
        "output_dir": str(tmp_out),
        "processed_files": {"R1 - FITC_puncta": {"path": str(canonical), "type": "points"}},
    }
    sfile = tmp_out / constants.SESSIONS_DIR / "zfisher_session_1.json"
    sfile.write_text(json.dumps(data))
    session.load_session_file(sfile)

    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == bak
    assert Path(p).exists()


def test_existing_local_path_is_preserved(tmp_out):
    # Reloading your own session (path already resolves) must leave it untouched.
    for sub in (constants.REPORTS_DIR, constants.LOGS_DIR, constants.SESSIONS_DIR):
        (tmp_out / sub).mkdir(parents=True, exist_ok=True)
    real_csv = tmp_out / constants.REPORTS_DIR / "R1_-_FITC_puncta.csv"
    real_csv.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")
    data = {
        "output_dir": str(tmp_out),
        "processed_files": {"R1 - FITC_puncta": {"path": str(real_csv), "type": "points"}},
    }
    sfile = tmp_out / constants.SESSIONS_DIR / "zfisher_session_1.json"
    sfile.write_text(json.dumps(data))

    session.load_session_file(sfile)
    p = session.get_data("processed_files")["R1 - FITC_puncta"]["path"]
    assert Path(p) == real_csv
    assert Path(p).exists()
