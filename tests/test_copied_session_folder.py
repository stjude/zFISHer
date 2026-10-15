"""Regression tests: a copied output folder uses its own files.

Opening the session file of a copied output folder while the original folder
still existed kept every registered path (masks, puncta CSVs, aligned images,
the report) pointing into the original, because a stored path that existed was
kept before it was re-based onto the folder the session file now lives in. The
GUI then loaded the original's layers, and removing a layer renamed the
original's file to ``.bak``.

The core tests run anywhere; the removal and mask-save tests import napari/Qt
and are skipped where napari is not installed.

Run:  python -m pytest tests/test_copied_session_folder.py -q
"""
import json
import logging
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from zfisher import constants
from zfisher.core import io, session

MASK = "R1 - DAPI_masks"
PUNCTA = "R1 - FITC_puncta"


@pytest.fixture(autouse=True)
def _restore_log_levels():
    """The session log opens DEBUG on these loggers and leaves it on; restore
    them so later tests do not log at DEBUG (see synthetic_rounds.run_batch)."""
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    yield
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture
def folders(tmp_path):
    """An output folder with a mask, a puncta CSV and a raw input outside it,
    copied to ``copy/`` with the original left in place."""
    raw = tmp_path / "R1.tif"
    raw.write_bytes(b"raw")
    orig = tmp_path / "orig"
    session.clear_session()
    try:
        session.initialize_new_session(orig, raw, raw)
        mask = io.mask_path(orig / constants.SEGMENTATION_DIR, MASK)
        io.write_label_tif(mask, np.ones((2, 16, 16), np.uint32), voxel_size=(1.0, 0.5, 0.5))
        csv = constants.puncta_csv_path(orig / constants.REPORTS_DIR, PUNCTA)
        csv.write_text("Z,Y,X,Nucleus_ID,Intensity,SNR,puncta_id,Source\n1,2,3,1,10,5,1,auto\n")
        session.set_processed_file(MASK, str(mask), "labels", {"subtype": "mask"})
        session.set_processed_file(PUNCTA, str(csv), "points", {"subtype": "puncta_csv"})
        session.set_processed_file("computed", "", "points", {"subtype": "computed_ids"})
    finally:
        session.clear_session()
    copy = tmp_path / "copy"
    shutil.copytree(orig, copy)
    yield orig, copy, raw
    session.clear_session()


def _load(folder):
    session.load_session_file(folder / constants.SESSIONS_DIR / constants.SESSION_FILENAME)


def _snapshot(folder):
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*")
            if p.is_file() and constants.LOGS_DIR not in p.relative_to(folder).parts}


def test_copy_paths_point_into_the_copy(folders):
    orig, copy, raw = folders
    _load(copy)
    assert Path(session.get_data("output_dir")) == copy
    processed = session.get_data("processed_files")
    for name in (MASK, PUNCTA):
        path = Path(processed[name]["path"])
        assert path.is_relative_to(copy), f"{name} still points at {path}"
        assert path.exists()
    assert processed["computed"]["path"] == ""
    # A raw input outside the output folder is left where it is.
    assert Path(session.get_data("r1_path")) == raw
    # The copy's new session file holds the copy's paths too.
    saved = (copy / constants.SESSIONS_DIR / "zfisher_session_2.json").read_text()
    assert str(orig / constants.SEGMENTATION_DIR) not in saved


def test_original_still_loads_its_own_files(folders):
    orig, copy, raw = folders
    _load(orig)
    processed = session.get_data("processed_files")
    for name in (MASK, PUNCTA):
        assert Path(processed[name]["path"]).is_relative_to(orig)


def test_file_missing_from_the_copy_is_not_read_from_the_original(folders):
    orig, copy, raw = folders
    constants.puncta_csv_path(copy / constants.REPORTS_DIR, PUNCTA).unlink()
    _load(copy)
    path = Path(session.get_data("processed_files")[PUNCTA]["path"])
    assert path.is_relative_to(copy) and not path.exists()


def test_raw_input_inside_the_folder_falls_back_to_the_original(folders):
    orig, copy, raw = folders
    inside = orig / constants.INPUT_DIR / "R2.tif"
    inside.write_bytes(b"raw2")
    sess = copy / constants.SESSIONS_DIR / constants.SESSION_FILENAME
    data = json.loads(sess.read_text())
    data["r2_path"] = str(inside)
    sess.write_text(json.dumps(data))
    _load(copy)
    # Missing from the copy, so read from where it was ...
    assert Path(session.get_data("r2_path")) == inside
    session.clear_session()
    # ... and the copy's own file once it has one.
    (copy / constants.INPUT_DIR / "R2.tif").write_bytes(b"raw2")
    _load(copy)
    assert Path(session.get_data("r2_path")) == copy / constants.INPUT_DIR / "R2.tif"


def test_moved_folder_still_reanchors(folders):
    orig, copy, raw = folders
    shutil.rmtree(orig)
    _load(copy)
    for name in (MASK, PUNCTA):
        path = Path(session.get_data("processed_files")[name]["path"])
        assert path.is_relative_to(copy) and path.exists()


@pytest.fixture
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_removing_a_layer_backs_up_only_the_copy(folders, _qapp):
    from zfisher.ui import events
    orig, copy, raw = folders
    before = _snapshot(orig)
    _load(copy)
    events._remove_layer_and_file(PUNCTA)
    copy_csv = constants.puncta_csv_path(copy / constants.REPORTS_DIR, PUNCTA)
    assert not copy_csv.exists() and copy_csv.with_name(copy_csv.name + ".bak").exists()
    assert _snapshot(orig) == before


def test_mask_edit_writes_only_the_copy(folders, _qapp, monkeypatch):
    from napari.layers import Labels
    import zfisher.ui.widgets.mask_editor_widget as me
    orig, copy, raw = folders
    monkeypatch.setattr(me.napari, "current_viewer", lambda: None)
    before = _snapshot(orig)
    _load(copy)
    path = Path(session.get_data("processed_files")[MASK]["path"])
    layer = Labels(np.asarray(io.read_label_tif(path)), name=MASK, scale=(1.0, 0.5, 0.5))
    layer.data[0, :4, :4] = 7
    written = me._write_mask_to_disk(layer)
    try:
        assert written is not None and Path(written).is_relative_to(copy)
        assert Path(session.get_data("processed_files")[MASK]["path"]).is_relative_to(copy)
        assert _snapshot(orig) == before
    finally:
        me._label_tiles.pop(id(layer), None)
        me._dirty_masks.clear()
