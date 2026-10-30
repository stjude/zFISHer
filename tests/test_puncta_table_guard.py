"""A puncta table whose rows do not match its points is never written.

The puncta CSV is built by joining a layer's feature table to its points by
position. With one row too few or too many, every value after it, Nucleus_ID
included, landed on the wrong punctum, and nothing said so; when a write
failed, the layer was saved with its coordinates only, dropping every
punctum's Nucleus_ID, puncta_id and Source. Both are refused now, and the file
keeps its last good version. Puncta ids carried by more than one punctum are
logged; older sessions can hold them, so they do not stop a save.

Run:  python -m pytest tests/test_puncta_table_guard.py -q
"""
import logging
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import puncta, session

NAME = "Aligned R1 - FITC_puncta"


def _features(n, ids=None):
    return pd.DataFrame({"puncta_id": ids if ids is not None else range(n), "Nucleus_ID": [7] * n,
                         "Intensity": [1.0] * n, "SNR": [2.0] * n, "Source": ["auto"] * n})


def test_a_table_with_a_row_too_few_is_refused_and_the_file_kept(tmp_path):
    csv = tmp_path / "x.csv"
    puncta.write_puncta_csv(csv, np.ones((3, 3)), _features(3))
    before = csv.read_bytes()
    with pytest.raises(puncta.PunctaTableError, match="3 points but 2 feature rows"):
        puncta.write_puncta_csv(csv, np.ones((3, 3)), _features(2))
    assert csv.read_bytes() == before


def test_repeated_ids_are_found():
    assert puncta.repeated_puncta_ids(_features(4, ids=[1, 2, 2, 5])) == [2]
    assert puncta.repeated_puncta_ids(_features(3)) == []
    assert puncta.repeated_puncta_ids(None) == []


@pytest.fixture
def gui(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from zfisher.ui import events
    viewer = types.SimpleNamespace(status="")
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    data = {"output_dir": str(tmp_path), "processed_files": {}}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "set_processed_file", lambda name, path, **k: data["processed_files"].update({name: path}))
    csv = constants.puncta_csv_path(tmp_path / constants.REPORTS_DIR, NAME)
    yield types.SimpleNamespace(events=events, viewer=viewer, csv=csv, napari=napari)


def test_a_layer_whose_table_does_not_match_is_not_saved(gui):
    good = types.SimpleNamespace(name=NAME, data=np.ones((3, 3)), features=_features(3))
    assert gui.events.save_puncta_layer(good)
    before = gui.csv.read_bytes()
    bad = types.SimpleNamespace(name=NAME, data=np.ones((4, 3)), features=_features(3))

    assert gui.events.save_puncta_layer(bad) is False

    assert gui.csv.read_bytes() == before                         # not rewritten, not cut to coordinates
    assert "were not saved" in gui.viewer.status and "4 points but 3 feature rows" in gui.viewer.status


def test_a_data_event_save_tries_once_more_before_reporting(gui, monkeypatch):
    layer = gui.napari.layers.Points(np.ones((2, 3)), name=NAME, features=_features(2))
    calls, later = [], []
    real_write = puncta.write_puncta_csv

    def write_once_incomplete(path, coords, features=None):
        calls.append(len(coords))
        if len(calls) == 1:
            raise puncta.PunctaTableError("2 points but 1 feature rows")
        real_write(path, coords, features)
    monkeypatch.setattr(gui.events._puncta, "write_puncta_csv", write_once_incomplete)
    monkeypatch.setattr(gui.events, "QTimer", types.SimpleNamespace(singleShot=lambda ms, fn: later.append(fn)))
    gui.events.attach_puncta_listener(layer, NAME)
    try:
        layer.events.data(value=layer.data)
        assert not gui.csv.exists() and gui.viewer.status == "" and len(later) == 1
        later[0]()                                                # after the handlers that finish the table
        assert gui.csv.exists() and len(pd.read_csv(gui.csv)) == 2 and gui.viewer.status == ""
    finally:
        gui.events.detach_puncta_listener(layer)


def test_repeated_ids_are_logged_but_saved(gui, caplog):
    layer = types.SimpleNamespace(name=NAME, data=np.ones((3, 3)), features=_features(3, ids=[4, 4, 9]))
    with caplog.at_level(logging.WARNING, logger="zfisher.ui.events"):
        logging.getLogger("zfisher").propagate = True
        try:
            assert gui.events.save_puncta_layer(layer)
        finally:
            logging.getLogger("zfisher").propagate = False
    assert list(pd.read_csv(gui.csv)["puncta_id"]) == [4, 4, 9]
    assert "carried by more than one punctum" in caplog.text
