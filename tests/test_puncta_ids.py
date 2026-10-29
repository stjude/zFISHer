"""A puncta_id never names two different puncta, and one channel has one set
of detection settings.

Replace used to number its new puncta from 0, or from the highest manual
punctum it kept, so a re-detection handed the ids of the puncta it removed to
different puncta. New ids now come from ``puncta.allocate_puncta_ids``, which
keeps the next free id per channel in the session. GUI detection used to store
its settings under the image layer's name (``'Warped R2 - FITC'``) while batch
used ``'R2 - FITC'``, so a re-detection left the old row in the report's
Parameters sheet; both now use the channel.

Run:  python -m pytest tests/test_puncta_ids.py -q
"""
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import puncta, session


@pytest.fixture
def data(monkeypatch):
    """An in-memory session."""
    store = {"output_dir": tempfile.mkdtemp(), "processed_files": {}, "puncta_params": {}}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: store.get(key, default) if key else store)
    monkeypatch.setattr(session, "update_data", lambda key, value: store.__setitem__(key, value))
    return store


# --- core -------------------------------------------------------------------

@pytest.mark.parametrize("name, key", [
    ("R2 - FITC", "R2 - FITC"),
    ("R2 - FITC_puncta", "R2 - FITC"),
    ("Warped R2 - FITC_puncta", "R2 - FITC"),
    ("Aligned R1 - Cy5 - 2", "R1 - Cy5 - 2"),
    ("Points", None),
    ("FR2 - x", None),
])
def test_channel_key(name, key):
    assert puncta.channel_key(name) == key


def test_ids_rise_per_channel_and_are_kept_in_the_session(data):
    assert list(puncta.allocate_puncta_ids("R1 - FITC_puncta", 3)) == [0, 1, 2]
    assert list(puncta.allocate_puncta_ids("Aligned R1 - FITC_puncta", 2)) == [3, 4]   # same channel
    assert list(puncta.allocate_puncta_ids("R2 - FITC_puncta", 1)) == [0]              # another one
    assert data["puncta_next_id"] == {"R1 - FITC": 5, "R2 - FITC": 1}
    # Existing ids above the counter win; the counter never goes back.
    assert list(puncta.allocate_puncta_ids("R1 - FITC_puncta", 1, existing_ids=[9, np.nan])) == [10]
    assert list(puncta.allocate_puncta_ids("R1 - FITC_puncta", 1, existing_ids=[2])) == [11]


def test_a_session_without_a_counter_starts_above_the_channel_csvs(data):
    """Batch and older sessions keep no counter: the raw CSV can hold ids the
    aligned layer lost to the nucleus filter, and they must not come back."""
    reports = Path(data["output_dir"])
    for name, ids in [("R2 - FITC_puncta", [0, 1, 7]), ("Warped R2 - FITC_puncta", [0, 1]),
                      ("R1 - FITC_puncta", [50])]:
        path = reports / f"{name}.csv"
        pd.DataFrame({"puncta_id": ids, "Z": 0, "Y": 0, "X": 0}).to_csv(path, index=False)
        data["processed_files"][name] = {"path": str(path), "type": "points", "subtype": "puncta_csv"}
    assert list(puncta.allocate_puncta_ids("Warped R2 - FITC_puncta", 2, existing_ids=[0, 1])) == [8, 9]


def test_a_layer_without_a_channel_is_numbered_from_its_own_ids(data):
    assert list(puncta.allocate_puncta_ids("Points", 2, existing_ids=[4])) == [5, 6]
    assert "puncta_next_id" not in data


def test_one_channel_keeps_one_set_of_settings(data):
    data["puncta_params"] = {"R2 - FITC": {"sensitivity": 0.1}, "Warped R2 - FITC": {"sensitivity": 0.2},
                             "R1 - FITC": {"sensitivity": 0.3}}
    # Older sessions: the entry stored last describes the current puncta.
    assert puncta.detection_settings() == {"R2 - FITC": {"sensitivity": 0.2}, "R1 - FITC": {"sensitivity": 0.3}}
    puncta.record_detection_settings("Warped R2 - FITC", {"sensitivity": 0.4, "detected_in": "aligned"})
    assert data["puncta_params"] == {"R1 - FITC": {"sensitivity": 0.3},
                                     "R2 - FITC": {"sensitivity": 0.4, "detected_in": "aligned"}}


# --- GUI: detection's Replace and Merge, and the editor ----------------------

@pytest.fixture
def gui(data, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from napari.components import ViewerModel
    from zfisher.ui import viewer_helpers
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    source = types.SimpleNamespace(name="R1 - FITC", scale=(1, 1, 1), translate=(0, 0, 0))
    return types.SimpleNamespace(viewer=ViewerModel(), helpers=viewer_helpers, source=source,
                                 name=f"R1 - FITC{constants.PUNCTA_SUFFIX}")


def _rows(n, offset=0):
    coords = np.column_stack([np.ones(n), np.arange(n) * 10 + offset, np.arange(n) * 10 + offset])
    return np.column_stack([coords, np.ones(n), np.full(n, 50.0), np.full(n, 4.0), np.arange(n)]).astype(float)


def test_replace_does_not_reuse_the_ids_it_removes(gui):
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(3))          # ids 0-2
    layer = gui.viewer.layers[gui.name]
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(2, offset=5), existing_action="replace")
    assert list(layer.features["puncta_id"]) == [3, 4]
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(2, offset=7), existing_action="replace_all")
    assert list(layer.features["puncta_id"]) == [5, 6]
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(1, offset=200), existing_action="merge")
    assert list(layer.features["puncta_id"]) == [5, 6, 7]


def test_replace_numbers_above_puncta_deleted_before_a_reload(gui, data):
    """The counter is in the session, so ids deleted in an earlier run of the
    app are not handed out after the session is opened again."""
    data["puncta_next_id"] = {"R1 - FITC": 40}
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(3))          # loaded ids 0-2
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(2, offset=5), existing_action="replace")
    assert list(gui.viewer.layers[gui.name].features["puncta_id"]) == [40, 41]


def test_points_added_by_hand_take_ids_from_the_same_counter(gui, data, monkeypatch):
    from zfisher.ui.widgets import puncta_editor_widget as pe
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(3))
    layer = gui.viewer.layers[gui.name]
    data["puncta_next_id"] = {"R1 - FITC": 20}
    assert pe._next_puncta_ids(layer, 2) == [20, 21]
    assert pe._next_puncta_ids(layer, 1) == [22]
    gui.helpers.add_or_update_puncta_layer(gui.viewer, gui.source, _rows(1), existing_action="replace_all")
    assert list(layer.features["puncta_id"]) == [23]
