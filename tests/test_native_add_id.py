"""Regression tests for native-add puncta-ID assignment in the Puncta Editor.

When a point is placed with napari's built-in Add tool, napari grows the layer's
feature table by broadcasting the *last* row's values into every new row before
emitting ``events.data`` — so each new point arrives carrying a duplicate
``puncta_id``. ``_on_points_data_changed`` must overwrite that with a fresh
monotonic id (and recompute Source/Nucleus_ID/Intensity/SNR), while leaving
genuinely-unique ids (set by the fishing hook) and undo-restored rows alone.

These exercise the real napari Points layer + handler, so they need napari/Qt.
They are skipped automatically where napari is not installed (e.g. the core CI
environment).

Run (in an env with napari):  python -m pytest tests/test_native_add_id.py -q
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def _qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def pe(_qapp, monkeypatch):
    """The puncta_editor_widget module, with side-effecting saves stubbed and the
    magicgui layer selector replaced by a settable shim."""
    import types
    from zfisher.ui import events as zevents
    monkeypatch.setattr(zevents, "save_puncta_layer", lambda *a, **k: None)
    import zfisher.ui.widgets.puncta_editor_widget as pe_mod
    shim = types.SimpleNamespace(points_layer=types.SimpleNamespace(value=None))
    monkeypatch.setattr(pe_mod, "_puncta_editor_widget", shim)
    pe_mod._puncta_undo.clear()
    pe_mod._skip_data_sync = False
    return pe_mod


def _layer(ids, sources=None):
    import napari  # noqa: F401  (import guarded by importorskip)
    from napari.layers import Points
    ids = list(ids)
    sources = sources or ["auto"] * len(ids)
    data = np.array([[0, 10 + i, 10 + i] for i in range(len(ids))], float)
    feats = pd.DataFrame({
        "puncta_id": ids, "Nucleus_ID": [1] * len(ids),
        "Intensity": [5.0] * len(ids), "SNR": [2.0] * len(ids), "Source": sources,
    })
    return Points(data, features=feats, name="R1 - FITC_puncta",
                  text={"string": "{puncta_id:.0f}"})


def _attach(pe, layer):
    """Replicate the layer-attach wiring (_on_layer_change) for a standalone layer."""
    pe._puncta_editor_widget.points_layer.value = layer
    pe._prev_point_count[0] = len(layer.data)
    pe._prev_snapshot[0] = (layer.data.copy(), layer.features.copy())
    pe._seed_next_puncta_id(layer)
    layer.events.data.connect(pe._on_points_data_changed)


def test_native_add_assigns_fresh_unique_id(pe):
    """The reported bug: native Add must not leave the broadcast duplicate id."""
    layer = _layer([10, 11, 12])
    _attach(pe, layer)
    layer.add(np.array([0.0, 40.0, 40.0]))
    ids = list(layer.features["puncta_id"])
    assert ids == [10, 11, 12, 13]
    assert len(set(ids)) == len(ids)
    assert list(layer.features["Source"])[-1] == "manual"


def test_multiple_native_adds_stay_unique(pe):
    layer = _layer([10, 11, 12])
    _attach(pe, layer)
    for k in range(3):
        layer.add(np.array([0.0, 40.0 + k, 40.0 + k]))
    ids = list(layer.features["puncta_id"])
    assert ids == [10, 11, 12, 13, 14, 15]


def test_duplicate_default_after_delete_is_reassigned(pe):
    """A post-delete layer whose feature-default duplicates a surviving id."""
    layer = _layer([10, 12])
    _attach(pe, layer)
    layer.add(np.array([0.0, 80.0, 80.0]))
    ids = list(layer.features["puncta_id"])
    assert len(set(ids)) == len(ids)
    assert ids == [10, 12, 13]


def test_skip_guarded_add_is_not_renumbered(pe):
    """A point added under _skip_data_sync with the count synced (the fishing-hook
    pattern) is left exactly as-is — the handler must not touch it."""
    layer = _layer([10, 11, 12])
    _attach(pe, layer)
    orig_feats = layer.features.copy()
    pe._skip_data_sync = True
    new_id = pe._next_puncta_ids(layer, 1)[0]  # allocate from the counter (13)
    layer.data = np.vstack([layer.data, [0.0, 70.0, 70.0]])
    layer.features = pd.concat(
        [orig_feats, pd.DataFrame([{"puncta_id": new_id, "Nucleus_ID": 5,
                                    "Intensity": 1.0, "SNR": 1.0, "Source": "manual"}])],
        ignore_index=True)
    pe._prev_point_count[0] = len(layer.data)  # fishing hook syncs the count
    pe._skip_data_sync = False
    pe._on_points_data_changed()  # a stray event must be a no-op
    assert list(layer.features["puncta_id"]) == [10, 11, 12, 13]


def test_delete_highest_then_add_does_not_reuse_id(pe):
    """After deleting the highest-numbered point, a native add must advance the
    counter, never reuse the freed id (napari broadcasts the deleted id)."""
    layer = _layer([10, 11, 12])
    _attach(pe, layer)
    layer.selected_data = {2}
    layer.remove_selected()              # delete id 12 (the highest)
    layer.add(np.array([0.0, 40.0, 40.0]))
    ids = list(layer.features["puncta_id"])
    assert 12 not in ids
    assert ids == [10, 11, 13]


def test_counter_continues_past_fishing_hook_id(pe):
    layer = _layer([10, 11, 12, 99], sources=["auto", "auto", "auto", "manual"])
    _attach(pe, layer)
    layer.add(np.array([0.0, 90.0, 90.0]))
    assert list(layer.features["puncta_id"]) == [10, 11, 12, 99, 100]


def test_undo_does_not_renumber_restored_points(pe, monkeypatch):
    layer = _layer([10, 11, 12])
    _attach(pe, layer)
    pe._puncta_undo.push(layer)               # snapshot of all three points
    pe._skip_data_sync = True                  # emulate a delete down to two
    layer.data = layer.data[:2]
    layer.features = layer.features.iloc[:2].reset_index(drop=True)
    pe._skip_data_sync = False
    pe._prev_point_count[0] = 2

    class _FakeViewer:
        def __init__(self, lyr):
            self.layers = {"R1 - FITC_puncta": lyr}
            self.status = ""

    monkeypatch.setattr(pe.napari, "current_viewer", lambda: _FakeViewer(layer))
    pe._on_puncta_undo()
    assert list(layer.features["puncta_id"]) == [10, 11, 12]
