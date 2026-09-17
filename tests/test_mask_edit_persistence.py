"""Regression tests: brush/erase/fill edits on a mask reach disk and cascade.

napari fires ``layer.events.data`` only when the whole array is assigned. Brush,
eraser and fill strokes commit through the layer's undo history and fire
``layer.events.paint``. The mask editor's autosave used to listen to ``data``
only, so painted edits were never written to disk unless a merge/delete/extrude
followed. The editor now marks the layer dirty on either event and flushes
(save, refresh IDs, resync puncta) on an idle timer or on tool release.

These import the mask editor module, which builds Qt widgets at import time, so
they need a QApplication and are skipped where napari is not installed.

Run (in an env with napari):  python -m pytest tests/test_mask_edit_persistence.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def _qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def me(_qapp, monkeypatch):
    """The mask_editor_widget module with a temp output dir and the resync spied."""
    import zfisher.ui.widgets.mask_editor_widget as me_mod
    from zfisher.core import session
    from zfisher.ui import viewer_helpers
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (tmp if key == "output_dir" else default))
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(viewer_helpers, "resync_puncta_nucleus_ids",
                        lambda *a, **k: (calls.append(a), {"updated_layers": 0, "removed_total": 0, "changed_total": 0})[1])
    monkeypatch.setattr(me_mod.napari, "current_viewer", lambda: None)
    me_mod._dirty_masks.clear()
    me_mod._tmp = Path(tmp)
    me_mod._resync_calls = calls
    return me_mod


def _labels(name):
    from napari.layers import Labels
    return Labels(np.zeros((1, 32, 32), np.int32), name=name)


def _consensus_name():
    from zfisher import constants
    return constants.CONSENSUS_MASKS_NAME


def _tif(me, layer):
    from zfisher import constants
    return me._tmp / constants.SEGMENTATION_DIR / f"{layer.name}.tif"


def test_paint_stroke_marks_dirty_and_flush_writes_tif(me):
    import tifffile
    layer = _labels(_consensus_name())
    me._on_mask_layer_changed(layer)
    assert me._mask_editor_widget._current_callbacks[1] in layer.events.paint.callbacks
    assert me._mask_editor_widget._current_callbacks[0] in layer.events.data.callbacks

    layer.mode = "paint"
    layer.selected_label = 5
    layer.paint((0, 10, 10), 5)              # commits one history atom -> events.paint
    assert id(layer) in me._dirty_masks
    assert not _tif(me, layer).exists()      # nothing written yet (debounced)

    me._flush_mask_edits(layer, refresh_ids=False)
    assert id(layer) not in me._dirty_masks
    saved = tifffile.imread(_tif(me, layer))
    assert saved[0, 10, 10] == 5
    me._on_mask_layer_changed(None)


def test_flush_cascades_into_puncta_for_consensus_only(me):
    consensus = _labels(_consensus_name())
    per_round = _labels("R1 - DAPI_masks")

    class _V:
        layers = []
    me.napari.current_viewer = lambda: _V()

    me._mark_mask_dirty(consensus)
    me._flush_mask_edits(consensus, refresh_ids=False)
    assert len(me._resync_calls) == 1

    me._mark_mask_dirty(per_round)
    me._flush_mask_edits(per_round, refresh_ids=False)
    assert len(me._resync_calls) == 1            # per-round masks do not cascade
    assert _tif(me, per_round).exists()          # but they are saved


def test_fill_and_whole_assignment_both_mark_dirty(me):
    layer = _labels(_consensus_name())
    me._on_mask_layer_changed(layer)
    layer.fill((0, 3, 3), 2)                      # events.paint
    assert id(layer) in me._dirty_masks
    me._dirty_masks.clear()
    layer.data = layer.data.copy()                # events.data
    assert id(layer) in me._dirty_masks
    me._on_mask_layer_changed(None)


def test_switching_layer_flushes_previous_and_disconnects(me):
    a = _labels(_consensus_name())
    b = _labels("R2 - DAPI_masks")
    me._on_mask_layer_changed(a)
    a.paint((0, 4, 4), 3)
    cbs = me._mask_editor_widget._current_callbacks
    me._on_mask_layer_changed(b)
    assert _tif(me, a).exists()                   # flushed before switching
    assert cbs[0] not in a.events.data.callbacks
    assert cbs[1] not in a.events.paint.callbacks
    me._on_mask_layer_changed(None)


def test_flush_defers_id_refresh_while_tool_active(me, monkeypatch):
    from zfisher.ui import viewer_helpers
    refreshed = []
    monkeypatch.setattr(viewer_helpers, "add_or_update_label_ids", lambda *a, **k: refreshed.append(a))
    layer = _labels(_consensus_name())

    class _V:
        layers = {f"{layer.name}_IDs": object()}
    me.napari.current_viewer = lambda: _V()

    layer.mode = "paint"
    me._mark_mask_dirty(layer)
    me._flush_mask_edits(layer)                   # refresh_ids=None -> deferred in paint mode
    assert refreshed == []
    layer.mode = "pan_zoom"
    me._mark_mask_dirty(layer)
    me._flush_mask_edits(layer)
    assert len(refreshed) == 1
