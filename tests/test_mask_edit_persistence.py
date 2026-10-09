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
    yield me_mod
    # A layer left dirty would flush from a real timer during a later test.
    me_mod._flush_timer.stop()
    me_mod._save_timer.stop()
    me_mod._dirty_masks.clear()


def _labels(name):
    from napari.layers import Labels
    return Labels(np.zeros((1, 32, 32), np.int32), name=name)


def _consensus_name():
    from zfisher import constants
    return constants.CONSENSUS_MASKS_NAME


def zio_read(path):
    from zfisher.core import io as zio
    return zio.read_label_tif(path)


def _tif(me, layer):
    from zfisher import constants
    return me._tmp / constants.SEGMENTATION_DIR / f"{layer.name}.ome.tif"


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
    saved = zio_read(_tif(me, layer))
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


# --- The save itself: compressed, swapped into place, and never failing silently ---

def test_saved_mask_is_compressed_and_identical(me):
    import tifffile
    layer = _labels(_consensus_name())
    layer.data[0, 2:20, 2:20] = 7
    me._mark_mask_dirty(layer)
    me._flush_mask_edits(layer, refresh_ids=False)
    path = _tif(me, layer)
    with tifffile.TiffFile(path) as tif:
        assert int(tif.pages[0].compression) != 1          # 1 means uncompressed
    back = zio_read(path)
    assert back.dtype == layer.data.dtype and np.array_equal(back, layer.data)
    assert not path.with_name(path.name + ".tmp").exists()


def test_failed_save_keeps_layer_unsaved_and_tells_the_user(me, monkeypatch):
    from zfisher.core import io as zio
    layer = _labels(_consensus_name())

    class _V:
        layers = []
        status = ""
    viewer = _V()
    me.napari.current_viewer = lambda: viewer
    working_writer = zio.write_label_tif

    def _disk_full(path, data, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(zio, "write_label_tif", _disk_full)

    me._mark_mask_dirty(layer)
    me._flush_mask_edits(layer, refresh_ids=False)         # must not raise
    assert id(layer) in me._dirty_masks                    # still unsaved, so the next flush retries
    assert "Could not save mask" in viewer.status and "disk full" in viewer.status
    assert not _tif(me, layer).exists()
    assert len(me._resync_calls) == 1                      # puncta still follow the mask on screen

    monkeypatch.setattr(zio, "write_label_tif", working_writer)
    me._flush_mask_edits(layer, refresh_ids=False)         # the retry
    assert id(layer) not in me._dirty_masks
    assert _tif(me, layer).exists()


def test_debounced_save_failure_is_not_silent_either(me, monkeypatch):
    from zfisher.core import io as zio
    layer = _labels("R1 - DAPI_masks")

    def _disk_full(path, data, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(zio, "write_label_tif", _disk_full)

    me._save_pending_layer = layer
    me._do_save()                                          # the 500 ms timer's slot; must not raise
    assert id(layer) in me._dirty_masks
    assert me._save_pending_layer is None


# --- Pending saves when the session changes or the app quits ----------------
#
# Saves are debounced, so the last edits sit in memory for up to a few seconds.
# Without a flush they were lost at quit, and after a session switch the timer
# wrote the old session's mask into the new session's folder.

@pytest.fixture
def out_dir(me, monkeypatch):
    """The session output folder as a mutable holder, to switch sessions."""
    from zfisher.core import session
    holder = {"dir": str(me._tmp)}
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (holder["dir"] if key == "output_dir" else default))
    return holder


def test_flush_writes_a_pending_delete_save_now(me, out_dir):
    import tifffile
    layer = _labels("R1 - DAPI_masks")
    layer.data[0, 2:9, 2:9] = 4
    me._schedule_save(layer)
    assert not _tif(me, layer).exists()                    # debounced
    me.flush_pending_mask_saves()
    assert np.array_equal(zio_read(_tif(me, layer)), layer.data)
    assert me._save_pending_layer is None and not me._save_timer.isActive()


def test_flush_runs_the_cascade_for_dirty_strokes(me, out_dir):
    class _V:
        layers = []
    me.napari.current_viewer = lambda: _V()
    consensus = _labels(_consensus_name())
    me._mark_mask_dirty(consensus, delay_ms=3000)
    me.flush_pending_mask_saves()
    assert _tif(me, consensus).exists()
    assert len(me._resync_calls) == 1                      # puncta Nucleus_ID follow the mask
    assert me._dirty_masks == {} and not me._flush_timer.isActive()


def test_layer_pending_and_dirty_is_written_once(me, out_dir, monkeypatch):
    from zfisher.core import io as zio
    writes = []
    real = zio.write_label_tif
    monkeypatch.setattr(zio, "write_label_tif", lambda path, data, **k: (writes.append(path), real(path, data, **k))[1])
    layer = _labels(_consensus_name())
    me._schedule_save(layer)
    me._mark_mask_dirty(layer)
    me.flush_pending_mask_saves()
    assert len(writes) == 1


def test_nothing_reaches_the_next_session_folder(me, out_dir):
    a = _labels("R1 - DAPI_masks")
    b = _labels(_consensus_name())
    me._schedule_save(a)
    me._mark_mask_dirty(b)
    me.flush_pending_mask_saves()
    old = me._tmp
    new = Path(tempfile.mkdtemp())
    out_dir["dir"] = str(new)                              # the next session is now active
    me._do_save()                                          # what the timers would have run
    me._flush_mask_edits()
    from zfisher import constants
    assert not (new / constants.SEGMENTATION_DIR).exists()
    assert (old / constants.SEGMENTATION_DIR / f"{a.name}.ome.tif").exists()
    assert (old / constants.SEGMENTATION_DIR / f"{b.name}.ome.tif").exists()


def test_reset_saves_pending_edits_before_clearing(me, out_dir):
    layer = _labels("R1 - DAPI_masks")
    me._schedule_save(layer)
    me.reset_mask_editor_state()
    assert _tif(me, layer).exists()
    assert me._save_pending_layer is None and me._dirty_masks == {}


def test_failed_save_at_session_change_is_logged_and_dropped(me, out_dir, monkeypatch, caplog):
    from zfisher.core import io as zio

    def _disk_full(path, data, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(zio, "write_label_tif", _disk_full)
    layer = _labels(_consensus_name())
    me._mark_mask_dirty(layer)
    with caplog.at_level("ERROR"):
        me.flush_pending_mask_saves()
    assert me._dirty_masks == {}                           # nothing left to land in the next session
    assert any("could not be saved before the session changed" in r.getMessage() for r in caplog.records)
