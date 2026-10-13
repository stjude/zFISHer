"""Mask saves compress only the tiles an edit touched.

The mask editor keeps the compressed tiles of each saved mask
(``core.io.LabelTiles``). Every in-place edit marks its region; napari's own
undo and redo, which fire no event, are found at the next save in napari's
history. These tests check that after every edit path the file on disk equals
the mask while only the touched tiles were compressed again, and that the flush
at a session change saves a mask whole once more, which repairs an edit that
went unmarked.

The mask editor module builds Qt widgets at import time, so these need a
QApplication and are skipped where napari is not installed.

Run (in an env with napari):  python -m pytest tests/test_mask_tiles.py -q
"""
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

MASK = "R1 - DAPI_masks"
TILES = 4 * 2 * 3           # 4 planes of 2 x 3 tiles of 256


def _volume():
    """Labels inside one tile, across tile edges, and on the volume border."""
    data = np.zeros((4, 300, 600), np.uint32)
    data[0:2, 20:60, 20:60] = 1             # inside tile (0, 0)
    data[1:3, 240:270, 250:262] = 2         # across four tiles
    data[2:4, 100:140, 560:600] = 3         # in the right-hand edge tile
    data[3, 299, 599] = 4
    data[0:4, 150:180, 300:330] = 5
    return data


@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(_qapp, monkeypatch):
    """A headless viewer holding a saved mask; the mask editor with deferred
    callbacks run at once; the session's registered files."""
    from napari.components import ViewerModel
    import zfisher.ui.widgets.mask_editor_widget as me
    from zfisher import constants
    from zfisher.core import session
    from zfisher.ui import viewer_helpers

    tmp = Path(tempfile.mkdtemp())
    registered = {}
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (str(tmp) if key == "output_dir" else default))
    monkeypatch.setattr(session, "set_processed_file",
                        lambda name, path, *a, **k: registered.__setitem__(name, path))
    viewer = ViewerModel()
    monkeypatch.setattr(me.napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(me, "QTimer", types.SimpleNamespace(singleShot=lambda _ms, fn: fn()))
    for cache in (me._mask_undo_stacks, me._label_boxes, me._label_tiles, me._dirty_masks):
        cache.clear()
    viewer_helpers.reset_label_ids_tracking()

    layer = viewer.add_labels(_volume(), name=MASK, scale=(0.3, 0.108, 0.108))
    layer.brush_size = 3
    me._on_mask_layer_changed(layer)
    real = me._mask_editor_widget
    monkeypatch.setattr(me, "_mask_editor_widget", types.SimpleNamespace(
        mask_layer=types.SimpleNamespace(value=layer), _function=real._function))
    path = tmp / constants.SEGMENTATION_DIR / f"{MASK}.ome.tif"
    me._write_mask_to_disk(layer)                # the first save compresses everything
    assert me._last_save == {"tiles": TILES, "all_tiles": True}
    yield types.SimpleNamespace(me=me, viewer=viewer, layer=layer, path=path, registered=registered)
    me._save_timer.stop()
    me._flush_timer.stop()
    for cache in (me._mask_undo_stacks, me._label_boxes, me._label_tiles, me._dirty_masks):
        cache.clear()
    viewer_helpers.reset_label_ids_tracking()


def _save(env):
    """Save as the debounced timer does; returns the number of tiles compressed."""
    env.me._save_pending_layer = env.layer
    env.me._do_save()
    from zfisher.core import io
    np.testing.assert_array_equal(io.read_label_tif(env.path), env.layer.data)
    return env.me._last_save["tiles"]


def test_file_is_ome_tiff_with_the_layer_scale_and_registered(env):
    import tifffile
    with tifffile.TiffFile(env.path) as tif:
        assert tif.is_ome and 'PhysicalSizeZ="0.3"' in tif.ome_metadata
    assert env.registered[MASK] == str(env.path)


def test_delete_compresses_only_the_label_tiles(env):
    env.me._delete_label_inplace(env.layer, 2)
    assert _save(env) == 8                      # 2 planes x 4 tiles
    env.me._delete_label_inplace(env.layer, 1)
    assert _save(env) == 2


def test_merge_compresses_only_the_source_tiles(env):
    env.me._mask_editor_widget._function(env.layer, 3, 5)
    assert _save(env) == 2


def test_editor_undo_compresses_only_the_undone_tiles(env):
    env.me._delete_label_inplace(env.layer, 2)
    _save(env)
    env.me._on_mask_undo()
    assert _save(env) == 8


def test_paint_stroke_is_marked_from_napari_event(env):
    env.layer.mode = "paint"
    env.layer.paint((1, 10, 10), 7)
    env.layer.mode = "pan_zoom"
    assert 0 < _save(env) <= 2


def test_napari_undo_and_redo_are_found_in_its_history(env):
    env.layer.n_edit_dimensions = 3
    env.layer.paint((2, 200, 400), 8)
    _save(env)
    env.layer.undo()                            # fires no event
    assert 0 < _save(env) < TILES
    env.layer.redo()
    assert 0 < _save(env) < TILES


def test_whole_volume_edits_compress_everything(env):
    env.me._extrude_spinbox.value = 2          # planes 1-2, so it grows into 0 and 3
    env.me._on_extrude()
    assert _save(env) == TILES
    env.me._on_mask_undo()                      # a whole-volume undo record
    assert _save(env) == TILES


def test_replaced_array_starts_new_tiles(env):
    env.layer.data = env.layer.data.copy()
    assert _save(env) == TILES


def test_flush_saves_a_tile_by_tile_mask_whole_again(env):
    env.me._delete_label_inplace(env.layer, 2)
    _save(env)
    env.layer.data[3, 299, 599] = 0             # an in-place edit no hook marked
    env.me.flush_pending_mask_saves()
    assert env.me._last_save == {"tiles": TILES, "all_tiles": True}
    from zfisher.core import io
    np.testing.assert_array_equal(io.read_label_tif(env.path), env.layer.data)
    env.me.flush_pending_mask_saves()           # nothing saved tile by tile since
    assert env.me._last_save == {"tiles": TILES, "all_tiles": True}


@pytest.mark.parametrize("seed", range(6))
def test_random_edits_keep_the_file_exact(env, seed):
    rng = np.random.default_rng(seed)
    env.layer.n_edit_dimensions = 3
    partial = 0
    for _ in range(25):
        labels = [int(v) for v in np.unique(env.layer.data) if v]
        op = rng.integers(6)
        if op == 0 and labels:
            env.me._delete_label_inplace(env.layer, int(rng.choice(labels)))
        elif op == 1 and len(labels) >= 2:
            src, dst = rng.choice(labels, 2, replace=False)
            env.me._mask_editor_widget._function(env.layer, int(src), int(dst))
        elif op == 2:
            env.me._on_mask_undo()
        elif op == 3:
            pos = (int(rng.integers(4)), int(rng.integers(300)), int(rng.integers(600)))
            env.layer.paint(pos, int(rng.integers(0, 12)))
        elif op == 4:
            env.layer.undo()
        else:
            env.layer.redo()
        partial += _save(env) < TILES
    assert partial >= 20


# --- The other mask writers ---------------------------------------------------

def test_gui_segmentation_writes_the_same_format(env):
    """GUI segmentation used to write its masks uncompressed (1.19 GB)."""
    import tifffile
    from zfisher.core import io
    from zfisher.ui import viewer_helpers
    source = env.viewer.add_image(np.zeros((4, 300, 600), np.uint16), name="R2 - DAPI",
                                  scale=(0.3, 0.108, 0.108))
    masks = _volume()
    viewer_helpers.add_segmentation_results_to_viewer(env.viewer, source, masks, np.empty((0, 3)))
    path = env.path.with_name("R2 - DAPI_masks.ome.tif")
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome and tif.pages[0].is_tiled and 'PhysicalSizeX="0.108"' in tif.ome_metadata
    np.testing.assert_array_equal(io.read_label_tif(path), masks)
    assert path.stat().st_size < masks.nbytes / 20
    assert env.registered["R2 - DAPI_masks"] == str(path)


def test_consensus_mask_writes_the_same_format(env, monkeypatch):
    import tifffile
    from zfisher import constants
    from zfisher.core import io, segmentation
    mask = _volume()
    merged, _ = segmentation.process_consensus_nuclei(
        mask, mask.copy(), output_dir=str(env.path.parent.parent), voxel_size=(0.3, 0.1, 0.1))
    path = env.path.with_name(f"{constants.CONSENSUS_MASKS_NAME}.ome.tif")
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome and 'PhysicalSizeX="0.1"' in tif.ome_metadata
    np.testing.assert_array_equal(io.read_label_tif(path), merged)
