"""Incremental nucleus-ID overlay updates (gap analysis §13.9, fix 2).

After every mask edit the ID overlay used to be rebuilt from regionprops over
the whole mask (about 1 s on a typical volume). It now re-measures only the
labels the edit touched (``viewer_helpers.refresh_label_ids``). These tests
check that the overlay stays exactly equal to a full rebuild through every edit
path, including napari's own undo and redo, and that the full rebuild is not
what keeps it equal.

The overlay is a real napari Points layer in a headless ``ViewerModel``; no
display is needed. The mask editor module builds Qt widgets at import time, so
these need a QApplication and are skipped where napari is not installed.

Run (in an env with napari):  python -m pytest tests/test_label_ids.py -q
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from skimage.measure import regionprops

from zfisher.core import label_edits

MASK = "R1 - DAPI_masks"


def _volume():
    """Labels on every border, one label in two pieces, touching labels."""
    data = np.zeros((6, 40, 48), np.uint32)
    data[0:3, 0:6, 0:6] = 1
    data[3:6, 34:40, 42:48] = 2
    data[1:4, 5:9, 20:26] = 3
    data[2:5, 30:35, 10:14] = 3
    data[0:6, 15:20, 0:4] = 4
    data[2:4, 12:16, 30:40] = 5
    data[2:4, 16:20, 30:40] = 6
    data[5, 0, 47] = 7
    return data


def _full_table(data):
    props = regionprops(data)
    coords = np.array([p.centroid for p in props], float).reshape(-1, data.ndim)
    return coords, np.array([p.label for p in props], np.int64)


# --- core -------------------------------------------------------------------

def test_label_centroids_equal_regionprops_exactly():
    data = _volume()
    boxes = label_edits.LabelBoxes(data)
    # Boxes larger than the label must not change the result.
    boxes.extend(3, (slice(0, 6), slice(0, 40), slice(0, 48)))
    boxes.extend(5, (slice(1, 5), slice(10, 25), slice(28, 44)))
    got = label_edits.label_centroids(data, boxes, range(1, 8))
    for p in regionprops(data):
        assert got[p.label] == p.centroid       # exact, not approximate
    assert label_edits.label_centroids(data, boxes, [0, 9]) == {9: None}


def test_update_centroid_table_replaces_removes_adds_and_sorts():
    coords = np.array([[0, 0, 1.0], [0, 0, 2.0], [0, 0, 3.0]])
    labels = np.array([1, 2, 3])
    changed = {2: None, 3: (9.0, 9.0, 9.0), 0: None, 10: (5.0, 5.0, 5.0), 4: (4.0, 4.0, 4.0)}
    out_coords, out_labels = label_edits.update_centroid_table(coords, labels, changed, 3)
    np.testing.assert_array_equal(out_labels, [1, 3, 4, 10])
    np.testing.assert_array_equal(out_coords, [[0, 0, 1], [9, 9, 9], [4, 4, 4], [5, 5, 5]])
    empty_coords, empty_labels = label_edits.update_centroid_table(
        np.empty((0, 3)), np.empty(0), {7: (1.0, 2.0, 3.0)}, 3)
    np.testing.assert_array_equal(empty_labels, [7])
    assert empty_coords.shape == (1, 3)


# --- mask editor and overlay -------------------------------------------------

@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(_qapp, monkeypatch):
    """A headless viewer holding a mask, its full ID overlay and its centroids
    layer; the mask editor with deferred callbacks run at once, and a count of
    full regionprops passes."""
    from napari.components import ViewerModel
    import zfisher.ui.widgets.mask_editor_widget as me
    from zfisher import constants
    from zfisher.core import session, segmentation
    from zfisher.ui import viewer_helpers

    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (tmp if key == "output_dir" else default))
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    viewer = ViewerModel()
    monkeypatch.setattr(me.napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(me, "QTimer", types.SimpleNamespace(singleShot=lambda _ms, fn: fn()))
    me._mask_undo_stacks.clear()
    me._label_boxes.clear()
    me._dirty_masks.clear()
    viewer_helpers.reset_label_ids_tracking()

    layer = viewer.add_labels(_volume(), name=MASK)
    layer.brush_size = 3
    viewer_helpers.add_or_update_label_ids(viewer, layer)
    centroids_name = MASK.replace(constants.MASKS_SUFFIX, constants.CENTROIDS_SUFFIX)
    coords, labels = _full_table(layer.data)
    viewer.add_points(coords, name=centroids_name, properties={'id': labels})

    me._on_mask_layer_changed(layer)
    real = me._mask_editor_widget
    monkeypatch.setattr(me, "_mask_editor_widget", types.SimpleNamespace(
        mask_layer=types.SimpleNamespace(value=layer), _function=real._function))

    full_passes = []
    real_centroids = segmentation.get_mask_centroids
    monkeypatch.setattr(segmentation, "get_mask_centroids",
                        lambda mask: (full_passes.append(1), real_centroids(mask))[1])
    yield types.SimpleNamespace(me=me, viewer=viewer, layer=layer, helpers=viewer_helpers,
                                full_passes=full_passes, centroids=viewer.layers[centroids_name])
    me._save_timer.stop()
    me._flush_timer.stop()
    me._dirty_masks.clear()
    me._mask_undo_stacks.clear()
    me._label_boxes.clear()
    viewer_helpers.reset_label_ids_tracking()


def _assert_ids_exact(env):
    ids = env.viewer.layers[f"{MASK}_IDs"]
    coords, labels = _full_table(env.layer.data)
    np.testing.assert_array_equal(np.asarray(ids.properties['label']), labels)
    np.testing.assert_array_equal(np.asarray(ids.data), coords)


def test_delete_updates_overlay_without_full_pass(env):
    env.me._delete_label_inplace(env.layer, 3)
    _assert_ids_exact(env)
    assert 3 not in env.viewer.layers[f"{MASK}_IDs"].properties['label']
    assert env.full_passes == []


def test_merge_moves_target_id_without_full_pass(env):
    env.me._mask_editor_widget._function(env.layer, 5, 6)
    _assert_ids_exact(env)
    assert env.full_passes == []


def test_paint_strokes_reach_overlay_on_flush(env):
    env.layer.mode = "paint"
    env.layer.paint((2, 25, 25), 11)        # a new nucleus
    env.layer.paint((1, 17, 3), 2)          # label 2 grows into label 4's area
    env.layer.paint((0, 2, 2), 0)           # erase part of label 1
    env.layer.mode = "pan_zoom"
    env.me._flush_mask_edits(env.layer, refresh_ids=True)
    _assert_ids_exact(env)
    assert env.full_passes == []


def test_napari_undo_and_redo_reach_overlay(env):
    """napari's own undo and redo fire no event; the update reads its history."""
    env.layer.paint((2, 25, 25), 11)
    env.me._refresh_ids(env.viewer, env.layer)
    env.layer.undo()
    env.me._refresh_ids(env.viewer, env.layer)
    _assert_ids_exact(env)
    env.layer.redo()
    env.me._refresh_ids(env.viewer, env.layer)
    _assert_ids_exact(env)
    assert env.full_passes == []


def test_napari_undo_after_editor_merge_reaches_overlay(env):
    """Paint label 11, merge 11 into 2 in the editor, then napari's undo: napari
    resets the stroke's voxels to their old value, removing label-2 voxels its
    history never recorded. Label 2's centroid must still be updated."""
    env.layer.n_edit_dimensions = 3
    env.layer.paint((4, 36, 40), 11)        # next to label 2
    env.me._refresh_ids(env.viewer, env.layer)
    env.me._mask_editor_widget._function(env.layer, 11, 2)
    env.layer.undo()                        # napari's undo of the stroke
    env.me._refresh_ids(env.viewer, env.layer)
    _assert_ids_exact(env)
    assert env.full_passes == []


def test_editor_undo_restores_ids_and_centroids(env):
    before = (np.asarray(env.centroids.data).copy(), np.asarray(env.centroids.properties['id']).copy())
    env.me._delete_label_inplace(env.layer, 3)
    assert 3 not in env.centroids.properties['id']
    env.me._on_mask_undo()
    _assert_ids_exact(env)
    np.testing.assert_array_equal(env.centroids.properties['id'], before[1])
    np.testing.assert_array_equal(env.centroids.data, before[0])
    assert env.full_passes == []


def test_whole_array_assignment_falls_back_to_full_refresh(env):
    moved = np.zeros_like(env.layer.data)
    moved[0:2, 30:40, 30:40] = 1
    moved[3:5, 0:10, 0:10] = 2
    env.layer.data = moved
    env.me._refresh_ids(env.viewer, env.layer)
    _assert_ids_exact(env)
    assert len(env.full_passes) == 1


def test_missing_overlay_is_created_by_full_refresh(env):
    env.viewer.layers.remove(f"{MASK}_IDs")
    env.me._delete_label_inplace(env.layer, 3)
    assert f"{MASK}_IDs" in env.viewer.layers
    _assert_ids_exact(env)
    assert len(env.full_passes) == 1


def test_no_change_leaves_overlay_untouched(env):
    ids = env.viewer.layers[f"{MASK}_IDs"]
    events = []
    ids.events.data.connect(lambda e: events.append(e))
    env.me._refresh_ids(env.viewer, env.layer)
    assert events == []


@pytest.mark.parametrize("seed", range(6))
def test_random_edits_keep_overlay_exact(env, seed):
    rng = np.random.default_rng(seed)
    for _ in range(40):
        op = rng.choice(["delete", "merge", "paint", "erase", "undo", "napari_undo", "napari_redo"])
        labels = [int(v) for v in np.unique(env.layer.data) if v]
        if op == "delete" and labels:
            env.me._delete_label_inplace(env.layer, int(rng.choice(labels)))
        elif op == "merge" and len(labels) >= 2:
            source, target = (int(v) for v in rng.choice(labels, 2, replace=False))
            env.me._mask_editor_widget._function(env.layer, source, target)
        elif op in ("paint", "erase"):
            undo = env.me._undo_for(env.layer)
            undo.begin_session()
            for _ in range(int(rng.integers(1, 4))):
                coord = (int(rng.integers(0, 6)), int(rng.integers(0, 40)), int(rng.integers(0, 48)))
                env.layer.paint(coord, int(rng.integers(1, 12)) if op == "paint" else 0)
            undo.end_session()
        elif op == "undo":
            env.me._on_mask_undo()
        elif op == "napari_undo":
            env.layer.undo()
        elif op == "napari_redo":
            env.layer.redo()
        env.me._refresh_ids(env.viewer, env.layer)
        _assert_ids_exact(env)
    assert env.full_passes == []
