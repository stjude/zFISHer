"""Bounding-box mask edits and undo (gap analysis §13.9, fix 1).

Deleting or merging one nucleus used to snapshot, scan and diff the whole mask.
It now touches only the label's bounding box (``core.label_edits``), and paint
and erase sessions take their undo record from napari's ``events.paint`` atoms
instead of a whole-volume snapshot. These tests check that masks and undo
results are identical to the old implementation, and that the boxes stay
supersets of their labels through every edit path, including napari's own undo.

The widget tests import the mask editor module, which builds Qt widgets at
import time, so they need a QApplication and are skipped where napari is not
installed.

Run (in an env with napari):  python -m pytest tests/test_label_edits.py -q
"""
import logging
import os
import sys
import tempfile
import tracemalloc
import types
from collections import deque

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy import ndimage

from zfisher.core import label_edits


def _volume():
    """A small mask with the awkward cases: labels on every border, a label in
    two separate pieces, two labels far apart, and labels that touch."""
    data = np.zeros((6, 40, 48), np.uint32)
    data[0:3, 0:6, 0:6] = 1          # corner, touches three borders
    data[3:6, 34:40, 42:48] = 2      # opposite corner
    data[1:4, 5:9, 20:26] = 3        # label 3, first piece
    data[2:5, 30:35, 10:14] = 3      # label 3, second piece, far from the first
    data[0:6, 15:20, 0:4] = 4        # spans all of z, left border
    data[2:4, 12:16, 30:40] = 5      # touches label 6
    data[2:4, 16:20, 30:40] = 6
    data[5, 0, 47] = 7               # a single voxel in a corner
    return data


def _assert_boxes_cover(data, boxes):
    """Every voxel of every label lies inside that label's box."""
    for label in np.unique(data):
        if label == 0:
            continue
        region = boxes.get(label)
        assert region is not None, f"label {label} has no box"
        inside = np.count_nonzero(data[region] == label)
        assert inside == np.count_nonzero(data == label), f"label {label} outside its box"


# --- core -------------------------------------------------------------------

def test_boxes_match_find_objects():
    data = _volume()
    boxes = label_edits.LabelBoxes(data)
    for i, region in enumerate(ndimage.find_objects(data)):
        assert boxes.get(i + 1) == region
    assert boxes.get(0) is None
    assert boxes.get(99) is None


@pytest.mark.parametrize("new", [0, 6, 50])
def test_replace_label_equals_whole_volume_replace(new):
    for old in range(1, 8):
        data = _volume()
        expected = data.copy()
        expected[expected == old] = new
        before = data.copy()
        boxes = label_edits.LabelBoxes(data)
        indices = label_edits.replace_label(data, boxes, old, new)
        np.testing.assert_array_equal(data, expected)
        # The returned indices are exactly the voxels that held ``old``.
        changed = np.zeros(data.shape, bool)
        changed[indices] = True
        np.testing.assert_array_equal(changed, before == old)
        _assert_boxes_cover(data, boxes)


def test_replace_missing_label_returns_none_and_changes_nothing():
    data = _volume()
    before = data.copy()
    boxes = label_edits.LabelBoxes(data)
    assert label_edits.replace_label(data, boxes, 42, 0) is None
    assert label_edits.count_label(data, boxes, 42) == 0
    np.testing.assert_array_equal(data, before)


def test_label_without_box_is_searched_in_whole_volume(caplog):
    data = _volume()
    boxes = label_edits.LabelBoxes(np.zeros_like(data))   # knows no labels
    with caplog.at_level(logging.WARNING, logger="zfisher.core.label_edits"):
        indices = label_edits.replace_label(data, boxes, 3, 0)
    assert indices is not None and not np.any(data == 3)
    assert "no bounding box" in caplog.text


def test_count_label_counts_inside_box():
    data = _volume()
    boxes = label_edits.LabelBoxes(data)
    for label in range(1, 8):
        assert label_edits.count_label(data, boxes, label) == np.count_nonzero(data == label)


def test_extend_values_grows_and_ignores_background():
    boxes = label_edits.LabelBoxes(np.zeros((4, 10, 10), np.uint32))
    indices = (np.array([1, 2]), np.array([3, 7]), np.array([0, 9]))
    boxes.extend_values(indices, np.uint32(5))
    assert boxes.get(5) == (slice(1, 3), slice(3, 8), slice(0, 10))
    boxes.extend_values(indices, np.array([0, 8], np.uint32))
    assert boxes.get(0) is None
    assert boxes.get(8) == boxes.get(5)
    # Boxes never shrink.
    boxes.extend(5, (slice(2, 3), slice(4, 5), slice(4, 5)))
    assert boxes.get(5) == (slice(1, 3), slice(3, 8), slice(0, 10))


# --- mask editor ------------------------------------------------------------

class _OldUndoStack:
    """The whole-volume snapshot-and-diff undo stack this change replaces."""

    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)
        self._pre_edit = None

    def begin(self, data):
        self._pre_edit = data.copy()

    def end(self, data):
        if self._pre_edit is None:
            return
        diff_mask = self._pre_edit != data
        if np.any(diff_mask):
            indices = np.where(diff_mask)
            self._stack.append((indices, self._pre_edit[indices]))
        self._pre_edit = None

    def undo(self, data):
        if not self._stack:
            return False
        indices, old_values = self._stack.pop()
        data[indices] = old_values
        return True


@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def me(_qapp, monkeypatch):
    """The mask editor module with a stub viewer, no session side effects and no
    deferred Qt callbacks."""
    import zfisher.ui.widgets.mask_editor_widget as me_mod
    from zfisher.core import session
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (tmp if key == "output_dir" else default))
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    viewer = types.SimpleNamespace(status="", layers=[])
    monkeypatch.setattr(me_mod.napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(me_mod, "QTimer", types.SimpleNamespace(singleShot=lambda *a, **k: None))
    monkeypatch.setattr(me_mod.viewer_helpers, "add_or_update_label_ids", lambda *a, **k: None)
    me_mod._mask_undo_stacks.clear()
    me_mod._label_boxes.clear()
    me_mod._dirty_masks.clear()
    yield me_mod
    me_mod._save_timer.stop()
    me_mod._flush_timer.stop()
    me_mod._dirty_masks.clear()
    me_mod._mask_undo_stacks.clear()
    me_mod._label_boxes.clear()


def _layer(me, monkeypatch, data, name="R1 - DAPI_masks"):
    """A mask layer selected in the editor, with its paint callbacks connected."""
    from napari.layers import Labels
    layer = Labels(data, name=name)
    layer.brush_size = 3
    # magicgui refuses to have a widget attribute replaced, so the module's
    # reference to the editor is swapped for a stub holding the selection.
    if not isinstance(me._mask_editor_widget, types.SimpleNamespace):
        real = me._mask_editor_widget
        monkeypatch.setattr(me, "_mask_editor_widget", types.SimpleNamespace(
            mask_layer=types.SimpleNamespace(value=None), _function=real._function,
            _current_layer=None, _current_callbacks=None))
    _select(me, layer)
    return layer


def _select(me, layer):
    """Choose ``layer`` in the editor's "Layer to Edit"."""
    me._mask_editor_widget.mask_layer.value = layer
    me._on_mask_layer_changed(layer)


def _merge(me, layer, source, target):
    me._mask_editor_widget._function(layer, source, target)


def test_delete_touches_only_the_box(me, monkeypatch):
    data = np.zeros((20, 256, 256), np.uint32)
    data[5:9, 100:110, 100:110] = 3
    data[0:20, 0:50, 0:50] = 4
    layer = _layer(me, monkeypatch, data)
    me._boxes_for(layer)               # built once per layer, not per edit
    tracemalloc.start()
    me._delete_label_inplace(layer, 3)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert not np.any(layer.data == 3)
    assert np.count_nonzero(layer.data == 4) == 20 * 50 * 50
    # The old path copied the volume and built a full-size boolean diff.
    assert peak < data.nbytes / 20, f"delete allocated {peak} bytes"


def test_delete_and_merge_undo_round_trip(me, monkeypatch):
    original = _volume()
    layer = _layer(me, monkeypatch, original.copy())
    me._delete_label_inplace(layer, 3)
    _merge(me, layer, 5, 6)
    me._delete_label_inplace(layer, 1)
    me._on_mask_undo()
    me._on_mask_undo()
    me._on_mask_undo()
    np.testing.assert_array_equal(layer.data, original)
    _assert_boxes_cover(layer.data, me._boxes_for(layer))


def test_paint_session_undo_uses_napari_atoms(me, monkeypatch):
    original = _volume()
    layer = _layer(me, monkeypatch, original.copy())
    layer.mode = "paint"
    undo = me._undo_for(layer)
    undo.begin_session()
    layer.paint((2, 10, 10), 9)
    layer.paint((2, 11, 12), 9)        # overlaps the first stroke
    layer.paint((2, 15, 2), 0)         # erases part of label 4
    undo.end_session()
    assert undo._pre_edit is None      # no snapshot was taken
    assert not np.array_equal(layer.data, original)
    me._on_mask_undo()
    np.testing.assert_array_equal(layer.data, original)


def test_paint_after_boxes_built_extends_them(me, monkeypatch):
    layer = _layer(me, monkeypatch, _volume())
    me._boxes_for(layer)
    layer.paint((4, 38, 2), 3)         # label 3 now also far outside its box
    _assert_boxes_cover(layer.data, me._boxes_for(layer))
    me._delete_label_inplace(layer, 3)
    assert not np.any(layer.data == 3)


def test_napari_undo_cannot_escape_the_boxes(me, monkeypatch):
    """napari's own undo writes old values back without any event. Painting
    over the right edge of label 4 shrinks what find_objects reports, and
    napari's undo then puts label 4 back outside that box."""
    layer = _layer(me, monkeypatch, _volume())
    layer.n_edit_dimensions = 3
    layer.brush_size = 1
    with layer.block_history():
        for z in range(6):
            for y in range(15, 20):
                layer.paint((z, y, 3), 9)
    assert ndimage.find_objects(layer.data)[3][2] == slice(0, 3)
    me._boxes_for(layer)               # built after the stroke
    layer.undo()                       # napari's undo, not the editor's
    assert np.count_nonzero(layer.data == 4) == 6 * 5 * 4
    me._delete_label_inplace(layer, 4)
    assert not np.any(layer.data == 4)


def test_whole_array_assignment_drops_boxes(me, monkeypatch):
    layer = _layer(me, monkeypatch, _volume())
    me._boxes_for(layer)
    moved = np.zeros_like(layer.data)
    moved[0:2, 30:40, 30:40] = 1
    layer.data = moved
    me._delete_label_inplace(layer, 1)
    assert not np.any(layer.data == 1)


def test_extrude_then_delete_removes_every_voxel(me, monkeypatch):
    layer = _layer(me, monkeypatch, _volume())
    me._boxes_for(layer)
    monkeypatch.setattr(me, "_extrude_spinbox", types.SimpleNamespace(value=3))
    me._on_extrude()
    assert np.count_nonzero(layer.data[:, 5:9, 20:26] == 3) == 6 * 4 * 6
    me._delete_label_inplace(layer, 3)
    assert not np.any(layer.data == 3)


def test_edit_during_paint_session_keeps_undo_order(me, monkeypatch):
    """Paint with label 5, merge 5 into 6 while still painting, paint again.
    Undoing everything must give back the original, which needs the strokes
    before the merge to be undone after the merge, not before it."""
    original = _volume()
    layer = _layer(me, monkeypatch, original.copy())
    undo = me._undo_for(layer)
    undo.begin_session()
    layer.paint((1, 25, 25), 5)
    _merge(me, layer, 5, 6)
    layer.paint((1, 30, 40), 2)
    undo.end_session()
    while len(undo):
        me._on_mask_undo()
    np.testing.assert_array_equal(layer.data, original)


@pytest.mark.parametrize("seed", range(6))
def test_random_edits_match_old_implementation(me, monkeypatch, seed):
    """Mask after every step, and after every undo, equals the old
    whole-volume implementation."""
    rng = np.random.default_rng(seed)
    layer = _layer(me, monkeypatch, _volume())
    ref = _volume()
    old = _OldUndoStack()
    for _ in range(40):
        op = rng.choice(["delete", "merge", "paint", "erase", "undo", "undo"])
        labels = [int(v) for v in np.unique(ref) if v]
        if op == "delete" and labels:
            label = int(rng.choice(labels))
            me._delete_label_inplace(layer, label)
            old.begin(ref)
            ref[ref == label] = 0
            old.end(ref)
        elif op == "merge" and len(labels) >= 2:
            source, target = (int(v) for v in rng.choice(labels, 2, replace=False))
            _merge(me, layer, source, target)
            old.begin(ref)
            ref[ref == source] = target
            old.end(ref)
        elif op in ("paint", "erase"):
            old.begin(ref)
            me._undo_for(layer).begin_session()
            for _ in range(int(rng.integers(1, 4))):
                coord = (int(rng.integers(0, 6)), int(rng.integers(0, 40)), int(rng.integers(0, 48)))
                layer.paint(coord, int(rng.integers(1, 12)) if op == "paint" else 0)
            me._undo_for(layer).end_session()
            ref[...] = layer.data      # same strokes, recorded the old way
            old.end(ref)
        elif op == "undo":
            me._on_mask_undo()
            old.undo(ref)
        np.testing.assert_array_equal(layer.data, ref)
        _assert_boxes_cover(layer.data, me._boxes_for(layer))


def test_undo_never_writes_into_another_layer(me, monkeypatch):
    """Edit A, select B, undo: B is untouched. A's edit is still undoable
    once A is selected again."""
    original_a = _volume()
    original_b = _volume()[:, ::-1, :].copy()
    a = _layer(me, monkeypatch, original_a.copy())
    b = _layer(me, monkeypatch, original_b.copy(), name="R2 - DAPI_masks")
    _select(me, a)
    me._delete_label_inplace(a, 3)
    deleted_a = a.data.copy()
    _select(me, b)
    me._on_mask_undo()
    np.testing.assert_array_equal(b.data, original_b)
    np.testing.assert_array_equal(a.data, deleted_a)
    assert me.napari.current_viewer().status == "Nothing to undo."
    _select(me, a)
    me._on_mask_undo()
    np.testing.assert_array_equal(a.data, original_a)
    np.testing.assert_array_equal(b.data, original_b)
    _assert_boxes_cover(a.data, me._boxes_for(a))


def test_layer_change_closes_paint_session(me, monkeypatch):
    """A paint session started on A is stored on A when B is selected, and
    B's strokes go to B's own history."""
    original_a = _volume()
    original_b = _volume()[::-1].copy()
    a = _layer(me, monkeypatch, original_a.copy())
    b = _layer(me, monkeypatch, original_b.copy(), name="R2 - DAPI_masks")
    b.mode = "paint"                   # B is selected later, already painting
    _select(me, a)
    a.mode = "paint"
    me._undo_for(a).begin_session()
    a.paint((2, 10, 10), 9)
    _select(me, b)
    assert not me._undo_for(a).in_session
    assert len(me._undo_for(a)) == 1
    assert me._undo_for(b).in_session
    b.paint((3, 25, 25), 8)
    painted_b = b.data.copy()
    _select(me, a)
    assert not me._undo_for(b).in_session
    me._on_mask_undo()
    np.testing.assert_array_equal(a.data, original_a)
    np.testing.assert_array_equal(b.data, painted_b)
    _select(me, b)
    me._on_mask_undo()
    np.testing.assert_array_equal(b.data, original_b)


def test_undo_after_data_replaced_does_nothing(me, monkeypatch):
    """Records index into the array they came from; once the layer holds a
    new array (re-segmentation, consensus rebuild) they are dropped."""
    layer = _layer(me, monkeypatch, _volume())
    me._delete_label_inplace(layer, 3)
    replacement = _volume()[:, :, ::-1].copy()
    layer.data = replacement.copy()
    me._on_mask_undo()
    np.testing.assert_array_equal(layer.data, replacement)
    assert me.napari.current_viewer().status == "Nothing to undo."
