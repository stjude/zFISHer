"""Partial redraw of a mask after an in-place edit.

Delete, merge and undo in the mask editor used to call ``layer.refresh()``,
which re-colours the whole displayed slice and uploads it again: in 3D the whole
volume. They now go through ``viewer_helpers.redraw_labels_voxels``, which
re-colours only the changed voxels and uploads their bounding box, the way
napari redraws a brush stroke. These tests check that the displayed slice then
equals what a full refresh produces, in 2D and 3D and with a transposed axis
order, that the upload covers only the edit, and that no full refresh ran.

The GPU side (the texture upload in ``VispyLabelsLayer``) needs a display and is
not tested here. The helper uses napari private API; napari is pinned to 0.6.6.

Run (in an env with napari):  python -m pytest tests/test_labels_redraw.py -q
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(_qapp, monkeypatch):
    """A headless viewer holding a mask; the mask editor with deferred callbacks
    run at once; counts of full refreshes and the partial updates sent to the
    vispy layer."""
    from napari.components import ViewerModel
    import zfisher.ui.widgets.mask_editor_widget as me
    from zfisher.core import session
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
    me._on_mask_layer_changed(layer)
    real = me._mask_editor_widget
    monkeypatch.setattr(me, "_mask_editor_widget", types.SimpleNamespace(
        mask_layer=types.SimpleNamespace(value=layer), _function=real._function))

    refreshes, updates = [], []
    real_refresh = layer.refresh
    monkeypatch.setattr(layer, "refresh", lambda *a, **k: (refreshes.append(1), real_refresh(*a, **k))[1])
    layer.events.labels_update.connect(
        lambda e: updates.append((tuple(e.offset), np.shape(e.data))))
    yield types.SimpleNamespace(me=me, viewer=viewer, layer=layer, helpers=viewer_helpers,
                                refreshes=refreshes, updates=updates, real_refresh=real_refresh)
    me._save_timer.stop()
    me._flush_timer.stop()
    me._dirty_masks.clear()
    me._mask_undo_stacks.clear()
    me._label_boxes.clear()
    viewer_helpers.reset_label_ids_tracking()


def _assert_shown_equals_full_refresh(env):
    """The displayed slice equals the one a full refresh computes from the data."""
    shown = np.array(env.layer._slice.image.view, copy=True)
    env.real_refresh()
    np.testing.assert_array_equal(shown, env.layer._slice.image.view)


def _set_view(env, ndisplay, order=(0, 1, 2), z=2):
    """``z`` is the position on the first axis of ``order``, the 2D slider."""
    env.viewer.dims.ndisplay = ndisplay
    env.viewer.dims.order = order
    env.viewer.dims.set_point(order[0], z)


VIEWS = [(3, (0, 1, 2)), (3, (2, 0, 1)), (2, (0, 1, 2)), (2, (1, 0, 2))]


@pytest.mark.parametrize("ndisplay,order", VIEWS)
def test_delete_redraws_only_the_label(env, ndisplay, order):
    # Label 5's box: z 2:4, y 12:16, x 30:40, as (start, size) per axis.
    box = {0: (2, 2), 1: (12, 4), 2: (30, 10)}
    _set_view(env, ndisplay, order, z=box[order[0]][0] + 1)   # a 2D slice through it
    env.me._delete_label_inplace(env.layer, 5)
    assert not np.any(env.layer.data == 5)
    _assert_shown_equals_full_refresh(env)
    assert env.refreshes == []
    assert len(env.updates) == 1                # only the box was uploaded
    displayed = order[-ndisplay:]
    offset, shape = env.updates[0]
    assert offset == tuple(box[ax][0] for ax in displayed)
    assert shape == tuple(box[ax][1] for ax in displayed)


@pytest.mark.parametrize("ndisplay,order", VIEWS)
def test_merge_redraws_only_the_source(env, ndisplay, order):
    _set_view(env, ndisplay, order)
    env.me._mask_editor_widget._function(env.layer, 3, 4)
    assert not np.any(env.layer.data == 3)
    _assert_shown_equals_full_refresh(env)
    assert env.refreshes == []


def test_edit_outside_the_shown_plane_uploads_nothing(env):
    _set_view(env, 2, z=0)                      # label 5 lies in planes 2-3
    before = np.array(env.layer._slice.image.view, copy=True)
    env.me._delete_label_inplace(env.layer, 5)
    np.testing.assert_array_equal(env.layer._slice.image.view, before)
    _assert_shown_equals_full_refresh(env)
    assert env.updates == [] and env.refreshes == []
    env.viewer.dims.set_point(0, 2)             # moving to the plane re-slices it
    assert not np.any(env.layer._slice.image.raw == 5)


@pytest.mark.parametrize("ndisplay", [2, 3])
def test_editor_undo_redraws_without_refresh(env, ndisplay):
    _set_view(env, ndisplay)
    original = env.layer.data.copy()
    env.me._delete_label_inplace(env.layer, 3)
    env.me._mask_editor_widget._function(env.layer, 5, 6)
    env.me._on_mask_undo()
    env.me._on_mask_undo()
    np.testing.assert_array_equal(env.layer.data, original)
    _assert_shown_equals_full_refresh(env)
    assert env.refreshes == []


def test_undo_of_paint_session_redraws_without_refresh(env):
    _set_view(env, 3)
    original = env.layer.data.copy()
    env.layer.brush_size = 3
    env.layer.mode = "paint"
    undo = env.me._undo_for(env.layer)
    undo.begin_session()                        # as the Paint button does
    env.layer.paint((2, 25, 25), 11)
    env.layer.paint((1, 17, 3), 2)
    env.me._on_mask_undo()                      # undoes both strokes as one record
    np.testing.assert_array_equal(env.layer.data, original)
    _assert_shown_equals_full_refresh(env)
    assert env.refreshes == []


def test_whole_volume_undo_still_refreshes(env):
    """Extrude's undo record is a whole-volume diff; it keeps the full refresh."""
    _set_view(env, 3)
    env.me._extrude_spinbox.value = 5
    env.me._on_extrude()
    env.refreshes.clear()
    env.me._on_mask_undo()
    assert env.refreshes == [1]
    np.testing.assert_array_equal(env.layer.data, _volume())


def test_expanded_colormap_is_redrawn_exactly(env):
    """The hover highlighter gives every label its own colour slot, which moves
    the texture from uint8 to uint16."""
    from napari.utils.colormaps import label_colormap
    _set_view(env, 3)
    env.layer.colormap = label_colormap(1001)  # napari refreshes on this itself
    env.refreshes.clear()
    env.me._mask_editor_widget._function(env.layer, 1, 900)
    assert env.layer._slice.image.view.dtype == np.uint16
    _assert_shown_equals_full_refresh(env)
    assert env.refreshes == []


def test_contours_fall_back_to_refresh(env):
    _set_view(env, 2)
    env.layer.contour = 1
    env.refreshes.clear()
    env.me._delete_label_inplace(env.layer, 5)
    assert env.refreshes == [1]
    _assert_shown_equals_full_refresh(env)


def test_nothing_changed_draws_nothing(env):
    _set_view(env, 3)
    assert env.helpers.redraw_labels_voxels(env.layer, [None, (np.array([], int),) * 3])
    assert env.updates == [] and env.refreshes == []


@pytest.mark.parametrize("seed", range(6))
def test_random_edits_keep_display_exact(env, seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        if rng.random() < 0.2:
            ndisplay, order = VIEWS[rng.integers(len(VIEWS))]
            _set_view(env, ndisplay, order, z=int(rng.integers(6)))
        labels = [int(v) for v in np.unique(env.layer.data) if v]
        op = rng.integers(3)
        if op == 0 and labels:
            env.me._delete_label_inplace(env.layer, int(rng.choice(labels)))
        elif op == 1 and len(labels) >= 2:
            src, dst = rng.choice(labels, 2, replace=False)
            env.me._mask_editor_widget._function(env.layer, int(src), int(dst))
        else:
            env.me._on_mask_undo()
        # No refresh between steps, so redraws build on each other.
        raw = np.asarray(env.layer._slice.image.raw)
        np.testing.assert_array_equal(env.layer._slice.image.view, env.layer._raw_to_displayed(raw))
    assert env.refreshes == []
    _assert_shown_equals_full_refresh(env)
