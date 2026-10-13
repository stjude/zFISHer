"""Regression tests for shrinking a puncta Points layer without detaching its features.

napari truncates ``layer.features`` to the first N rows the instant ``layer.data``
shrinks. Code that shrinks the data and then subsets the features therefore either
skips the subset (length check fails) or reads an already-mangled table, and every
surviving punctum inherits the ``puncta_id``/``Nucleus_ID``/``Source`` of whichever
punctum used to sit at its new row index. ``viewer_helpers.subset_points_layer`` is
the single safe path; the Refilter Puncta widget and its undo stack use it.

These exercise a real napari Points layer, so they need napari/Qt and are skipped
where napari is not installed.

Run (in an env with napari):  python -m pytest tests/test_refilter_features.py -q
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


def _features(n=5):
    return pd.DataFrame({
        "puncta_id": [101 + i for i in range(n)],
        "Nucleus_ID": [1 + i for i in range(n)],
        "Intensity": [1.0 + i for i in range(n)],
        "SNR": [1.5 + i for i in range(n)],
        "Source": ["auto", "manual", "auto", "manual", "auto"][:n],
    })


def _layer(n=5, features=True, **kwargs):
    from napari.layers import Points
    data = np.array([[0.0, 10.0 * (i + 1), 10.0 * (i + 1)] for i in range(n)])
    feats = _features(n) if features else None
    return Points(data, features=feats, name="Aligned R1 - FITC_puncta", **kwargs)


def test_subset_keeps_features_aligned(_qapp):
    """The reported bug: surviving points must carry their own feature rows."""
    from zfisher.ui import viewer_helpers
    layer = _layer()
    keep = np.array([False, True, False, True, True])
    removed = viewer_helpers.subset_points_layer(layer, keep)
    assert removed == 2
    assert len(layer.data) == 3
    np.testing.assert_array_equal(layer.data[:, 1], [20.0, 40.0, 50.0])
    expected = _features().iloc[keep].reset_index(drop=True)
    pd.testing.assert_frame_equal(layer.features[expected.columns], expected, check_dtype=False)


def test_shrink_then_subset_is_the_bug(_qapp):
    """Documents why the order matters: shrinking first leaves the first-N rows behind."""
    layer = _layer()
    keep = np.array([False, True, False, True, True])
    layer.data = layer.data[keep]
    # napari has already truncated the table; the old length guard cannot pass.
    assert len(layer.features) == 3
    assert list(layer.features["puncta_id"]) == [101, 102, 103]  # not 102, 104, 105


def test_subset_no_features(_qapp):
    from zfisher.ui import viewer_helpers
    layer = _layer(features=False)
    removed = viewer_helpers.subset_points_layer(layer, np.array([True, False, True, False, True]))
    assert removed == 2
    assert len(layer.data) == 3


def test_subset_all_kept(_qapp):
    from zfisher.ui import viewer_helpers
    layer = _layer()
    before = layer.features.copy()
    removed = viewer_helpers.subset_points_layer(layer, np.ones(5, bool))
    assert removed == 0
    assert len(layer.data) == 5
    pd.testing.assert_frame_equal(layer.features[before.columns], before, check_dtype=False)


def test_subset_rejects_wrong_length_mask(_qapp):
    from zfisher.ui import viewer_helpers
    layer = _layer()
    with pytest.raises(ValueError):
        viewer_helpers.subset_points_layer(layer, np.ones(4, bool))


def test_refilter_end_to_end(_qapp):
    """The refilter loop body against a real Labels mask on a different scale/translate."""
    from napari.layers import Labels
    from zfisher.core import puncta
    from zfisher.ui import viewer_helpers

    # Points at y=x=10,20,30,40,50 in their own frame; scale 2 -> world 20..100.
    layer = _layer(scale=(1, 2, 2))
    # Mask on scale 1 with a +5 translate in y/x; label the world positions of
    # points 1, 3, 4 (world y=x=40, 80, 100 -> mask voxels 35, 75, 95).
    mask = np.zeros((1, 128, 128), np.int32)
    for label, v in ((7, 35), (8, 75), (9, 95)):
        mask[0, v - 2:v + 3, v - 2:v + 3] = label
    mask_layer = Labels(mask, translate=(0, 5, 5))

    labels = puncta.lookup_label_ids(
        layer.data, layer.scale, layer.translate,
        mask_layer.data, mask_layer.scale, mask_layer.translate,
    )
    assert list(labels) == [0, 7, 0, 8, 9]

    removed = viewer_helpers.subset_points_layer(layer, labels > 0)
    assert removed == 2
    assert list(layer.features["puncta_id"]) == [102, 104, 105]
    assert list(layer.features["Nucleus_ID"]) == [2, 4, 5]
    assert list(layer.features["Source"]) == ["manual", "manual", "auto"]


def test_undo_restores_data_and_features(_qapp, monkeypatch):
    """The refilter undo stack round-trips both arrays through set_points_data."""
    from zfisher.ui import viewer_helpers
    import zfisher.ui.widgets.refilter_puncta_widget as rw

    layer = _layer()
    orig_data = layer.data.copy()
    orig_feats = layer.features.copy()

    stack = rw._RefilterUndoStack()
    stack.push([layer])
    viewer_helpers.subset_points_layer(layer, np.array([False, True, False, True, True]))
    assert len(layer.data) == 3

    calls = []
    real = viewer_helpers.set_points_data
    monkeypatch.setattr(viewer_helpers, "set_points_data",
                        lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    assert stack.undo() is True
    assert len(calls) == 1  # undo went through the GL-safe path
    np.testing.assert_array_equal(layer.data, orig_data)
    pd.testing.assert_frame_equal(layer.features[orig_feats.columns], orig_feats, check_dtype=False)
    assert stack.undo() is False
