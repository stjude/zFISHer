"""Regression tests for re-deriving puncta Nucleus_ID from the consensus mask.

``Nucleus_ID`` on a punctum is a cache of which mask label sits under it.
``viewer_helpers.resync_puncta_nucleus_ids`` refreshes that cache for the
aligned/warped puncta layers and is now cheap to call after every edit: a layer
whose IDs did not change is not touched, not re-saved, and not re-registered.
``reconcile_puncta_with_consensus`` wraps it for the load and export paths.

These exercise real napari layers, so they need napari/Qt and are skipped where
napari is not installed.

Run (in an env with napari):  python -m pytest tests/test_resync_nucleus_ids.py -q
"""
import os
import sys
import tempfile
from pathlib import Path

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


class _Viewer:
    """Minimal stand-in: resync only iterates ``viewer.layers``."""
    def __init__(self, layers):
        self.layers = list(layers)


def _points(name, ys, nucleus_ids, scale=(1, 1, 1)):
    from napari.layers import Points
    data = np.array([[0.0, y, y] for y in ys])
    feats = pd.DataFrame({
        "puncta_id": np.arange(100, 100 + len(ys)),
        "Nucleus_ID": list(nucleus_ids),
        "Intensity": [1.0] * len(ys), "SNR": [2.0] * len(ys),
        "Source": ["auto"] * len(ys),
    })
    return Points(data, features=feats, name=name, scale=scale)


def _consensus(labels_at):
    """Consensus Labels layer with a 5x5 block of each label centred at (y, y)."""
    from napari.layers import Labels
    from zfisher import constants
    mask = np.zeros((1, 64, 64), np.int32)
    for label, y in labels_at.items():
        mask[0, y - 2:y + 3, y - 2:y + 3] = label
    return Labels(mask, name=constants.CONSENSUS_MASKS_NAME)


def _session_output_dir(path):
    from zfisher.core import session
    orig_get = session.get_data
    orig_set = session.set_processed_file
    calls = []
    session.get_data = lambda key=None, default=None: (str(path) if key == "output_dir" else default)
    session.set_processed_file = lambda *a, **k: calls.append(a)
    return calls, (orig_get, orig_set)


def _restore_session(originals):
    from zfisher.core import session
    session.get_data, session.set_processed_file = originals


def test_resync_updates_only_changed_aligned_layers(_qapp):
    from zfisher import constants
    from zfisher.ui import viewer_helpers
    aligned = _points("Aligned R1 - FITC_puncta", [10, 20, 30], [7, 7, 7])   # stale: all 7
    raw = _points("R1 - FITC_puncta", [10, 20, 30], [7, 7, 7])               # raw space: must be left alone
    mask = _consensus({1: 10, 2: 20})                                          # y=30 is background
    viewer = _Viewer([mask, aligned, raw])

    with tempfile.TemporaryDirectory() as d:
        calls, originals = _session_output_dir(d)
        try:
            out = viewer_helpers.resync_puncta_nucleus_ids(viewer, mask, remove_extranuclear=False, save_csv=True)
            assert out == {"updated_layers": 1, "removed_total": 0, "changed_total": 3}
            assert list(aligned.features["Nucleus_ID"]) == [1, 2, 0]
            assert list(raw.features["Nucleus_ID"]) == [7, 7, 7]
            assert len(aligned.data) == 3                       # nothing removed
            csv = constants.puncta_csv_path(Path(d) / constants.REPORTS_DIR, aligned.name)
            assert csv.exists()
            cols = list(pd.read_csv(csv).columns)
            assert cols[:len(constants.PUNCTA_CSV_COLUMNS)] == constants.PUNCTA_CSV_COLUMNS
            assert len(calls) == 1                              # re-registered once

            # Second call: nothing changed -> no write, no registration.
            csv.unlink()
            out2 = viewer_helpers.resync_puncta_nucleus_ids(viewer, mask, remove_extranuclear=False, save_csv=True)
            assert out2["changed_total"] == 0 and out2["updated_layers"] == 0
            assert not csv.exists()
            assert len(calls) == 1
        finally:
            _restore_session(originals)


def test_resync_remove_extranuclear_keeps_features_aligned(_qapp):
    from zfisher.ui import viewer_helpers
    aligned = _points("Warped R2 - Cy5_puncta", [10, 20, 30], [0, 0, 0])
    mask = _consensus({1: 10, 2: 20})
    viewer = _Viewer([mask, aligned])
    out = viewer_helpers.resync_puncta_nucleus_ids(viewer, mask, remove_extranuclear=True, save_csv=False)
    assert out["removed_total"] == 1
    assert len(aligned.data) == 2
    assert list(aligned.features["Nucleus_ID"]) == [1, 2]
    assert list(aligned.features["puncta_id"]) == [100, 101]   # the survivors keep their own rows


def test_resync_respects_layer_scale(_qapp):
    from zfisher.ui import viewer_helpers
    # Points at data y=5,10 with scale 2 -> world y=10,20 -> mask labels 1, 2.
    aligned = _points("Aligned R1 - FITC_puncta", [5, 10], [0, 0], scale=(1, 2, 2))
    mask = _consensus({1: 10, 2: 20})
    out = viewer_helpers.resync_puncta_nucleus_ids(_Viewer([mask, aligned]), mask, save_csv=False)
    assert out["changed_total"] == 2
    assert list(aligned.features["Nucleus_ID"]) == [1, 2]


def test_reconcile_finds_consensus_or_returns_none(_qapp):
    from zfisher.ui import viewer_helpers
    aligned = _points("Aligned R1 - FITC_puncta", [10], [0])
    assert viewer_helpers.reconcile_puncta_with_consensus(_Viewer([aligned])) is None
    mask = _consensus({4: 10})
    out = viewer_helpers.reconcile_puncta_with_consensus(_Viewer([aligned, mask]), save_csv=False)
    assert out["changed_total"] == 1
    assert list(aligned.features["Nucleus_ID"]) == [4]
