"""The Puncta Editor's undo swaps points through ``set_points_data``.

Undo assigned ``layer.data`` directly, clearing napari's view-index cache by
hand. ``viewer_helpers.set_points_data`` is the one place that knows the safe
order (cache cleared, then data, then features); a point-count change against a
stale cache can corrupt the GL buffer or raise in vispy. The refilter undo was
routed through it in September; this does the same for the editor.

Run:  python -m pytest tests/test_puncta_editor_undo.py -q
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_undo_goes_through_set_points_data_and_restores_ids(_qapp, monkeypatch):
    from napari.layers import Points
    from zfisher.ui import viewer_helpers
    import zfisher.ui.widgets.puncta_editor_widget as pe

    feats = pd.DataFrame({"puncta_id": [10, 11, 12, 13, 14], "Nucleus_ID": 1, "Intensity": 5.0,
                          "SNR": 2.0, "Source": ["auto", "auto", "manual", "auto", "auto"]})
    layer = Points(np.array([[1, i * 10, i * 10] for i in range(5)], float), features=feats,
                   name="R1 - FITC_puncta", text={"string": "{puncta_id:.0f}"})
    layer.refresh()
    stack = pe._PunctaUndoStack()
    stack.push(layer)
    viewer_helpers.subset_points_layer(layer, np.array([True, False, True, False, False]))   # a delete
    assert len(layer.data) == 2

    calls = []
    real = viewer_helpers.set_points_data
    monkeypatch.setattr(viewer_helpers, "set_points_data",
                        lambda *a, **k: (calls.append(len(a[1])), real(*a, **k))[1])
    assert stack.undo(current_layer=layer) == ("data", layer)
    assert calls == [5]                                    # one GL-safe swap, to the 5 points
    assert list(layer.features["puncta_id"]) == [10, 11, 12, 13, 14]
    assert layer.text.string.format == "{puncta_id:.0f}"     # text format restored
    assert len(layer._indices_view) <= len(layer.data)
