"""Re-detecting a channel from the GUI, on its original image.

In a reopened session, Detect Puncta on an aligned or warped image layer, with
"Detect on original image, then warp" ticked, re-detects the channel on
the round's original image and warps the puncta onto the aligned layer
(``stages.redetect_channel``), replacing the layer in place. Unticked, it
detects on the aligned image as before, after a warning for R2. The export
section says when the report is out of date.

Headless: a napari ``ViewerModel`` stands in for the viewer, and the progress
dialog and pop-ups are stand-ins. What the window shows can only be checked in
the app.

Run:  python -m pytest tests/test_redetect_gui.py -q
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import contextlib

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import io, session, stages

from test_stages import CHANNEL, _append, _export_report, _states, finished, reopened  # noqa: F401  (fixtures)

R2_IMAGE = f"{constants.WARPED_PREFIX} R2 - {CHANNEL}"
R2_PUNCTA = f"{R2_IMAGE}{constants.PUNCTA_SUFFIX}"


@pytest.fixture
def gui(reopened, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import events, popups, viewer_helpers
    from zfisher.ui.widgets import colocalization_widget, puncta_widget

    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

        def __setattr__(self, name, value):
            setattr(model, name, value)

    viewer = _Viewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)

    class _Dialog(contextlib.nullcontext):
        def __init__(self, *a, **k):
            super().__init__(self)

        def update_progress(self, *a):
            pass

        def freeze_canvas(self):
            pass

    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    monkeypatch.setattr(popups, "resolve_raw_path", lambda parent, round_id: io.resolve_raw_path(round_id))
    shown = []
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: shown.append((title, text)))

    scale = tuple(session.get_data("canvas_scale"))
    image = model.add_image(np.zeros((4, 8, 8), np.uint16), name=R2_IMAGE, scale=scale)
    csv = constants.puncta_csv_path(reopened.reports, R2_PUNCTA)
    viewer_helpers._load_points_layer(model, R2_PUNCTA, csv, scale, {"subtype": "puncta_csv"}, (0, 0, 0))
    events.lock_layer(model.layers[R2_PUNCTA])
    return types.SimpleNamespace(viewer=viewer, model=model, image=image, csv=csv, shown=shown,
                                 widget=puncta_widget, coloc=colocalization_widget, popups=popups,
                                 helpers=viewer_helpers, reports=reopened.reports, reopened=reopened)


def _params(**changes):
    return dict(stages.recorded_detection_parameters("R2", CHANNEL), **changes)


def test_redetect_replaces_the_aligned_layer_in_place_and_keeps_hand_picks(gui):
    # One punctum placed by hand on the aligned layer, as the editor saves it.
    _append(gui.csv, [{"puncta_id": 9000, "Z": 15.5, "Y": 60.25, "X": 70.75, "Nucleus_ID": 0,
                       "Intensity": 0, "SNR": 0, "Source": "manual"}])
    gui.model.layers.remove(R2_PUNCTA)
    gui.helpers._load_points_layer(gui.model, R2_PUNCTA, gui.csv, gui.image.scale, {"subtype": "puncta_csv"}, (0, 0, 0))
    layer = gui.model.layers[R2_PUNCTA]
    layer._locked = True
    before = len(layer.data)

    gui.widget._redetect_on_original(gui.viewer, gui.image, _params(threshold_rel=0.6), "replace")

    assert gui.model.layers[R2_PUNCTA] is layer and layer._locked          # the same layer, still locked
    table = pd.read_csv(gui.csv)
    np.testing.assert_allclose(layer.data, table[["Z", "Y", "X"]].to_numpy(float))
    assert list(layer.features["puncta_id"]) == list(table["puncta_id"])
    assert len(table) < before
    manual = table[table["Source"] == "manual"]
    assert list(manual["puncta_id"]) == [9000] and list(manual[["Z", "Y", "X"]].iloc[0]) == [15.5, 60.25, 70.75]
    assert gui.model.status.startswith(f"Re-detected R2 {CHANNEL} on the original image and warped it: ")
    assert "1 placed by hand kept" in gui.model.status
    assert _states()[R2_PUNCTA] == ("fresh", False)                        # the layer's own save is not an edit


def test_the_export_section_says_when_the_report_is_out_of_date(gui):
    _export_report(gui.reopened)
    gui.coloc.refresh_report_note()
    assert not gui.coloc._report_note.isVisibleTo(gui.coloc._report_note.parentWidget())
    gui.widget._redetect_on_original(gui.viewer, gui.image, _params(threshold_rel=0.6), "replace")
    assert "is out of date" in gui.model.status and "Export it again" in gui.model.status
    assert "is out of date" in gui.coloc._report_note.text()
    assert gui.coloc._report_note.isVisibleTo(gui.coloc._report_note.parentWidget())


def test_the_detect_puncta_button_takes_the_original_image_path(gui, monkeypatch):
    """Through the widget's own function: the existing layer is asked about, then re-detected."""
    asked = []
    monkeypatch.setattr(gui.helpers, "ask_replace_or_merge",
                        lambda parent, name, n, n_manual=0: asked.append(name) or "replace")
    gui.widget._update_detect_on_original(gui.image)
    assert gui.widget._detect_on_original.isEnabled() and gui.widget._detect_on_original.isChecked()
    p = _params(threshold_rel=0.6)
    gui.widget._puncta_widget(image_layer=gui.image, nuclei_layer=None, nuclei_only=True, method=p["method"],
                              threshold=0.6, min_distance=p["min_distance"], sigma=0.0, z_scale=p["z_scale"],
                              use_tophat=False, tophat_radius=15)
    assert asked == [R2_PUNCTA] and not gui.shown
    assert gui.model.status.startswith(f"Re-detected R2 {CHANNEL} on the original image")
    record = __import__("zfisher.core.provenance", fromlist=["latest"]).latest(gui.csv)
    assert record["event"] in ("transformed", "saved") and record.get("detected_in") == "raw"


def test_the_tick_box_follows_the_layer_and_the_saved_registration(gui):
    box = gui.widget._detect_on_original
    raw_layer = gui.model.add_image(np.zeros((2, 4, 4), np.uint16), name=f"R2 - {CHANNEL}")
    gui.widget._update_detect_on_original(raw_layer)
    assert not box.isVisibleTo(box.parentWidget())
    gui.widget._update_detect_on_original(gui.image)
    assert box.isVisibleTo(box.parentWidget()) and box.isEnabled() and box.isChecked()
    saved = session.get_data("registration")
    session.update_data("registration", {})                     # a session made before 2.1.0
    gui.widget._update_detect_on_original(gui.image)
    assert not box.isEnabled() and not box.isChecked() and box.toolTip().startswith("Not available:")
    note = gui.widget._original_note
    assert note.isVisibleTo(note.parentWidget()) and note.text() == stages.NO_SAVED_WARP
    session.update_data("registration", saved)
    gui.widget._update_detect_on_original(gui.image)
    assert box.isEnabled() and box.isChecked() and not note.isVisibleTo(note.parentWidget())


def test_without_the_original_image_nothing_changes(gui, monkeypatch):
    monkeypatch.setattr(gui.popups, "resolve_raw_path", lambda parent, round_id: None)
    before = gui.csv.read_bytes()
    gui.widget._redetect_on_original(gui.viewer, gui.image, _params(), "replace")
    assert gui.csv.read_bytes() == before
    assert gui.model.status == f"R2 {CHANNEL} was not re-detected: its original image was not found."


def test_detecting_on_the_warped_r2_image_asks_first(gui, monkeypatch):
    asked = []
    monkeypatch.setattr(gui.popups, "show_yes_no_popup", lambda parent, title, text: asked.append(text) or False)
    gui.widget._update_detect_on_original(gui.image)
    assert gui.widget._confirm_aligned_r2(gui.viewer, gui.image) is False
    assert "not the method batch uses" in asked[0] and "Tick" in asked[0]
