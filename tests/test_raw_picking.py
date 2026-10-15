"""Picking puncta by hand on a channel's original image, then warping them.

"Pick on Original Image" opens one raw channel and its raw puncta over the
aligned data (the rigid part of the saved registration), hides that channel's
aligned layers, and "Warp Picks into Aligned Space" takes the raw puncta,
picks included, onto the aligned puncta layer. A pick at a known raw voxel
must land where the saved registration puts that voxel.

Headless: a napari ``ViewerModel`` stands in for the viewer; the progress
dialog and pop-ups are stand-ins.

Run:  python -m pytest tests/test_raw_picking.py -q
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
from zfisher.core import io, puncta, registration, session, stages

from test_stages import CHANNEL, _states, finished, reopened  # noqa: F401  (fixtures)


@pytest.fixture
def gui(reopened, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import events, popups, viewer_helpers
    from zfisher.ui.widgets.raw_picking_widget import RawPickingWidget

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

    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    monkeypatch.setattr(popups, "resolve_raw_path", lambda parent, round_id: io.resolve_raw_path(round_id))
    shown = []
    for name in ("show_error_popup", "show_warning_popup"):
        monkeypatch.setattr(popups, name, lambda parent, title, text, _n=name: shown.append((_n, title, text)))

    scale = tuple(session.get_data("canvas_scale"))
    for round_id, prefix in (("R1", constants.ALIGNED_PREFIX), ("R2", constants.WARPED_PREFIX)):
        name = f"{prefix} {round_id} - {CHANNEL}"
        model.add_image(np.zeros((4, 8, 8), np.uint16), name=name, scale=scale)
        csv = constants.puncta_csv_path(reopened.reports, f"{name}{constants.PUNCTA_SUFFIX}")
        viewer_helpers._load_points_layer(model, f"{name}{constants.PUNCTA_SUFFIX}", csv, scale,
                                          {"subtype": "puncta_csv"}, (0, 0, 0))
        events.lock_layer(model.layers[f"{name}{constants.PUNCTA_SUFFIX}"])
        model.layers[name].visible = model.layers[f"{name}{constants.PUNCTA_SUFFIX}"].visible = True
    widget = RawPickingWidget(viewer)
    widget.reset_choices()
    return types.SimpleNamespace(viewer=viewer, model=model, widget=widget, shown=shown, scale=scale,
                                 reports=reopened.reports, out=reopened.out)


def _pick(layer, raw_zyx):
    """Add one punctum by hand at ``raw_zyx``, as the Puncta Editor does."""
    from zfisher.ui import viewer_helpers
    feats = layer.features.copy()
    row = {c: 0 for c in feats.columns}
    row.update(puncta_id=int(puncta.allocate_puncta_ids(layer.name, 1, feats["puncta_id"])[0]),
               Source=constants.PUNCTA_SOURCE_MANUAL)
    feats = pd.concat([feats, pd.DataFrame([row])], ignore_index=True)
    viewer_helpers.set_points_data(layer, np.vstack([layer.data, raw_zyx]), feats)
    return row["puncta_id"]


def test_the_channel_choices_are_the_aligned_puncta(gui):
    assert list(gui.widget.channel.choices) == [f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"]


@pytest.mark.parametrize("round_id", ["R1", "R2"])
def test_a_pick_on_the_original_image_lands_where_the_registration_puts_it(gui, round_id):
    key = f"{round_id} - {CHANNEL}"
    prefix = constants.WARPED_PREFIX if round_id == "R2" else constants.ALIGNED_PREFIX
    gui.widget.channel.value = key
    gui.widget._on_open()
    assert not gui.shown
    raw_points = gui.model.layers[f"{key}{constants.PUNCTA_SUFFIX}"]
    image = gui.model.layers[key]
    scale, translate = stages.raw_overlay(round_id, gui.scale)
    for layer in (raw_points, image):
        np.testing.assert_allclose(layer.scale, scale)
        np.testing.assert_allclose(layer.translate, translate)
    assert not gui.model.layers[f"{prefix} {key}"].visible                       # aligned layers hidden
    assert not gui.model.layers[f"{prefix} {key}{constants.PUNCTA_SUFFIX}"].visible
    n_raw = len(raw_points.data)

    # A nucleus centre of the synthetic round, in that round's raw voxels.
    raw_zyx = np.array([16.0, 43.0, 38.0]) if round_id == "R2" else np.array([16.0, 40.0, 40.0])
    pid = _pick(raw_points, raw_zyx)
    gui.widget._on_warp()

    saved = registration.load_registration()
    if round_id == "R1":
        expected = raw_zyx - saved.canvas_offset
    else:
        expected = registration.transform_points_inverse_bspline(
            (raw_zyx + saved.shift - saved.canvas_offset)[None], saved.bspline)[0]
    aligned = gui.model.layers[f"{prefix} {key}{constants.PUNCTA_SUFFIX}"]
    row = aligned.features.index[aligned.features["puncta_id"] == pid]
    assert len(row) == 1
    np.testing.assert_allclose(aligned.data[row[0]], expected, atol=1e-6)
    assert aligned.features.loc[row[0], "Source"] == constants.PUNCTA_SOURCE_MANUAL
    assert aligned._locked and aligned.visible and gui.model.layers[f"{prefix} {key}"].visible
    assert key not in gui.model.layers and f"{key}{constants.PUNCTA_SUFFIX}" not in gui.model.layers
    raw_csv = constants.puncta_csv_path(gui.reports, stages.raw_puncta_name(round_id, CHANNEL))
    assert len(pd.read_csv(raw_csv)) == n_raw + 1                               # the pick is in the raw CSV
    assert session.get_data("processed_files")[f"{key}{constants.PUNCTA_SUFFIX}"]["space"] == "raw"
    assert _states()[f"{prefix} {key}{constants.PUNCTA_SUFFIX}"] == ("fresh", False)
    assert gui.model.status.startswith(f"Warped {key}: ")


def test_closing_without_warping_keeps_the_picks_and_says_the_channel_is_out_of_date(gui):
    key = f"R2 - {CHANNEL}"
    gui.widget.channel.value = key
    gui.widget.show_mask.value = True
    gui.widget._on_open()
    assert "R2 nuclei (original image)" in gui.model.layers
    _pick(gui.model.layers[f"{key}{constants.PUNCTA_SUFFIX}"], np.array([16.0, 43.0, 38.0]))
    gui.widget._on_close()
    for name in (key, f"{key}{constants.PUNCTA_SUFFIX}", "R2 nuclei (original image)"):
        assert name not in gui.model.layers
    raw_csv = constants.puncta_csv_path(gui.reports, stages.raw_puncta_name("R2", CHANNEL))
    mask = stages.round_mask_path("R2")
    assert raw_csv.exists() and mask.exists()                                   # nothing backed up to .bak
    assert not list(gui.reports.glob("*.bak")) and not list(mask.parent.glob("*.bak"))
    assert (pd.read_csv(raw_csv)["Source"] == constants.PUNCTA_SOURCE_MANUAL).sum() == 1
    assert _states()[f"Warped {key}{constants.PUNCTA_SUFFIX}"][0] == "stale"
    assert "out of date until you warp them" in gui.model.status
    assert gui.model.layers[f"Warped {key}"].visible and gui.widget.open_btn.enabled


def test_a_session_without_a_saved_registration_cannot_open(gui):
    session.update_data("registration", {})
    gui.widget.channel.value = f"R2 - {CHANNEL}"
    gui.widget._on_open()
    assert gui.shown and gui.shown[0][1] == "Cannot Pick on the Original Image"
    assert f"R2 - {CHANNEL}" not in gui.model.layers


def test_autosave_keeps_a_raw_csv_marked_raw(gui):
    """Editing a raw puncta layer used to re-register its CSV without
    ``space='raw'``, so the next reload drew raw puncta over aligned images."""
    from zfisher.ui import events
    key = f"R1 - {CHANNEL}"
    gui.widget.channel.value = key
    gui.widget._on_open()
    events.save_puncta_layer(gui.model.layers[f"{key}{constants.PUNCTA_SUFFIX}"])
    entry = session.get_data("processed_files")[f"{key}{constants.PUNCTA_SUFFIX}"]
    assert entry["space"] == "raw" and entry["round"] == "R1"


def test_a_pick_outside_every_nucleus_is_kept_when_automatic_ones_are_dropped(gui):
    """Batch removes puncta outside nuclei when warping; a punctum placed by
    hand is curation and stays."""
    key = f"R1 - {CHANNEL}"
    gui.widget.channel.value = key
    gui.widget._on_open()
    raw_points = gui.model.layers[f"{key}{constants.PUNCTA_SUFFIX}"]
    pid = _pick(raw_points, np.array([16.0, 90.0, 140.0]))          # between nuclei
    gui.widget._on_warp()
    aligned = gui.model.layers[f"Aligned {key}{constants.PUNCTA_SUFFIX}"]
    row = aligned.features[aligned.features["puncta_id"] == pid]
    assert len(row) == 1 and row["Nucleus_ID"].iloc[0] == 0
    automatic = aligned.features[aligned.features["Source"] == constants.PUNCTA_SOURCE_AUTO]
    assert (automatic["Nucleus_ID"] > 0).all()                       # still filtered, as batch made them
