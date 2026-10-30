"""The Detect button reads the raw image and the round's nuclei mask from disk.

On a raw image layer (``R1 - FITC``), Detect Puncta now runs
``stages.detect_channel``, the function batch uses, on the round's image file
and nuclei mask file, instead of on the arrays in the window. GUI and batch
give the same puncta for the same settings, and the provenance record names the
files that were read. Unsaved puncta and mask edits are saved first.

Headless: a napari ``ViewerModel`` behind the attributes the widget uses, on a
small batch session (``test_stages``).

Run:  python -m pytest tests/test_detect_from_disk.py -q
"""
import contextlib
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import io, provenance, session, stages

from test_stages import CHANNEL, finished, reopened  # noqa: F401  (fixtures)

R1_IMAGE = f"R1 - {CHANNEL}"
R1_PUNCTA = stages.raw_puncta_name("R1", CHANNEL)


class _Dialog(contextlib.nullcontext):
    def __init__(self, *a, **k):
        super().__init__(self)

    def update_progress(self, *a):
        pass

    def freeze_canvas(self):
        pass


@pytest.fixture
def gui(reopened, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import popups, viewer_helpers
    from zfisher.ui.widgets import puncta_widget

    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

        def __setattr__(self, name, value):
            setattr(model, name, value)

    viewer = _Viewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    monkeypatch.setattr(popups, "resolve_raw_path", lambda parent, round_id: io.resolve_raw_path(round_id))
    monkeypatch.setattr(viewer_helpers, "ask_replace_or_merge",
                        lambda parent, name, n, n_manual=0: "replace")
    voxels = io.image_identity(io.resolve_raw_path("R1"))["voxels"]
    # Zeros in the window: what is detected can only come from the file.
    image = model.add_image(np.zeros((4, 8, 8), np.uint16), name=R1_IMAGE, scale=voxels)
    batch = pd.read_csv(constants.puncta_csv_path(reopened.reports, R1_PUNCTA))
    yield types.SimpleNamespace(viewer=viewer, model=model, image=image, widget=puncta_widget,
                                reports=reopened.reports, batch=batch, helpers=viewer_helpers)


def _detect(gui, nuclei_layer=None, **changes):
    """Press Detect Puncta with the settings batch recorded, every one of them."""
    p = dict(stages.recorded_detection_parameters("R1", CHANNEL), **changes)
    gui.widget._puncta_widget(image_layer=gui.image, nuclei_layer=nuclei_layer, nuclei_only=p["nuclei_only"],
                              method=p["method"], threshold=p["threshold_rel"], min_distance=p["min_distance"],
                              sigma=p["sigma"], z_scale=p["z_scale"], use_tophat=p["use_tophat"],
                              tophat_radius=p["tophat_radius"])
    return pd.read_csv(constants.puncta_csv_path(gui.reports, R1_PUNCTA))


def test_detect_on_a_raw_layer_gives_batchs_puncta_from_the_files(gui):
    table = _detect(gui)

    cols = ["Z", "Y", "X", "Nucleus_ID", "Intensity", "SNR"]
    np.testing.assert_allclose(table[cols].to_numpy(float), gui.batch[cols].to_numpy(float))
    layer = gui.model.layers[R1_PUNCTA]
    np.testing.assert_allclose(layer.data, table[["Z", "Y", "X"]].to_numpy(float))
    assert list(layer.features["puncta_id"]) == list(table["puncta_id"])
    record = provenance.latest(constants.puncta_csv_path(gui.reports, R1_PUNCTA))
    assert record["event"] == "detected" and record["run_by"] == "gui" and record["detected_in"] == "raw"
    assert record["source_image"]["path"].endswith("R1.ome.tif")
    assert record["mask"]["path"].endswith("R1 - DAPI_masks.ome.tif")
    assert gui.model.status == f"Found {len(table)} spots in {R1_IMAGE}."


def test_a_mask_edited_in_the_gui_is_still_the_rounds_mask(gui):
    info = dict(session.get_data("processed_files")["R1 - DAPI_masks"], subtype="edited_mask")
    session.set_processed_file("R1 - DAPI_masks", info.pop("path"), info.pop("type"), metadata=info)
    assert stages.round_mask_path("R1") is not None
    assert len(_detect(gui)) == len(gui.batch)


def test_a_point_placed_by_hand_and_not_yet_saved_is_kept(gui):
    csv = constants.puncta_csv_path(gui.reports, R1_PUNCTA)
    gui.helpers._load_points_layer(gui.model, R1_PUNCTA, csv, gui.image.scale, {"subtype": "puncta_csv"}, (0, 0, 0))
    layer = gui.model.layers[R1_PUNCTA]
    features = layer.features.copy()
    extra = features.iloc[[0]].assign(puncta_id=99999, Source=constants.PUNCTA_SOURCE_MANUAL)
    gui.helpers.set_points_data(layer, np.vstack([layer.data, [[1.0, 30.0, 30.0]]]),
                                pd.concat([features, extra], ignore_index=True))

    table = _detect(gui)                                    # Replace keeps puncta placed by hand

    manual = table[table["Source"] == constants.PUNCTA_SOURCE_MANUAL]
    assert list(manual["puncta_id"]) == [99999]
    assert list(manual[["Z", "Y", "X"]].iloc[0]) == [1.0, 30.0, 30.0]


def test_another_rounds_mask_in_the_widget_is_named_in_the_status(gui):
    other = gui.model.add_labels(np.zeros((4, 8, 8), np.int32), name="R2 - DAPI_masks")
    table = _detect(gui, nuclei_layer=other)
    assert len(table) == len(gui.batch)
    assert gui.model.status.endswith("Kept inside R1's own nuclei mask, not R2 - DAPI_masks.")
