"""The GUI's automated registration writes what batch writes (plan item B11).

On a copy of a small batch session (``test_stages``), the GUI's layers are
rebuilt from the raw files (images, round masks, centroids, raw puncta) and
the Automated Registration & Warping widget is run with batch's settings.
Its aligned images, aligned masks, consensus mask and aligned puncta CSVs
must equal batch's: the images are no longer cast to float32, the rounds'
nuclear channel names are passed, and puncta are warped from the raw CSVs by
the same function batch uses.

Run:  python -m pytest tests/test_gui_on_stages.py -q
"""
import contextlib
import logging
import os
import shutil
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest
import tifffile

from zfisher import constants
from zfisher.core import io, session, stages

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


class _Dialog(contextlib.nullcontext):
    def __init__(self, *a, **k):
        super().__init__(self)

    def update_progress(self, *a):
        pass

    def freeze_canvas(self):
        pass


@pytest.fixture
def gui(finished, tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import popups, viewer_helpers
    from zfisher.ui.widgets import automated_preprocessing_widget as apw

    batch = tmp_path / "batch"
    shutil.copytree(finished / "out", batch)
    out = tmp_path / "gui"
    shutil.copytree(finished / "out", out)
    for sub in (constants.ALIGNED_DIR,):
        shutil.rmtree(out / sub)
    for f in (out / constants.REPORTS_DIR).glob("Aligned_*"):
        f.unlink()
    for f in (out / constants.REPORTS_DIR).glob("Warped_*"):
        f.unlink()
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    session.load_session_file(out / constants.SESSIONS_DIR / constants.SESSION_FILENAME)

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
    errors = []
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: errors.append((title, text)))

    seg = out / constants.SEGMENTATION_DIR
    for rnd in ("R1", "R2"):
        raw = io.load_image_session(finished / f"{rnd}.ome.tif")
        for i, ch in enumerate(raw.channels):
            model.add_image(raw.data[:, i], name=f"{rnd} - {ch}", scale=raw.voxels)
        model.add_labels(io.read_label_tif(io.find_mask(seg, f"{rnd} - DAPI_masks")), name=f"{rnd} - DAPI_masks",
                         scale=raw.voxels)
        model.add_points(np.load(seg / f"{rnd} - DAPI_centroids.npy"), name=f"{rnd} - DAPI_centroids",
                         scale=raw.voxels)
        csv = constants.puncta_csv_path(out / constants.REPORTS_DIR, stages.raw_puncta_name(rnd, CHANNEL))
        viewer_helpers._load_points_layer(model, f"{rnd} - {CHANNEL}_puncta", csv, raw.voxels,
                                          {"subtype": "puncta_csv"}, (0, 0, 0))
    yield types.SimpleNamespace(widget=apw, model=model, out=out, batch=batch, errors=errors)
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_automated_registration_in_the_gui_writes_batchs_files(gui):
    gui.widget._automated_preprocessing_magic_widget(
        r1_dapi_layer=gui.model.layers["R1 - DAPI"], r2_dapi_layer=gui.model.layers["R2 - DAPI"],
        max_distance=0, apply_warp=True, match_nuclei=True, overlap_method="Intersection", match_threshold=0,
        remove_outliers=True, show_checkerboard=True, show_deformation=True, hide_raw=False)
    assert gui.errors == []

    aligned = sorted(p.name for p in (gui.batch / constants.ALIGNED_DIR).glob("*.tif")
                     if constants.WARPED_CHECKERBOARD_NAME not in p.name)
    assert aligned and aligned == sorted(p.name for p in (gui.out / constants.ALIGNED_DIR).glob("*.tif")
                                         if constants.WARPED_CHECKERBOARD_NAME not in p.name)
    for name in aligned:
        a = tifffile.imread(gui.batch / constants.ALIGNED_DIR / name)
        b = tifffile.imread(gui.out / constants.ALIGNED_DIR / name)
        assert a.dtype == b.dtype, name
        np.testing.assert_array_equal(a, b, err_msg=name)
    seg_b, seg_g = gui.batch / constants.SEGMENTATION_DIR, gui.out / constants.SEGMENTATION_DIR
    np.testing.assert_array_equal(io.read_label_tif(io.find_mask(seg_b, constants.CONSENSUS_MASKS_NAME)),
                                  io.read_label_tif(io.find_mask(seg_g, constants.CONSENSUS_MASKS_NAME)))
    for name in (f"Aligned_R1_-_{CHANNEL}_puncta.csv", f"Warped_R2_-_{CHANNEL}_puncta.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(gui.batch / constants.REPORTS_DIR / name),
                                      pd.read_csv(gui.out / constants.REPORTS_DIR / name), check_exact=True)
    layer = gui.model.layers[f"Warped R2 - {CHANNEL}_puncta"]
    table = pd.read_csv(gui.out / constants.REPORTS_DIR / f"Warped_R2_-_{CHANNEL}_puncta.csv")
    np.testing.assert_allclose(layer.data, table[["Z", "Y", "X"]].to_numpy(float))
    assert f"R2 - {CHANNEL}_puncta" not in gui.model.layers                  # the raw layer retired


def test_the_manual_canvas_step_warps_the_raw_csvs_without_nuclei_yet(gui):
    from zfisher.ui.widgets import canvas_widget
    canvas_widget._canvas_widget(apply_warp=True, show_checkerboard=False, show_deformation=False, hide_raw=False)

    for name in ("Aligned_R1_DAPI.tif", "Warped_R2_FITC.tif", "Warped_R2_DAPI_masks.tif"):
        np.testing.assert_array_equal(tifffile.imread(gui.batch / constants.ALIGNED_DIR / name),
                                      tifffile.imread(gui.out / constants.ALIGNED_DIR / name), err_msg=name)
    table = pd.read_csv(gui.out / constants.REPORTS_DIR / f"Warped_R2_-_{CHANNEL}_puncta.csv")
    assert len(table) and set(table["Nucleus_ID"]) == {0}                   # Match Nuclei assigns them
    batch = pd.read_csv(gui.batch / constants.REPORTS_DIR / f"Warped_R2_-_{CHANNEL}_puncta.csv")
    kept = table.set_index("puncta_id").loc[batch["puncta_id"]]             # batch dropped extranuclear ones
    np.testing.assert_allclose(kept[["Z", "Y", "X"]].to_numpy(float), batch[["Z", "Y", "X"]].to_numpy(float))
    assert f"Warped R2 - {CHANNEL}_puncta" in gui.model.layers and f"R2 - {CHANNEL}_puncta" not in gui.model.layers
