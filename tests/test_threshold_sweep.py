"""Choosing the sensitivity from where the puncta count levels off (plan item C6).

``threshold_sweep`` counts the puncta detection would keep at each of a range
of sensitivities; ``find_plateau`` finds the widest stretch where the count
barely changes and suggests its middle. For Local Maxima one detection serves
the whole curve, and gives the counts detecting at each step would. The curve
is stored in the session; the API, the CLI and Puncta Detection's Suggest
Sensitivity button use it.

Run:  python -m pytest tests/test_threshold_sweep.py -q
"""
import json
import logging
import os
import shutil
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from zfisher import cli, constants
from zfisher.core import api, puncta, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


def _spots(n=30, seed=4):
    rng = np.random.default_rng(seed)
    image = rng.normal(100, 8, (12, 80, 80))
    image -= image.min()
    for _ in range(n):
        z, y, x = rng.integers(2, 10), rng.integers(4, 76), rng.integers(4, 76)
        image[z - 1:z + 2, y - 1:y + 2, x - 1:x + 2] += rng.uniform(900, 1100)
    return image


def test_find_plateau():
    t = np.linspace(0.1, 1.0, 10)
    p = puncta.find_plateau(t, [400, 200, 80, 41, 40, 40, 39, 38, 20, 5])
    assert (p["start"], p["end"]) == pytest.approx((0.4, 0.8)) and p["suggested"] == pytest.approx(0.6)
    assert puncta.find_plateau(t, [0] * 10) is None
    assert puncta.find_plateau(t, [1000, 500, 250, 120, 60, 30, 15, 8, 4, 2]) is None


def test_the_local_maxima_shortcut_counts_what_detection_keeps():
    image = _spots()
    mask = np.ones(image.shape, np.int32)
    mask[:, :, :20] = 0                                           # some spots outside "nuclei"
    params = {"method": "Local Maxima", "min_distance": 2, "nuclei_only": True, "z_edge_exclude": 2}
    thresholds = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    _, counts = puncta.threshold_sweep(image, mask, (0.3, 0.1, 0.1), params, thresholds)
    direct = [len(puncta.process_puncta_detection(image, mask_data=mask, voxels=(0.3, 0.1, 0.1),
                                                  params=dict(params, threshold_rel=t))) for t in thresholds]
    assert counts.tolist() == direct


def test_the_plateau_finds_the_planted_spots():
    image = _spots(n=30)
    thresholds, counts = puncta.threshold_sweep(image, None, (0.3, 0.1, 0.1),
                                                {"method": "Local Maxima", "min_distance": 2, "nuclei_only": False})
    plateau = puncta.find_plateau(thresholds, counts)
    assert plateau is not None and plateau["count"] <= 30 and plateau["count"] >= 28   # a spot pair may merge
    assert plateau["width"] >= 0.2
    _, log_counts = puncta.threshold_sweep(image, None, (0.3, 0.1, 0.1),
                                           {"method": "Laplacian of Gaussian", "sigma": 1.0, "nuclei_only": False},
                                           [0.1, 0.3, 0.5])
    assert len(log_counts) == 3 and log_counts[0] >= log_counts[-1]


@pytest.fixture
def folder(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    yield out
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_a_channel_sweep_is_stored_and_reported(folder, capsys):
    s = api.open_session(folder, read_only=False)
    result = api.threshold_sweep(s, "R1", CHANNEL)
    assert result.channel == f"R1 - {CHANNEL}" and len(result.counts) == 25
    assert result.counts == sorted(result.counts, reverse=True)
    saved = json.loads((folder / constants.SESSIONS_DIR / "zfisher_session_2.json").read_text())
    assert saved["threshold_sweeps"][f"R1 - {CHANNEL}"]["counts"] == result.counts
    assert cli.main(["sweep", str(folder), f"R2 - {CHANNEL}"]) == 0
    text = capsys.readouterr().out
    assert "  0.020: " in text and ("suggested sensitivity" in text or "do not level off" in text)


def test_the_gui_button_shows_the_plateau(folder, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import contextlib
    import napari
    from napari.components import ViewerModel
    from zfisher.core import io
    from zfisher.ui import popups
    from zfisher.ui.widgets import puncta_widget as pw

    class _Dialog(contextlib.nullcontext):
        def __init__(self, *a, **k):
            super().__init__(self)

        def update_progress(self, *a):
            pass

    model = ViewerModel()
    monkeypatch.setattr(napari, "current_viewer", lambda: types.SimpleNamespace(
        status="", window=types.SimpleNamespace(_qt_window=None), layers=model.layers))
    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    session.load_session_file(folder / constants.SESSIONS_DIR / constants.SESSION_FILENAME)
    monkeypatch.setattr(io, "resolve_raw_path", lambda round_id, **k: folder.parent / "rounds" / f"{round_id}.ome.tif")
    rounds = folder.parent / "rounds"
    rounds.mkdir()
    from synthetic_rounds import write_rounds
    write_rounds(rounds)
    layer = model.add_image(np.zeros((2, 4, 4), np.uint16), name=f"R1 - {CHANNEL}")
    pw._puncta_widget.image_layer.choices = [layer]
    pw._puncta_widget.image_layer.value = layer
    pw._suggest_sensitivity()
    text = pw._sweep_note.text()
    assert pw._sweep_note.isVisibleTo(pw._sweep_note.parentWidget())
    if "do not level off" not in text:
        assert "sensitivity set to" in text and pw._puncta_widget.threshold.value == pytest.approx(
            float(text.split("sensitivity set to ")[1].split(",")[0]))
