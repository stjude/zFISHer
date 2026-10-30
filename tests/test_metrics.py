"""Quality-control metrics from a session's files (plan item C2).

On a small batch session (``test_stages``): the registration's quality with
its RMSD in µm, per-channel counts, SNR quantiles and the fractions outside
nuclei, at low SNR and placed by hand, nucleus counts and volumes, saturated
voxels and how each channel was detected; planted changes show up.

Run:  python -m pytest tests/test_metrics.py -q
"""
import logging
import shutil

import numpy as np
import pandas as pd
import pytest

from zfisher import cli, constants
from zfisher.core import api, io, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


@pytest.fixture
def folder(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    yield out
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_metrics_from_a_batch_session(folder):
    m = api.get_metrics(api.open_session(folder), include_saturation=True)
    assert m.registration["method"] == "ransac" and m.registration["rmsd_um"] is not None
    assert 0 < m.registration["inlier_fraction"] <= 1
    mask = io.read_label_tif(io.find_mask(folder / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME))
    assert m.nuclei["n_nuclei"] == len(np.unique(mask)) - 1 and m.nuclei["volume_um3"]["p50"] > 0
    assert set(m.channels) == {f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"}
    r1 = m.channels[f"R1 - {CHANNEL}"]
    csv = pd.read_csv(folder / constants.REPORTS_DIR / f"Aligned_R1_-_{CHANNEL}_puncta.csv")
    assert r1["n_puncta"] == len(csv) and r1["snr"]["p50"] == pytest.approx(float(np.median(csv["SNR"])))
    assert r1["extranuclear_fraction"] == 0.0 and r1["manual_fraction"] == 0.0     # batch removed them
    assert r1["detected_on"] == "raw" and m.detection_mix == {"raw": 2}
    assert r1["puncta_per_nucleus"] == pytest.approx(len(csv) / m.nuclei["n_nuclei"])
    assert set(m.saturation) >= {f"Aligned R1 - {CHANNEL}"} and all(0 <= v <= 1 for v in m.saturation.values())


def test_planted_changes_show_up(folder):
    csv = folder / constants.REPORTS_DIR / f"Warped_R2_-_{CHANNEL}_puncta.csv"
    table = pd.read_csv(csv)
    extra = table.iloc[:2].copy()
    extra["Source"] = constants.PUNCTA_SOURCE_MANUAL
    extra[["Z", "Y", "X"]] = [[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]]               # outside every nucleus
    extra["puncta_id"] = [99998, 99999]
    pd.concat([table, extra], ignore_index=True).to_csv(csv, index=False)
    r2 = api.get_metrics(api.open_session(folder)).channels[f"R2 - {CHANNEL}"]
    assert r2["n_puncta"] == len(table) + 2
    assert r2["manual_fraction"] == pytest.approx(2 / (len(table) + 2))
    assert r2["extranuclear_fraction"] == pytest.approx(2 / (len(table) + 2))


def test_the_cli_prints_the_metrics(folder, capsys):
    assert cli.main(["metrics", str(folder)]) == 0
    text = capsys.readouterr().out
    assert "registration ransac" in text and f"R1 - {CHANNEL}:" in text and "detected on raw" in text
