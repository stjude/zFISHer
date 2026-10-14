"""Batch works when the two rounds name their nuclear channel differently.

Segmentation saved both rounds' masks under R1's nuclear channel
(``R2 - DAPI_masks``), while canvas generation, the consensus step and puncta
detection look R2's up under R2's own (``R2 - Hoechst_masks``). A dataset whose
R2 nuclear channel had another name therefore failed with "Consensus failed:
aligned masks not found". Each round's mask is now named after its own nuclear
channel, as GUI segmentation names it.

The batch test runs the whole pipeline on small synthetic rounds (about 2 s).

Run:  python -m pytest tests/test_nuclear_channel_names.py -q
"""
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from zfisher import constants
from zfisher.core import segmentation, session


def _round(path, nuclear, shift):
    """Five nuclei in the nuclear channel and four spots in each in FITC."""
    import tifffile
    Z, Y, X = 32, 192, 192
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    nuclei, spots = np.zeros((Z, Y, X), np.float32), np.zeros((Z, Y, X), np.float32)
    rng = np.random.default_rng(1)
    for cy, cx in [(40, 40), (40, 140), (140, 40), (140, 140), (90, 90)]:
        cy, cx = cy + shift[0], cx + shift[1]
        nuclei[((zz - 16) / 8.0) ** 2 + ((yy - cy) / 16.0) ** 2 + ((xx - cx) / 16.0) ** 2 <= 1] = 1000
        for _ in range(4):
            pz, py, px = 16 + rng.integers(-2, 3), cy + rng.integers(-8, 9), cx + rng.integers(-8, 9)
            spots[pz, py - 1:py + 2, px - 1:px + 2] = 3000
    noise = np.random.default_rng(0).normal(50, 5, (2, Z, Y, X))
    data = np.stack([nuclei + noise[0], spots + noise[1]], axis=1).clip(0).astype(np.uint16)
    tifffile.imwrite(path, data, photometric="minisblack",
                     metadata={"axes": "ZCYX", "Channel": {"Name": [nuclear, "FITC"]},
                               "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.3})


def test_each_round_mask_is_named_after_its_own_nuclear_channel(monkeypatch, tmp_path):
    data = {"output_dir": str(tmp_path), "processed_files": {}}
    monkeypatch.setattr(session, "get_nuclear_channel", lambda: "DAPI")
    monkeypatch.setattr(session, "set_processed_file",
                        lambda name, path, layer_type, metadata=None: data["processed_files"].__setitem__(name, path))
    image = np.zeros((40, 128, 128), np.uint16)
    yy, xx = np.mgrid[:128, :128]
    image[:, (yy - 64) ** 2 + (xx - 64) ** 2 < 30 ** 2] = 1000
    segmentation.process_session_dapi(image, image, output_dir=tmp_path, voxel_spacing=(0.3, 0.1, 0.1),
                                      nuclear_channels={"R2": "Hoechst"})
    assert {"R1 - DAPI_masks", "R2 - Hoechst_masks"} <= set(data["processed_files"])
    seg = tmp_path / constants.SEGMENTATION_DIR
    assert (seg / "R1 - DAPI_masks.ome.tif").exists() and (seg / "R2 - Hoechst_masks.ome.tif").exists()
    assert (seg / "R2 - Hoechst_centroids.npy").exists()


def test_batch_with_differently_named_nuclear_channels_finishes():
    pytest.importorskip("SimpleITK")
    from zfisher.core import pipeline
    import pandas as pd
    root = Path(tempfile.mkdtemp())
    _round(root / "R1.ome.tif", "DAPI", (0, 0))
    _round(root / "R2.ome.tif", "Hoechst", (3, -2))
    out = root / "out"
    # See tests/test_raw_puncta.py: the app's own log records, and the
    # session log's DEBUG levels restored afterwards.
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)
    try:
        pipeline.run_full_zfisher_pipeline(root / "R1.ome.tif", root / "R2.ome.tif", out, seg_method="Classical",
                                           r1_nuclear_channel="DAPI", r2_nuclear_channel="Hoechst")
        seg = out / constants.SEGMENTATION_DIR
        assert (seg / "R2 - Hoechst_masks.ome.tif").exists()
        assert (seg / "Consensus_Nuclei_masks.ome.tif").exists()
        warped = out / constants.REPORTS_DIR / "Warped_R2_-_FITC_puncta.csv"
        assert len(pd.read_csv(warped)) > 0
        # R2's puncta were filtered by R2's own mask: every one lies in a nucleus.
        raw = pd.read_csv(out / constants.REPORTS_DIR / "R2_-_FITC_puncta.csv")
        assert len(raw) > 0 and (raw["Nucleus_ID"] > 0).all()
    finally:
        logging.setLogRecordFactory(factory)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        session.clear_session()
        shutil.rmtree(root, ignore_errors=True)
