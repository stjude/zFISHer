"""Small synthetic two-round datasets for tests that run the batch pipeline.

Each round is a (Z, C, Y, X) OME-TIFF with a DAPI channel of five nuclei and a
FITC channel of four spots per nucleus; R2 is R1 shifted by (3, -2) px in Y, X.
"""
import logging
from pathlib import Path

import numpy as np


def write_rounds(root):
    """Write ``R1.ome.tif`` and ``R2.ome.tif`` into ``root``."""
    import tifffile
    rng = np.random.default_rng(0)
    Z, Y, X = 32, 192, 192

    def make_round(shift):
        dapi = np.zeros((Z, Y, X), np.float32)
        fitc = np.zeros_like(dapi)
        zz, yy, xx = np.ogrid[:Z, :Y, :X]
        spots = np.random.default_rng(1)
        for cz, cy, cx in [(16, 40, 40), (16, 40, 140), (16, 140, 40), (16, 140, 140), (16, 90, 90)]:
            cy, cx = cy + shift[0], cx + shift[1]
            dapi[((zz - cz) / 8.0) ** 2 + ((yy - cy) / 16.0) ** 2 + ((xx - cx) / 16.0) ** 2 <= 1] = 1000
            for _ in range(4):
                pz, py, px = cz + spots.integers(-2, 3), cy + spots.integers(-8, 9), cx + spots.integers(-8, 9)
                fitc[pz, py - 1:py + 2, px - 1:px + 2] = 3000
        dapi += rng.normal(50, 5, dapi.shape)
        fitc += rng.normal(50, 5, fitc.shape)
        return np.stack([dapi, fitc], axis=1).clip(0).astype(np.uint16)

    for name, shift in (("R1", (0, 0)), ("R2", (3, -2))):
        tifffile.imwrite(root / f"{name}.ome.tif", make_round(shift), photometric="minisblack",
                         metadata={"axes": "ZCYX", "Channel": {"Name": ["DAPI", "FITC"]},
                                   "PhysicalSizeX": 0.1, "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.3})


def run_batch(root, out):
    """Run the batch pipeline on the rounds in ``root`` into ``out``.

    napari's pytest plugin installs a log record class that turns numpy
    arguments into strings, which breaks "%d" log calls given numpy integers.
    The app uses Python's own records, so the pipeline runs with those here.
    The session log opens DEBUG on these loggers and leaves it on; they are
    restored afterwards so later tests do not log at DEBUG."""
    from zfisher.core import pipeline
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)
    try:
        pipeline.run_full_zfisher_pipeline(Path(root) / "R1.ome.tif", Path(root) / "R2.ome.tif", out,
                                           seg_method="Classical")
    finally:
        logging.setLogRecordFactory(factory)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
