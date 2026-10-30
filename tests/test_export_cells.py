"""One OME-TIFF per nucleus (``core.cells``, ``zfisher-cli export-cells``).

On a small batch session (``test_stages``): one file per consensus nucleus,
every aligned channel cropped exactly from the aligned image plus that
nucleus's mask, the physical voxel size and channel names in the OME header,
and the nucleus's puncta in the crop's coordinates; a batch root is walked;
nothing in the session changes.

Run:  python -m pytest tests/test_export_cells.py -q
"""
import logging
import shutil

import numpy as np
import pandas as pd
import pytest
import tifffile

from zfisher import cli, constants
from zfisher.core import api, cells, io, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


@pytest.fixture
def batch_root(finished, tmp_path):
    shutil.copytree(finished / "out", tmp_path / "ds1")
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    yield tmp_path
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_one_file_per_nucleus_with_exact_crops_and_scale(batch_root):
    out = batch_root / "ds1"
    sessions_before = sorted((out / constants.SESSIONS_DIR).iterdir())
    s = api.open_session(out)
    result = api.export_cells(s, padding_um=1.0)
    mask = io.read_label_tif(io.find_mask(out / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME))
    labels = [int(i) for i in np.unique(mask) if i]
    assert result.n_cells == len(labels) and sorted((out / cells.CELLS_DIR).glob("*.ome.tif"))
    assert sorted((out / constants.SESSIONS_DIR).iterdir()) == sessions_before     # the session is untouched

    first = labels[0]
    with session.use(s):
        channels = cells.aligned_channels()
    with tifffile.TiffFile(out / cells.CELLS_DIR / f"nucleus_{first:05d}.ome.tif") as tif:
        data = tif.series[0].asarray()
        ome = tif.ome_metadata
    assert data.ndim == 4 and data.shape[1] == len(channels) + 1                  # Z, C, Y, X
    assert 'PhysicalSizeZ="0.3"' in ome and 'PhysicalSizeX="0.1"' in ome and "Nucleus mask" in ome
    assert all(f'Name="{name}"' in ome for name, _ in channels)
    zz, yy, xx = np.nonzero(mask == first)
    margin = [int(np.ceil(1.0 / s_)) for s_ in (0.3, 0.1, 0.1)]
    origin = [max(0, int(a.min()) - m) for a, m in zip((zz, yy, xx), margin)]
    box = tuple(slice(o, o + n) for o, n in zip(origin, (data.shape[0], data.shape[2], data.shape[3])))
    name, path = channels[0]
    np.testing.assert_array_equal(data[:, 0], tifffile.imread(path)[box])
    np.testing.assert_array_equal(data[:, -1], (mask[box] == first).astype(data.dtype))

    table = pd.read_csv(out / cells.CELLS_DIR / f"nucleus_{first:05d}_puncta.csv")
    assert set(table["layer"]) <= {f"Aligned R1 - {CHANNEL}_puncta", f"Warped R2 - {CHANNEL}_puncta"}
    original = pd.read_csv(out / constants.REPORTS_DIR / f"Aligned_R1_-_{CHANNEL}_puncta.csv").set_index("puncta_id")
    row = table[table["layer"] == f"Aligned R1 - {CHANNEL}_puncta"].iloc[0]
    np.testing.assert_allclose(row[["Z", "Y", "X"]].to_numpy(float) + origin,
                               original.loc[row["puncta_id"], ["Z", "Y", "X"]].to_numpy(float))


def test_the_cli_walks_a_batch_root_and_takes_chosen_nuclei(batch_root, capsys):
    out = batch_root / "ds1"
    mask = io.read_label_tif(io.find_mask(out / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME))
    label = int(np.unique(mask)[-1])
    assert cli.main(["export-cells", str(batch_root), "--id", str(label), "--no-puncta"]) == 0
    assert "ds1: 1 nuclei" in capsys.readouterr().out
    assert [p.name for p in (out / cells.CELLS_DIR).iterdir()] == [f"nucleus_{label:05d}.ome.tif"]
    assert cli.main(["export-cells", str(batch_root / "nothing-here")]) == 1
