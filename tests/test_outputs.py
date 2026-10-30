"""Every file is written whole or not at all (``core.outputs``).

A writer that fails part-way, or a process killed mid-write, leaves the
previous file as it was and no temporary file; writers that go by extension
(an ``.ome.tif`` gets OME metadata, an ``.xlsx`` the Excel engine) behave as
they would on the real name; ``.npy`` bytes are what ``np.save`` writes.

Run:  python -m pytest tests/test_outputs.py -q
"""
import io as _io

import numpy as np
import pandas as pd
import pytest
import tifffile

from zfisher.core import outputs, puncta


def _leftovers(folder):
    return [p.name for p in folder.iterdir() if ".tmp." in p.name]


def test_a_failed_write_leaves_the_previous_file(tmp_path):
    path = tmp_path / "table.csv"
    outputs.write_csv(pd.DataFrame({"a": [1, 2]}), path)
    before = path.read_bytes()
    with pytest.raises(RuntimeError):
        with outputs.atomic_write(path) as tmp:
            tmp.write_text("a\n9\n")                              # half written...
            raise RuntimeError("disk full")                        # ...then the write fails
    assert path.read_bytes() == before and _leftovers(tmp_path) == []


def test_an_ome_tiff_stays_ome_and_npy_bytes_match_np_save(tmp_path):
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    path = outputs.write_tiff(tmp_path / "R1_converted.ome.tif", data, photometric="minisblack",
                              metadata={"axes": "ZCYX", "Channel": {"Name": ["DAPI", "FITC", "Cy5"]}})
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome and tif.series[0].axes == "ZCYX" and "FITC" in tif.ome_metadata
    arr = np.random.default_rng(0).normal(size=(7, 3))
    buffer = _io.BytesIO()
    np.save(buffer, arr)
    assert outputs.write_npy(tmp_path / "c.npy", arr).read_bytes() == buffer.getvalue()
    assert _leftovers(tmp_path) == []


def test_an_excel_file_is_swapped_in_only_when_complete(tmp_path):
    path = tmp_path / "report.xlsx"
    with outputs.excel_writer(path) as writer:
        pd.DataFrame({"x": [1]}).to_excel(writer, sheet_name="Stats", index=False)
    assert pd.read_excel(path, sheet_name="Stats")["x"].tolist() == [1]
    with pytest.raises(ValueError):
        with outputs.excel_writer(path) as writer:
            pd.DataFrame({"x": [2]}).to_excel(writer, sheet_name="Stats", index=False)
            raise ValueError("analysis failed")
    assert pd.read_excel(path, sheet_name="Stats")["x"].tolist() == [1] and _leftovers(tmp_path) == []


def test_a_puncta_file_interrupted_mid_write_keeps_its_contents(tmp_path, monkeypatch):
    csv = tmp_path / "Aligned_R1_-_FITC_puncta.csv"
    features = pd.DataFrame({"puncta_id": [0, 1], "Nucleus_ID": [3, 4], "Source": ["manual", "auto"]})
    puncta.write_puncta_csv(csv, np.ones((2, 3)), features)
    before = csv.read_bytes()

    def killed(self, path, **kwargs):
        open(path, "w").write("puncta_id,Z\n0,")                  # a truncated write
        raise KeyboardInterrupt
    monkeypatch.setattr(pd.DataFrame, "to_csv", killed)
    with pytest.raises(KeyboardInterrupt):
        puncta.write_puncta_csv(csv, np.zeros((5, 3)), pd.DataFrame({"puncta_id": range(5)}))
    assert csv.read_bytes() == before and _leftovers(tmp_path) == []
