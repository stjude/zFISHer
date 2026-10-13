"""Reading one raw channel, and finding a round's raw image again.

Re-detecting one channel of a reopened session needs that channel's raw image,
not the whole round. ``io.read_channel`` reads one channel without loading the
others and must give exactly what the full loader gives for that channel.
``io.resolve_raw_path`` finds the round's image: the ``input/`` copy, the stored
path, then a file the user points to, checked against the identity the session
recorded; the answer is remembered, and with nobody to ask it fails naming the
round and the file.

ND2 reading is checked against a stand-in for ``nd2.ND2File``, since no ND2
file can be written here.

Run:  python -m pytest tests/test_raw_channel.py -q
"""
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tifffile

pytest.importorskip("nd2")

from zfisher import constants
from zfisher.core import io


def _write(path, arr, **kw):
    tifffile.imwrite(path, arr, photometric="minisblack", **kw)
    return path


@pytest.fixture
def rounds(tmp_path):
    """Round images in the layouts the loader accepts."""
    rng = np.random.default_rng(0)
    ome = {"Channel": {"Name": ["DAPI", "FITC", "Cy5"]}, "PhysicalSizeX": 0.1,
           "PhysicalSizeY": 0.1, "PhysicalSizeZ": 0.3}
    return {
        "ome_zcyx": _write(tmp_path / "a.ome.tif", rng.integers(0, 999, (6, 3, 20, 24), dtype=np.uint16),
                           metadata={"axes": "ZCYX", **ome}),
        "ome_czyx": _write(tmp_path / "b.ome.tif", rng.integers(0, 999, (3, 7, 20, 24), dtype=np.uint16),
                           metadata={"axes": "CZYX", **ome}),
        "ome_compressed": _write(tmp_path / "c.ome.tif", rng.integers(0, 999, (6, 3, 40, 48), dtype=np.uint16),
                                 metadata={"axes": "ZCYX", **ome}, compression="zlib", tile=(16, 16)),
        "imagej": _write(tmp_path / "d.tif", rng.integers(0, 999, (6, 2, 20, 24), dtype=np.uint16),
                         imagej=True, metadata={"axes": "ZCYX", "Labels": ["DAPI", "FITC"] * 6}),
        "plain3d": _write(tmp_path / "e.tif", rng.integers(0, 999, (6, 20, 24), dtype=np.uint16)),
        "plain4d": _write(tmp_path / "f.tif", rng.integers(0, 999, (6, 3, 20, 24), dtype=np.uint16)),
    }


@pytest.mark.parametrize("name", ["ome_zcyx", "ome_czyx", "ome_compressed", "imagej", "plain3d", "plain4d"])
def test_read_channel_equals_the_full_load(rounds, name):
    path = rounds[name]
    full = io.load_image_session(path)
    for i, channel in enumerate(full.channels):
        expected = full.data[:, i]
        np.testing.assert_array_equal(io.read_channel(path, channel), expected)
        np.testing.assert_array_equal(io.read_channel(path, i), expected)


def test_read_channel_reads_only_that_channels_pages(rounds, monkeypatch):
    keys = []
    real = tifffile.TiffFile.asarray
    monkeypatch.setattr(tifffile.TiffFile, "asarray",
                        lambda self, *a, **k: (keys.append(k.get("key")), real(self, *a, **k))[1])
    io.read_channel(rounds["ome_zcyx"], "FITC")
    assert keys == [[1, 4, 7, 10, 13, 16]]                       # 6 planes of channel 1 of 3


def test_unknown_channel_is_an_error_not_channel_zero(rounds):
    with pytest.raises(KeyError, match="'GFP'"):
        io.read_channel(rounds["ome_zcyx"], "GFP")
    with pytest.raises(IndexError):
        io.read_channel(rounds["ome_zcyx"], 3)


def test_file_without_one_plane_per_page_falls_back_to_a_full_read(tmp_path, caplog):
    arr = np.random.default_rng(1).integers(0, 999, (2, 4, 32, 32), dtype=np.uint16)
    path = _write(tmp_path / "vol.tif", arr, tile=(4, 16, 16))   # volumetric tiles: a page holds all Z
    full = io.load_image_session(path)
    with caplog.at_level("WARNING"):
        np.testing.assert_array_equal(io.read_channel(path, 1), full.data[:, 1])
    assert any("one plane per page" in r.getMessage() for r in caplog.records)


# --- ND2, through a stand-in ---------------------------------------------------

class _FakeND2:
    """Enough of nd2.ND2File for the loader, read_channel and image_identity."""
    def __init__(self, data, sizes, names):
        self._data, self.sizes, self._names = data, sizes, names
        self.dtype = data.dtype
        self.metadata = types.SimpleNamespace(channels=[
            types.SimpleNamespace(channel=types.SimpleNamespace(name=n)) for n in names])

    def __call__(self, path):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def asarray(self):
        return self._data

    def to_dask(self):
        return self._data

    def voxel_size(self):
        return types.SimpleNamespace(x=0.1, y=0.1, z=0.3)


@pytest.mark.parametrize("axes", ["TZCYX", "CZYX", "ZYX"])
def test_nd2_read_channel_equals_the_full_load(tmp_path, monkeypatch, axes):
    rng = np.random.default_rng(2)
    dims = {"T": 1, "Z": 5, "C": 2, "Y": 12, "X": 14}
    sizes = {ax: dims[ax] for ax in axes}
    names = ["DAPI", "FITC"][:sizes.get("C", 1)]
    fake = _FakeND2(rng.integers(0, 999, tuple(sizes.values()), dtype=np.uint16), sizes, names)
    monkeypatch.setattr(io.nd2, "ND2File", fake)
    path = tmp_path / "r1.nd2"
    path.write_bytes(b"stand-in")
    full = io.load_nd2_session(str(path))
    for i, name in enumerate(full.channels):
        np.testing.assert_array_equal(io.read_channel(path, name), full.data[:, i])
    identity = io.image_identity(path)
    assert identity["shape_zcyx"] == list(full.data.shape) and identity["channels"] == names
    assert identity["voxels"] == [0.3, 0.1, 0.1] and identity["dtype"] == "uint16"


def test_nd2_with_several_positions_is_refused(tmp_path, monkeypatch):
    sizes = {"P": 2, "Z": 3, "C": 1, "Y": 4, "X": 4}
    monkeypatch.setattr(io.nd2, "ND2File", _FakeND2(np.zeros((2, 3, 1, 4, 4), np.uint16), sizes, ["DAPI"]))
    with pytest.raises(ValueError, match="axes"):
        io.read_channel(tmp_path / "x.nd2", "DAPI")


# --- identity and the resolver ---------------------------------------------------

def test_identity_and_its_differences(rounds):
    identity = io.image_identity(rounds["ome_zcyx"])
    assert identity["shape_zcyx"] == [6, 3, 20, 24] and identity["channels"] == ["DAPI", "FITC", "Cy5"]
    assert identity["voxels"] == [0.3, 0.1, 0.1] and identity["dtype"] == "uint16"
    assert io.identity_differences(identity, dict(identity)) == []
    other = io.image_identity(rounds["ome_czyx"])
    diffs = io.identity_differences(identity, other)
    assert any(d.startswith("shape (Z, C, Y, X)") for d in diffs)
    assert not any(d.startswith("channels") for d in diffs)        # same names, different shape


@pytest.fixture
def sess(tmp_path, monkeypatch):
    from zfisher.core import session
    data = {"output_dir": str(tmp_path / "out")}
    (tmp_path / "out" / constants.INPUT_DIR).mkdir(parents=True)
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "update_data", lambda key, value: data.__setitem__(key, value))
    return data


def test_input_copy_wins_then_the_stored_path(sess, rounds, tmp_path):
    sess["r1_path"] = str(rounds["ome_zcyx"])
    assert io.resolve_raw_path("R1") == rounds["ome_zcyx"]
    assert sess["raw_identity"]["R1"]["channels"] == ["DAPI", "FITC", "Cy5"]   # recorded on first find
    copy = tmp_path / "out" / constants.INPUT_DIR / "R1_converted.ome.tif"
    copy.write_bytes(rounds["ome_zcyx"].read_bytes())
    assert io.resolve_raw_path("R1") == copy


def test_missing_image_without_a_chooser_names_round_and_file(sess, tmp_path):
    sess["r2_path"] = str(tmp_path / "gone" / "R2.nd2")
    with pytest.raises(io.RawImageNotFound, match=r"R2 raw image .*R2\.nd2 cannot be found"):
        io.resolve_raw_path("R2")


def test_chosen_file_is_remembered_and_asked_once(sess, rounds, tmp_path):
    sess["r1_path"] = str(tmp_path / "moved" / "R1.ome.tif")
    sess["raw_identity"] = {"R1": io.image_identity(rounds["ome_zcyx"])}
    asked = []
    chooser = lambda r, missing: (asked.append((r, missing)), rounds["ome_zcyx"])[1]
    assert io.resolve_raw_path("R1", chooser=chooser) == rounds["ome_zcyx"]
    assert sess["r1_path"] == str(rounds["ome_zcyx"])
    assert sess["raw_repoints"][0]["from"].endswith("R1.ome.tif") and sess["raw_repoints"][0]["differences"] == []
    assert io.resolve_raw_path("R1", chooser=chooser) == rounds["ome_zcyx"]
    assert len(asked) == 1


def test_a_different_file_needs_confirmation(sess, rounds, tmp_path):
    sess["r1_path"] = str(tmp_path / "moved" / "R1.ome.tif")
    sess["raw_identity"] = {"R1": io.image_identity(rounds["ome_zcyx"])}
    seen = []
    with pytest.raises(io.RawImageNotFound, match="not accepted"):
        io.resolve_raw_path("R1", chooser=lambda r, m: rounds["plain4d"],
                            confirm=lambda r, p, diffs: (seen.append(diffs), False)[1])
    assert any(d.startswith("channels") for d in seen[0])
    assert sess["r1_path"].endswith("moved/R1.ome.tif") and "raw_repoints" not in sess
    assert io.resolve_raw_path("R1", chooser=lambda r, m: rounds["plain4d"],
                               confirm=lambda r, p, diffs: True) == rounds["plain4d"]
    assert sess["raw_repoints"][-1]["differences"]


def test_cancelling_the_chooser_is_an_error(sess, tmp_path):
    sess["r1_path"] = str(tmp_path / "gone.tif")
    with pytest.raises(io.RawImageNotFound, match="No file was chosen"):
        io.resolve_raw_path("R1", chooser=lambda r, m: None)


def test_new_session_records_each_rounds_identity(tmp_path, rounds):
    import logging
    from zfisher.core import session
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    try:
        assert session.initialize_new_session(tmp_path / "new", rounds["ome_zcyx"], rounds["imagej"])
        identity = session.get_data("raw_identity")
        assert identity["R1"]["channels"] == ["DAPI", "FITC", "Cy5"]
        assert identity["R2"]["channels"] == ["DAPI", "FITC"]
    finally:
        session.clear_session()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


# --- the GUI wrapper -------------------------------------------------------------

def test_gui_wrapper_warns_and_returns_none_when_not_located(sess, tmp_path, monkeypatch):
    pytest.importorskip("qtpy")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from zfisher.ui import popups
    warnings = []
    monkeypatch.setattr(popups, "choose_raw_image", lambda parent, r, missing: None)
    monkeypatch.setattr(popups, "show_warning_popup", lambda parent, title, text: warnings.append((title, text)))
    sess["r1_path"] = str(tmp_path / "gone.tif")
    assert popups.resolve_raw_path(None, "R1") is None
    assert warnings and warnings[0][0] == "R1 image not loaded"
