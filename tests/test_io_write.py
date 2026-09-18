"""Tests for ``zfisher.core.io.write_label_tif``.

Edited masks used to be written uncompressed and straight over the only copy on
disk: about 1.2 GB per save for a typical volume, and a truncated file if the
write died halfway. ``write_label_tif`` writes a compressed temporary file and
swaps it into place, and overwrites in place only when Windows refuses the swap
because another program holds the file open.

``zfisher.core.io`` imports the ND2 reader, so these are skipped where it is not
installed.

Run:  python -m pytest tests/test_io_write.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tifffile

pytest.importorskip("nd2")

from zfisher.core import io as zio


def _mask(label, dtype=np.uint32):
    data = np.zeros((4, 64, 64), dtype=dtype)       # 4 slices: the depth tifffile mistakes for RGBA
    data[1:3, 8:40, 8:40] = label
    return data


def _tmp_of(path):
    return path.with_name(path.name + ".tmp")


def _always_busy(src, dst):
    raise PermissionError(13, "Access is denied")


def test_round_trip_is_identical_compressed_and_greyscale(tmp_path):
    for dtype in (np.uint32, np.int32):
        path = tmp_path / f"mask_{np.dtype(dtype).name}.tif"
        data = _mask(7, dtype)
        assert zio.write_label_tif(path, data) == "atomic"
        back = tifffile.imread(path)
        assert back.dtype == data.dtype and np.array_equal(back, data)
        with tifffile.TiffFile(path) as tif:
            assert int(tif.pages[0].compression) != 1            # 1 means uncompressed
            assert tif.pages[0].photometric.name == "MINISBLACK"  # not colour planes
        assert path.stat().st_size < data.nbytes / 10
        assert not _tmp_of(path).exists()


def test_failed_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    path = tmp_path / "mask.tif"
    old = _mask(1)
    zio.write_label_tif(path, old)

    def _dies_halfway(target, data, **kwargs):
        with open(target, "wb") as f:
            f.write(b"II*\x00 half a file")
        raise OSError("disk full")
    monkeypatch.setattr(zio.tifffile, "imwrite", _dies_halfway)

    with pytest.raises(OSError):
        zio.write_label_tif(path, _mask(2))
    assert np.array_equal(tifffile.imread(path), old)      # the old mask is untouched
    assert not _tmp_of(path).exists()                      # and the debris is gone


def test_refused_swap_falls_back_to_overwriting_in_place(tmp_path, monkeypatch):
    path = tmp_path / "mask.tif"
    zio.write_label_tif(path, _mask(1))
    monkeypatch.setattr(zio.os, "replace", _always_busy)

    assert zio.write_label_tif(path, _mask(2)) == "direct"
    assert np.array_equal(tifffile.imread(path), _mask(2))
    assert not _tmp_of(path).exists()


def test_when_both_paths_fail_the_complete_copy_is_kept(tmp_path, monkeypatch):
    path = tmp_path / "mask.tif"
    zio.write_label_tif(path, _mask(1))
    real_imwrite = tifffile.imwrite

    def _only_the_temp_file_can_be_written(target, data, **kwargs):
        if str(target).endswith(".tmp"):
            return real_imwrite(target, data, **kwargs)
        raise OSError("destination is read-only")
    monkeypatch.setattr(zio.os, "replace", _always_busy)
    monkeypatch.setattr(zio.tifffile, "imwrite", _only_the_temp_file_can_be_written)

    with pytest.raises(OSError):
        zio.write_label_tif(path, _mask(2))
    assert np.array_equal(tifffile.imread(_tmp_of(path)), _mask(2))   # nothing was lost
