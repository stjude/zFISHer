"""Tests for ``zfisher.core.io.write_label_tif`` and the mask file helpers.

Edited masks used to be written uncompressed and straight over the only copy on
disk: about 1.2 GB per save for a typical volume, and a truncated file if the
write died halfway. ``write_label_tif`` writes a compressed temporary file and
swaps it into place, and overwrites in place only when Windows refuses the swap
because another program holds the file open.

Masks are tiled OME-TIFFs carrying their voxel size. ``LabelTiles`` keeps the
compressed tiles between saves, so a save compresses only the tiles an edit
marked.

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


# --- Tiled OME-TIFF ------------------------------------------------------------

def _volume(dtype=np.uint32):
    """Three planes, not a multiple of the tile size, labels across tile edges."""
    data = np.zeros((3, 300, 520), dtype=dtype)
    data[0:2, 240:270, 250:262] = 4
    data[1:3, 10:50, 500:520] = 9
    data[2, 299, 519] = 2
    return data


def test_mask_is_a_tiled_ome_tiff_with_its_voxel_size(tmp_path):
    path = tmp_path / "m.ome.tif"
    data = _volume()
    zio.write_label_tif(path, data, voxel_size=(0.3, 0.108, 0.108))
    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome and tif.series[0].axes == "ZYX"
        page = tif.pages[0]
        assert page.is_tiled and (page.tilelength, page.tilewidth) == (256, 256)
        ome = tif.ome_metadata
    for size in ('PhysicalSizeZ="0.3"', 'PhysicalSizeY="0.108"', 'PhysicalSizeX="0.108"'):
        assert size in ome
    assert np.array_equal(zio.read_label_tif(path), data)
    assert np.array_equal(tifffile.imread(path), data)    # any TIFF reader gets the same


def test_invalid_voxel_sizes_are_left_out_of_the_metadata(tmp_path):
    path = tmp_path / "m.ome.tif"
    zio.write_label_tif(path, _volume(), voxel_size=(float("nan"), 0, 0.1))
    with tifffile.TiffFile(path) as tif:
        ome = tif.ome_metadata
    assert "PhysicalSizeZ" not in ome and "PhysicalSizeY=" not in ome and 'PhysicalSizeX="0.1"' in ome


def test_one_plane_mask_reads_back_three_dimensional(tmp_path):
    path = tmp_path / "m.ome.tif"
    data = np.zeros((1, 40, 30), np.uint32)
    data[0, 5, 6] = 3
    zio.write_label_tif(path, data)
    back = zio.read_label_tif(path)
    assert back.shape == (1, 40, 30) and np.array_equal(back, data)


def test_legacy_plain_tiff_mask_still_reads(tmp_path):
    path = tmp_path / "old.tif"
    tifffile.imwrite(path, _volume(), compression="zlib")
    assert np.array_equal(zio.read_label_tif(path), _volume())


def test_only_three_dimensional_integer_masks(tmp_path):
    with pytest.raises(ValueError):
        zio.write_label_tif(tmp_path / "a.ome.tif", np.zeros((30, 30), np.uint32))
    with pytest.raises(ValueError):
        zio.write_label_tif(tmp_path / "b.ome.tif", np.zeros((2, 30, 30), np.float32))


def test_int64_labels_are_written_as_uint32_if_they_fit(tmp_path):
    path = tmp_path / "m.ome.tif"
    zio.write_label_tif(path, _volume(np.int64))
    back = zio.read_label_tif(path)
    assert back.dtype == np.uint32 and np.array_equal(back, _volume())
    too_big = _volume(np.int64)
    too_big[0, 0, 0] = 2**40
    with pytest.raises(ValueError):
        zio.write_label_tif(path, too_big)
    assert np.array_equal(zio.read_label_tif(path), _volume())     # old file kept


def test_only_marked_tiles_are_compressed_again(tmp_path):
    from zfisher.core import label_edits
    path = tmp_path / "m.ome.tif"
    data = _volume()
    tiles = zio.LabelTiles(data)
    assert tiles.pending == 3 * 2 * 3                    # 3 planes of 2 x 3 tiles
    zio.write_label_tif(path, data, tiles=tiles)
    assert tiles.pending == 0
    # Label 4 straddles four tiles in two planes.
    indices = np.nonzero(data == 4)
    data[indices] = 0
    tiles.mark(label_edits.index_extent(indices))
    assert tiles.pending == 8
    zio.write_label_tif(path, data, tiles=tiles)
    assert np.array_equal(zio.read_label_tif(path), data)
    # A box ending exactly on a tile edge marks no tile beyond it.
    tiles.mark((slice(0, 1), slice(0, 256), slice(256, 512)))
    assert tiles.pending == 1


def test_unmarked_edit_stays_stale_until_everything_is_marked(tmp_path):
    """The contract callers rely on: only marked tiles are compressed again."""
    path = tmp_path / "m.ome.tif"
    data = _volume()
    tiles = zio.LabelTiles(data)
    zio.write_label_tif(path, data, tiles=tiles)
    data[2, 299, 519] = 0                                # not marked
    zio.write_label_tif(path, data, tiles=tiles)
    assert zio.read_label_tif(path)[2, 299, 519] == 2
    tiles.mark_all()
    zio.write_label_tif(path, data, tiles=tiles)
    assert np.array_equal(zio.read_label_tif(path), data)


def test_random_marked_edits_keep_the_file_exact(tmp_path):
    from zfisher.core import label_edits
    rng = np.random.default_rng(0)
    path = tmp_path / "m.ome.tif"
    data = _volume()
    tiles = zio.LabelTiles(data)
    for _ in range(30):
        z0 = rng.integers(0, 3)
        y0, x0 = rng.integers(0, 300), rng.integers(0, 520)
        region = (slice(z0, z0 + 1), slice(y0, min(300, y0 + rng.integers(1, 80))),
                  slice(x0, min(520, x0 + rng.integers(1, 80))))
        data[region] = rng.integers(0, 20)
        tiles.mark(region)
        zio.write_label_tif(path, data, tiles=tiles)
        assert np.array_equal(zio.read_label_tif(path), data)


def test_find_mask_prefers_ome_tiff_and_falls_back_to_legacy(tmp_path):
    assert zio.find_mask(tmp_path, "R1 - DAPI_masks") is None
    (tmp_path / "R1 - DAPI_masks.tif").write_bytes(b"")
    assert zio.find_mask(tmp_path, "R1 - DAPI_masks").name == "R1 - DAPI_masks.tif"
    (tmp_path / "R1 - DAPI_masks.ome.tif").write_bytes(b"")
    assert zio.find_mask(tmp_path, "R1 - DAPI_masks").name == "R1 - DAPI_masks.ome.tif"
    assert zio.mask_path(tmp_path, "X_masks") == tmp_path / "X_masks.ome.tif"
