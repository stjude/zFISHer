"""The registration is saved and puncta can be transformed again from disk.

Canvas generation used to keep the fitted B-spline only in memory, and the
fit's random metric sampling was unseeded, so a reopened session could neither
reapply the warp nor reproduce it. Canvas generation now writes the B-spline to
``aligned/registration_bspline.tfm`` and records the shift, canvas offset and
per-round translations in the session's ``registration`` record, next to the
rigid fit's method, fallback reason and inlier fraction.
``registration.load_registration`` reads it back and
``puncta.transform_puncta_csv`` transforms a raw puncta CSV with it.

Needs SimpleITK; runs without napari.

Run:  python -m pytest tests/test_saved_registration.py -q
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

sitk = pytest.importorskip("SimpleITK")

from zfisher import constants
from zfisher.core import puncta, registration as reg, session


@pytest.fixture
def store(monkeypatch):
    """The session as an in-memory dict, with a temporary output folder."""
    out = Path(tempfile.mkdtemp())
    (out / constants.ALIGNED_DIR).mkdir()
    data = {"output_dir": str(out)}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None:
                        data.get(key, default) if key else data)
    monkeypatch.setattr(session, "update_data", lambda key, value: data.__setitem__(key, value))
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    monkeypatch.setattr(reg, "set_processed_file", lambda *a, **k: None)
    monkeypatch.setattr(session, "get_nuclear_channel", lambda: "DAPI")
    yield data
    shutil.rmtree(out, ignore_errors=True)


def _rounds():
    """R1 and R2 nuclei and one puncta channel; R2 is R1 shifted and bent."""
    from scipy import ndimage as ndi
    rng = np.random.default_rng(0)
    r1 = np.zeros((12, 128, 128), np.float32)
    for _ in range(25):
        z, y, x = rng.integers(2, 10), rng.integers(10, 118), rng.integers(10, 118)
        r1[z - 2:z + 2, y - 6:y + 6, x - 6:x + 6] = rng.uniform(100, 200)
    zz, yy, xx = np.meshgrid(np.arange(12), np.arange(128), np.arange(128), indexing="ij")
    r2 = ndi.map_coordinates(r1, [zz, yy + 3 + 2 * np.sin(xx / 30), xx - 2 + 1.5 * np.cos(yy / 25)],
                             order=1).astype(np.float32)

    def layers(prefix, dapi):
        return [{"name": f"{prefix} - DAPI", "data": dapi, "scale": (0.3, 0.1, 0.1)},
                {"name": f"{prefix} - FITC", "data": dapi * 0.5, "scale": (0.3, 0.1, 0.1)}]
    return layers("R1", r1), layers("R2", r2)


def _canvas(store, apply_warp=True, shift=(0.0, -3.4, 2.2)):
    r1_layers, r2_layers = _rounds()
    out = Path(store["output_dir"]) / constants.ALIGNED_DIR
    _, transform, offset = reg.generate_global_canvas(r1_layers, r2_layers, np.array(shift), out,
                                                      apply_warp=apply_warp)
    return transform, offset


def _raw_puncta(n=40, seed=1):
    rng = np.random.default_rng(seed)
    coords = np.column_stack([rng.uniform(1, 11, n), rng.uniform(5, 120, n), rng.uniform(5, 120, n)])
    return np.column_stack([coords, np.zeros(n), np.full(n, 90.0), np.full(n, 3.0)])


def test_canvas_saves_the_bspline_and_its_record(store):
    _canvas(store)
    canvas = store["registration"]["canvas"]
    path = Path(store["output_dir"]) / canvas["bspline"]["path"]
    assert path.name == constants.REGISTRATION_TRANSFORM_FILE and path.exists()
    assert not Path(canvas["bspline"]["path"]).is_absolute()      # survives the folder moving
    assert canvas["bspline"]["seed"] == constants.DEFORMABLE_RANDOM_SEED
    offset = np.array(canvas["canvas_offset_pixels"])
    np.testing.assert_allclose(canvas["raw_to_canvas"]["R1"], -offset)
    np.testing.assert_allclose(canvas["raw_to_canvas"]["R2"], np.array(canvas["shift"]) - offset)
    assert canvas["round_shapes"] == {"R1": [12, 128, 128], "R2": [12, 128, 128]}


def test_saved_registration_reproduces_the_in_memory_one(store):
    transform, offset = _canvas(store)
    saved = reg.load_registration()
    np.testing.assert_array_equal(saved.canvas_offset, offset)
    pts = np.random.default_rng(2).uniform(0, 120, (30, 3))
    for p in pts:
        assert saved.bspline.TransformPoint(list(p)) == transform.TransformPoint(list(p))


@pytest.mark.parametrize("round_id", ["R1", "R2"])
def test_puncta_csv_transformed_from_disk_equals_the_original_transform(store, round_id):
    transform, offset = _canvas(store)
    raw = _raw_puncta()
    ids = np.arange(100, 100 + len(raw))
    sources = np.where(ids % 3 == 0, "manual", "auto").astype(object)
    shift = np.array(store["registration"]["canvas"]["shift"])
    expected = puncta.transform_puncta_to_aligned_space(
        raw, round_id, shift, offset, bspline_transform=transform if round_id == "R2" else None,
        puncta_ids=ids, sources=sources)
    csv = Path(store["output_dir"]) / "raw.csv"
    puncta.write_puncta_csv(csv, raw[:, :3], pd.DataFrame({
        "puncta_id": ids, "Nucleus_ID": raw[:, 3], "Intensity": raw[:, 4], "SNR": raw[:, 5],
        "Source": sources}))
    data, out_sources = puncta.transform_puncta_csv(csv, round_id, reg.load_registration())
    np.testing.assert_array_equal(data, expected)        # the CSV is read back exactly
    np.testing.assert_array_equal(data[:, 6], expected[:, 6])
    assert list(out_sources) == list(sources)


def test_without_warp_r2_is_rigid_only(store):
    transform, offset = _canvas(store, apply_warp=False)
    assert transform is None
    saved = reg.load_registration()
    assert saved.bspline is None and store["registration"]["canvas"]["bspline"] is None
    raw = _raw_puncta(5)
    csv = Path(store["output_dir"]) / "raw.csv"
    puncta.write_puncta_csv(csv, raw[:, :3])
    data, _ = puncta.transform_puncta_csv(csv, "R2", saved)
    np.testing.assert_allclose(data[:, :3], raw[:, :3] + saved.shift - offset)


def test_the_bspline_fit_is_repeatable():
    r1, r2 = _rounds()
    a = reg.calculate_deformable_transform(r1[0]["data"], r2[0]["data"])
    b = reg.calculate_deformable_transform(r1[0]["data"], r2[0]["data"])
    assert a.GetParameters() == b.GetParameters()


def test_load_registration_explains_what_is_missing(store):
    with pytest.raises(LookupError):
        reg.load_registration()
    _canvas(store)
    (Path(store["output_dir"]) / store["registration"]["canvas"]["bspline"]["path"]).unlink()
    with pytest.raises(FileNotFoundError):
        reg.load_registration()


def test_moved_output_folder_still_finds_the_bspline(store):
    _canvas(store)
    moved = Path(tempfile.mkdtemp()) / "moved"
    shutil.copytree(store["output_dir"], moved)
    store["output_dir"] = str(moved)
    assert reg.load_registration().bspline is not None


def test_rigid_fit_records_method_inliers_and_fallback(store):
    rng = np.random.default_rng(3)
    fixed = rng.uniform(0, 100, (40, 3))
    shift, _ = reg.calculate_session_registration(fixed, fixed + [1.0, 2.0, -3.0])
    rigid = store["registration"]["rigid"]
    assert rigid["method"] == "ransac" and rigid["fallback_used"] is False
    assert rigid["fallback_reason"] is None and rigid["inlier_fraction"] > 0.9
    store["registration"]["canvas"] = {"stale": True}
    details = {}
    reg.align_centroids_ransac(fixed[:2], fixed[:2] + 1.0, details=details)
    assert details["method"] == "rough_shift" and details["fallback_reason"] == "too_few_pairs"
    reg.calculate_session_registration(fixed[:2], fixed[:2] + 1.0)
    assert store["registration"]["rigid"]["fallback_used"] is True
    assert "canvas" not in store["registration"]                   # a new shift drops the old canvas
