"""Headless tests for centroid registration (no SimpleITK needed).

Covers align_centroids_ransac recovering a known rigid shift and reverting to
the rough estimate when RANSAC can't run. The B-spline path (transform_points_
inverse_bspline) requires SimpleITK and is intentionally not covered here — add
it as a SimpleITK-gated test when that dependency is available in CI.

Run:  pytest tests/test_registration.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher.core import registration as reg


def test_align_recovers_known_shift():
    rng = np.random.RandomState(0)
    fixed = rng.uniform(0, 200, size=(40, 3))
    known = np.array([3.0, -10.0, 7.0])
    moving = fixed + known

    np.random.seed(0)  # RANSAC uses np.random; exact data → deterministic anyway
    shift, rmsd = reg.align_centroids_ransac(fixed, moving)

    assert shift is not None
    # Recovered shift aligns the clouds (sign convention agnostic).
    aligns = (np.allclose(moving - shift, fixed, atol=1.0)
              or np.allclose(moving + shift, fixed, atol=1.0))
    assert aligns, f"shift {shift} did not align the clouds"
    assert rmsd < 1.0


def test_align_returns_rough_shift_when_too_few_points_for_ransac():
    # Fewer than 3 matchable pairs → falls back to the vector-voting rough shift
    # (rmsd 0.0) instead of raising.
    fixed = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]])
    moving = fixed + np.array([1.0, 2.0, 3.0])
    shift, rmsd = reg.align_centroids_ransac(fixed, moving)
    assert shift is not None
    assert shift.shape == (3,)
    assert rmsd == 0.0


def _offset_clouds():
    """Fixed cloud plus a moving cloud whose true shift is one bin (5 px) plus a
    2 px residual along Y. Vector voting recovers only the 5 px part, so after
    the rough shift every pair is exactly 2 px apart. That residual is what
    max_distance is measured against."""
    rng = np.random.RandomState(1)
    fixed = rng.uniform(0, 400, size=(40, 3))
    moving = fixed + np.array([5.0, 7.0, 5.0])
    return fixed, moving


def test_max_distance_bounds_pair_search_in_pixels():
    fixed, moving = _offset_clouds()
    # Radius smaller than the 2 px residual: no pairs, fall back to rough shift.
    shift_tight, rmsd_tight = reg.align_centroids_ransac(fixed, moving, max_distance=1.0)
    assert rmsd_tight == 0.0
    np.testing.assert_allclose(np.abs(shift_tight), [5.0, 5.0, 5.0])
    # Radius larger than the residual: pairs found, RANSAC recovers the 7.
    shift_loose, rmsd_loose = reg.align_centroids_ransac(fixed, moving, max_distance=3.0)
    assert rmsd_loose < 1.0
    aligns = (np.allclose(moving - shift_loose, fixed, atol=0.5)
              or np.allclose(moving + shift_loose, fixed, atol=0.5))
    assert aligns, f"shift {shift_loose} did not align the clouds"


def test_max_distance_is_physical_when_voxels_given():
    fixed, moving = _offset_clouds()
    # 2 px residual along Y at 0.25 um/px is 0.5 um: inside a 1 um radius.
    shift, rmsd = reg.align_centroids_ransac(
        fixed, moving, max_distance=1.0, voxels=(1.0, 0.25, 0.25))
    assert rmsd < 1.0
    # The same 1 um radius with 1 um/px voxels is 1 px: the 2 px residual is outside.
    shift_iso, rmsd_iso = reg.align_centroids_ransac(
        fixed, moving, max_distance=1.0, voxels=(1.0, 1.0, 1.0))
    assert rmsd_iso == 0.0


def test_max_distance_zero_or_none_uses_default():
    fixed, moving = _offset_clouds()
    ref_shift, ref_rmsd = reg.align_centroids_ransac(fixed, moving)
    for md in (0, 0.0, None):
        shift, rmsd = reg.align_centroids_ransac(fixed, moving, max_distance=md)
        np.testing.assert_allclose(shift, ref_shift)
        assert rmsd == ref_rmsd


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
