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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
