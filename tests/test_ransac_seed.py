"""RANSAC is seeded, so the same centroids give the same shift (defect D42).

The B-spline fit was already seeded; RANSAC was not, so the rigid shift, and
with it every aligned coordinate, could differ between two runs on the same
images. The seed is recorded with the registration.

Run:  python -m pytest tests/test_ransac_seed.py -q
"""
import numpy as np

from zfisher import constants
from zfisher.core import registration


def _clouds():
    rng = np.random.default_rng(7)
    fixed = rng.uniform(0, 500, (120, 3))
    moving = fixed + np.array([2.0, -14.0, 9.0]) + rng.normal(0, 1.5, fixed.shape)
    moving[:30] = rng.uniform(0, 500, (30, 3))                 # a quarter are outliers
    return fixed, moving


def test_the_same_points_give_the_same_model_and_inliers():
    fixed, moving = _clouds()
    runs = [registration._refine_shift_with_ransac(moving, fixed) for _ in range(5)]
    first_model, first_inliers = runs[0]
    for model, inliers in runs[1:]:
        np.testing.assert_array_equal(model.translation, first_model.translation)
        np.testing.assert_array_equal(inliers, first_inliers)


def test_registration_is_repeatable_and_records_its_seed():
    fixed, moving = _clouds()
    results = []
    for _ in range(3):
        details = {}
        shift, rmsd = registration.align_centroids_ransac(fixed, moving, details=details)
        results.append((tuple(np.round(shift, 12)), round(float(rmsd), 12)))
    assert len(set(results)) == 1
    assert details["method"] == "ransac" and details["ransac_seed"] == constants.RANSAC_RANDOM_SEED
