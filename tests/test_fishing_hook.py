"""The Fishing Hook finds the brightest voxel on the line through the cursor.

It sampled that line at 1000 points across +-500 voxels, a step of 1.001
voxels, and rounding to whole planes skipped some planes: in a 2D view, where
the cursor sits on a whole plane, it skipped that very plane, so a punctum
picked on a spot went to the plane above or below it (0.3 to 0.5 um off), or,
for a spot one plane thick, to background. ``puncta.brightest_along_ray``
samples every half voxel.

Run:  python -m pytest tests/test_fishing_hook.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from zfisher.core.puncta import brightest_along_ray

SCALE = (0.3, 0.1, 0.1)
DOWN_Z = (-1.0, 0.0, 0.0)            # napari's view direction in a 2D view


def _stack(spot=(14, 91, 97), shape=(32, 192, 192)):
    rng = np.random.default_rng(0)
    image = rng.normal(50, 5, shape).clip(0).astype(np.uint16)
    image[spot] = 3000                 # a spot one plane thick, as in the synthetic rounds
    return image


@pytest.mark.parametrize("plane", range(32))
def test_a_2d_click_finds_the_spot_from_any_plane(plane):
    found = brightest_along_ray(_stack(), (plane, 91, 97), DOWN_Z, SCALE)
    assert found.tolist() == [14, 91, 97]


@pytest.mark.parametrize("plane", [13.5, 14.0, 14.49, 16.22])
def test_fractional_planes_too(plane):
    assert brightest_along_ray(_stack(), (plane, 91, 97), DOWN_Z, SCALE).tolist() == [14, 91, 97]


def test_an_oblique_3d_ray_finds_a_spot_on_it():
    image = _stack(spot=(20, 120, 60))
    direction = np.array([0.3, 0.1, -0.1]) * 3          # world: one voxel in each of Z, Y, -X per step
    start = (10, 110, 70)                               # ten voxel-steps back along the ray
    assert brightest_along_ray(image, start, direction, SCALE).tolist() == [20, 120, 60]


def test_a_ray_that_misses_the_image_finds_nothing():
    assert brightest_along_ray(_stack(), (10, 500, 500), DOWN_Z, SCALE) is None
    assert brightest_along_ray(_stack(), (10, 91, 97), (0.0, 0.0, 0.0), SCALE) is None


def test_refinement_moves_to_the_brightest_voxel_nearby():
    image = _stack()
    image[14, 92, 98] = 4000                             # brighter, one voxel away in Y and X
    assert brightest_along_ray(image, (14, 91, 97), DOWN_Z, SCALE).tolist() == [14, 91, 97]
    assert brightest_along_ray(image, (14, 91, 97), DOWN_Z, SCALE, refine_um=0.1).tolist() == [14, 92, 98]
