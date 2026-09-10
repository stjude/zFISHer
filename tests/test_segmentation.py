"""Headless tests for core segmentation array logic (no cellpose/GPU needed).

Covers the volume-threshold heuristic, the Intersection/Union mask merge
(including Z-padding), and the nucleus label-matching — the last as a
*characterization* test that documents CURRENT behavior, including the
order-dependent collision tie-break flagged for the Rank 6 correctness fix.

Run:  pytest tests/test_segmentation.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher.core import segmentation as seg


# --- compute_min_volume_threshold -------------------------------------------

def test_min_volume_threshold_small_sample_returns_one():
    assert seg.compute_min_volume_threshold(np.array([])) == 1
    assert seg.compute_min_volume_threshold(np.array([10, 20])) == 1   # < 3 volumes


def test_min_volume_threshold_median_minus_2mad():
    vols = np.array([5, 100, 110, 120])           # median 105, MAD 10
    assert seg.compute_min_volume_threshold(vols) == 85   # max(1, int(105 - 2*10))


# --- merge_labeled_masks -----------------------------------------------------

def test_merge_intersection_keeps_mask1_ids_only_where_both_present():
    m1 = np.zeros((1, 3, 3), dtype=np.int32); m1[0, 0, :] = 5
    m2 = np.zeros((1, 3, 3), dtype=np.int32); m2[0, 0, 0] = 9; m2[0, 2, 2] = 7
    merged = seg.merge_labeled_masks(m1, m2, method="Intersection")
    assert merged[0, 0, 0] == 5                   # overlap → mask1's id
    assert set(np.unique(merged)) == {0, 5}       # only the overlapping voxel survives
    assert int((merged > 0).sum()) == 1


def test_merge_union_fills_mask2_where_mask1_empty():
    m1 = np.zeros((1, 3, 3), dtype=np.int32); m1[0, 0, :] = 5
    m2 = np.zeros((1, 3, 3), dtype=np.int32); m2[0, 0, 0] = 9; m2[0, 2, 2] = 7
    merged = seg.merge_labeled_masks(m1, m2, method="Union")
    assert merged[0, 0, 0] == 5                   # mask1 wins where it has a label
    assert merged[0, 2, 2] == 7                   # mask2 fills the gap
    assert set(np.unique(merged)) == {0, 5, 7}


def test_merge_pads_mismatched_z():
    m1 = np.full((1, 2, 2), 3, dtype=np.int32)    # one Z slice
    m2 = np.zeros((2, 2, 2), dtype=np.int32); m2[1] = 4   # two Z slices
    merged = seg.merge_labeled_masks(m1, m2, method="Union")
    assert merged.shape == (2, 2, 2)
    assert (merged[0] == 3).all()                 # mask1's slice preserved
    assert (merged[1] == 4).all()                 # mask2's extra slice filled in


# --- match_nuclei_labels -----------------------------------------------------

def test_match_relabels_nearby_nucleus_to_mask1_id():
    m1 = np.zeros((1, 10, 10), dtype=np.int32); m1[0, 1:3, 1:3] = 1
    m2 = np.zeros((1, 10, 10), dtype=np.int32); m2[0, 1:3, 1:3] = 7   # same place, different id
    new_m2, _, _ = seg.match_nuclei_labels(m1, m2, threshold=5)
    assert 1 in np.unique(new_m2)                 # relabeled to mask1's id
    assert 7 not in np.unique(new_m2)


def test_match_far_nucleus_gets_a_fresh_id():
    m1 = np.zeros((1, 12, 12), dtype=np.int32); m1[0, 4:6, 4:6] = 1
    m2 = np.zeros((1, 12, 12), dtype=np.int32); m2[0, 9:11, 9:11] = 1  # far away
    new_m2, _, _ = seg.match_nuclei_labels(m1, m2, threshold=2)
    # Beyond threshold → not matched to id 1; gets next id (max(mask1)+1 = 2).
    assert 2 in np.unique(new_m2)
    assert 1 not in np.unique(new_m2)


def test_match_collision_closest_nucleus_wins():
    # Two mask2 nuclei both within threshold of mask1's single nucleus. The
    # CLOSER one must inherit mask1's id regardless of label/iteration order;
    # the farther one gets a fresh id. Deterministic + distance-based (Rank 6).
    m1 = np.zeros((1, 12, 12), dtype=np.int32); m1[0, 5:8, 5:8] = 1   # centroid ~ (6,6)
    m2 = np.zeros((1, 12, 12), dtype=np.int32)
    m2[0, 2:4, 2:4] = 1     # old label 1: FAR from (6,6)  (centroid ~ (2.5,2.5))
    m2[0, 5:7, 5:7] = 2     # old label 2: CLOSE to (6,6)  (centroid ~ (5.5,5.5))
    new_m2, _, _ = seg.match_nuclei_labels(m1, m2, threshold=8)
    vals = set(np.unique(new_m2)) - {0}
    assert vals == {1, 2}
    # The CLOSER nucleus (old label 2) inherits mask1's id 1, even though it is
    # NOT first in label order — proving the winner is distance-based, not order.
    assert new_m2[0, 5, 5] == 1
    # The farther nucleus (old label 1) is bumped to a fresh id.
    assert new_m2[0, 2, 2] == 2


def test_match_threshold_none_and_zero_both_auto_detect():
    # Sentinel contract (after the #1 clarity cleanup): both None and 0 mean
    # "auto-detect", and produce identical results. A positive value is honored
    # separately (see the far/near tests above).
    m1 = np.zeros((1, 10, 10), dtype=np.int32); m1[0, 1:3, 1:3] = 1
    m2 = np.zeros((1, 10, 10), dtype=np.int32); m2[0, 1:3, 1:3] = 7
    out_none, _, _ = seg.match_nuclei_labels(m1, m2, threshold=None)
    out_zero, _, _ = seg.match_nuclei_labels(m1, m2, threshold=0)
    assert 1 in np.unique(out_none) and 7 not in np.unique(out_none)   # auto-matched
    assert np.array_equal(out_none, out_zero)                          # 0 == None == auto


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
