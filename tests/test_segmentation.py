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



# --- no nuclei found --------------------------------------------------------
# peak_local_max returns its empty result as shape (0, 2) for a 3D image, which
# used to crash the classical watershed with a numpy broadcasting error.

def _disk_stack(n_planes, size=256, radius=40):
    """One bright cylinder through every plane of an (n_planes, size, size) stack."""
    img = np.zeros((n_planes, size, size), dtype=np.uint16)
    yy, xx = np.mgrid[:size, :size]
    img[:, (yy - size // 2) ** 2 + (xx - size // 2) ** 2 < radius ** 2] = 1000
    return img


def test_classical_blank_stack_returns_empty_result():
    steps = []
    labels, centroids = seg.segment_nuclei_classical(
        np.zeros((10, 64, 64), dtype=np.uint16), voxel_spacing=(0.5, 0.1, 0.1),
        progress_callback=lambda p, text: steps.append((p, text)))
    assert labels is None and centroids is None
    assert steps[-1] == (100, "No nuclei found.")


def test_classical_needs_min_z_planes():
    # The planes kept after Z downsampling must leave room for a peak away from
    # the top and bottom edges: 29 planes with the default constants.
    n = seg.min_classical_z_planes()
    assert n == 29
    labels, _ = seg.segment_nuclei_classical(_disk_stack(n - 1), voxel_spacing=(0.5, 0.1, 0.1))
    assert labels is None
    labels, centroids = seg.segment_nuclei_classical(_disk_stack(n), voxel_spacing=(0.5, 0.1, 0.1))
    assert labels.max() == 1 and centroids.shape == (1, 3)


def test_no_nuclei_message_explains_too_few_planes():
    short = seg.no_nuclei_message("R2 - DAPI", 12, "classical")
    assert short.startswith("No nuclei were found in R2 - DAPI.")
    assert "12 Z planes" in short and "at least 29" in short and "Cellpose" in short
    for text in (seg.no_nuclei_message("R2 - DAPI", 40, "classical"),
                 seg.no_nuclei_message("R2 - DAPI", 12, "cellpose")):
        assert "Z planes" not in text and "nuclear channel" in text


def test_session_segmentation_names_the_round_and_saves_nothing():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as out:
        try:
            seg.process_session_dapi(_disk_stack(40), np.zeros((12, 64, 64), dtype=np.uint16),
                                     output_dir=out, voxel_spacing=(0.5, 0.1, 0.1))
        except seg.NoNucleiFound as exc:
            message = str(exc)
        else:
            raise AssertionError("expected NoNucleiFound")
        assert message.startswith("No nuclei were found in R2.") and "R1" not in message
        assert "12 Z planes" in message
        assert not (Path(out) / "segmentation").exists()


def test_widget_warns_and_keeps_the_round_with_nuclei(monkeypatch, tmp_path):
    """The GUI shows the other round's result and a warning, not an error."""
    import types
    import pytest
    pytest.importorskip("napari")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import contextlib
    import napari
    from napari.components import ViewerModel
    from zfisher.core import session
    from zfisher.ui import popups
    from zfisher.ui.widgets import dapi_segmentation_widget as mod

    data = {"output_dir": str(tmp_path)}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

    viewer = _Viewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    shown = []
    monkeypatch.setattr(popups, "show_warning_popup", lambda parent, title, text: shown.append(("warning", title, text)))
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: shown.append(("error", title, text)))

    class _Dialog(contextlib.nullcontext):
        def __init__(self, *a, **k):
            super().__init__(self)

        def update_progress(self, *a):
            pass

        def freeze_canvas(self):
            pass

    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)

    r1 = model.add_image(_disk_stack(40), name="R1 - DAPI", scale=(0.5, 0.1, 0.1))
    r2 = model.add_image(np.zeros((12, 64, 64), dtype=np.uint16), name="R2 - DAPI", scale=(0.5, 0.1, 0.1))
    widget = mod.DapiSegmentationWidget()
    widget.r1_layer = types.SimpleNamespace(value=r1)
    widget.r2_layer = types.SimpleNamespace(value=r2)
    widget._on_run()

    assert [s[:2] for s in shown] == [("warning", "No Nuclei Found")]
    assert shown[0][2].startswith("No nuclei were found in R2 - DAPI.") and "12 Z planes" in shown[0][2]
    assert "R1 - DAPI_masks" in [layer.name for layer in model.layers]
    assert "R2 - DAPI_masks" not in [layer.name for layer in model.layers]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fns = [fn for fn in fns if not fn.__code__.co_argcount]   # pytest fixtures: run under pytest
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
