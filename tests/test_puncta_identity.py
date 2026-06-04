"""Headless tests for the puncta CSV identity fix (Phases A-B).

Covers the canonical naming helpers and the stable ``puncta_id`` lifecycle in
``zfisher.core.puncta`` (assignment at detection, carry-through the world-space
transform's extranuclear filter, and persistence to CSV in canonical column
order). Runs without napari/Qt.

Run:  python -m pytest tests/test_puncta_identity.py -q
or:   python tests/test_puncta_identity.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher import constants
from zfisher.core import puncta
from zfisher.core import analysis


NAME = "Aligned R1 - FITC_puncta"


def test_naming_helpers():
    assert constants.puncta_csv_stem(NAME) == "Aligned_R1_-_FITC_puncta"
    p = constants.puncta_csv_path("/tmp/reports", NAME)
    assert p.name == "Aligned_R1_-_FITC_puncta.csv"
    # All writers route through this one helper, so a layer maps to one file.
    assert constants.puncta_csv_path("/tmp/reports", NAME).name == p.name
    # A name without spaces is unchanged.
    assert constants.puncta_csv_stem("Foo_puncta") == "Foo_puncta"
    # Schema constant: puncta_id leads.
    assert constants.PUNCTA_CSV_COLUMNS[0] == "puncta_id"
    assert constants.PUNCTA_CSV_COLUMNS == [
        "puncta_id", "Z", "Y", "X", "Nucleus_ID", "Intensity", "SNR", "Source"
    ]


def _raw7(coords, ids, nuc_placeholder=0.0, intensity=100.0, snr=5.0):
    """Build an (N,7) raw_puncta array: Z,Y,X,Nucleus_ID,Intensity,SNR,puncta_id."""
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    return np.column_stack([
        coords,
        np.full(n, nuc_placeholder, dtype=float),
        np.full(n, intensity, dtype=float),
        np.full(n, snr, dtype=float),
        np.asarray(ids, dtype=float),
    ])


def _mask_with(shape, coord_to_label):
    m = np.zeros(shape, dtype=np.int32)
    for (z, y, x), label in coord_to_label.items():
        m[z, y, x] = label
    return m


def test_transform_carries_puncta_id_through_filter():
    # 5 puncta with arbitrary *stable* ids; 0,2,4 inside nuclei, 1,3 outside.
    coords = [(1, 1, 1), (1, 1, 5), (1, 5, 1), (1, 5, 5), (2, 8, 8)]
    ids = [10, 11, 12, 13, 14]
    raw = _raw7(coords, ids)
    mask = _mask_with((3, 10, 10), {
        (1, 1, 1): 5,   # id 10 -> nucleus 5
        (1, 5, 1): 7,   # id 12 -> nucleus 7
        (2, 8, 8): 9,   # id 14 -> nucleus 9
        # (1,1,5) and (1,5,5) stay background 0 -> dropped
    })

    out = puncta.transform_puncta_to_aligned_space(
        raw, round_id="R1", shift=np.zeros(3), canvas_offset=np.zeros(3),
        consensus_mask=mask, remove_extranuclear=True,
    )
    assert out.shape == (3, 7), out.shape
    # Surviving ids are the ORIGINAL ones (non-contiguous), not re-aranged.
    assert list(out[:, 6].astype(int)) == [10, 12, 14]
    # Nucleus IDs recomputed from the mask, aligned with the surviving ids.
    assert list(out[:, 3].astype(int)) == [5, 7, 9]


def test_transform_puncta_ids_kwarg_when_no_id_column():
    # Same scenario but raw_puncta is (N,6); ids supplied via kwarg.
    coords = [(1, 1, 1), (1, 1, 5), (1, 5, 1)]
    raw6 = _raw7(coords, [0, 0, 0])[:, :6]  # drop the id column
    mask = _mask_with((3, 10, 10), {(1, 1, 1): 5, (1, 5, 1): 7})
    out = puncta.transform_puncta_to_aligned_space(
        raw6, round_id="R1", shift=np.zeros(3), canvas_offset=np.zeros(3),
        consensus_mask=mask, remove_extranuclear=True,
        puncta_ids=[10, 11, 12],
    )
    assert out.shape == (2, 7)
    assert list(out[:, 6].astype(int)) == [10, 12]


def test_transform_writes_canonical_csv():
    coords = [(1, 1, 1), (1, 5, 1)]
    raw = _raw7(coords, [3, 4])
    mask = _mask_with((3, 10, 10), {(1, 1, 1): 5, (1, 5, 1): 7})
    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "x.csv")
        puncta.transform_puncta_to_aligned_space(
            raw, round_id="R1", shift=np.zeros(3), canvas_offset=np.zeros(3),
            consensus_mask=mask, remove_extranuclear=True,
            output_path=out_path, layer_name=NAME,
        )
        df = pd.read_csv(out_path)
        assert list(df.columns) == constants.PUNCTA_CSV_COLUMNS
        assert list(df["puncta_id"]) == [3, 4]
        assert list(df["Nucleus_ID"]) == [5, 7]


def test_transform_empty_returns_width_7():
    out = puncta.transform_puncta_to_aligned_space(
        np.empty((0, 6)), round_id="R1", shift=np.zeros(3), canvas_offset=np.zeros(3),
    )
    assert out.shape == (0, 7)


def test_detection_assigns_unique_puncta_id_and_schema():
    img = np.zeros((3, 12, 12), dtype=float)
    img[1, 2, 2] = 1000
    img[1, 2, 8] = 900
    img[1, 8, 5] = 800
    mask_all = np.ones((3, 12, 12), dtype=np.int32)  # every spot inside a nucleus
    params = {"method": "Local Maxima", "min_distance": 2, "threshold_rel": 0.1,
              "sigma": 0.0, "nuclei_only": True}

    with tempfile.TemporaryDirectory() as d:
        out_path = os.path.join(d, "raw.csv")
        res = puncta.process_puncta_detection(
            img, mask_data=mask_all, voxels=(1, 1, 1), params=params, output_path=out_path,
        )
        assert res.shape[1] == 7, res.shape
        assert res.shape[0] == 3, res.shape
        pid = res[:, 6].astype(int)
        assert len(set(pid)) == len(pid)            # unique
        assert set(pid) == {0, 1, 2}                # arange before filter, none dropped
        df = pd.read_csv(out_path)
        assert list(df.columns) == constants.PUNCTA_CSV_COLUMNS
        assert sorted(df["puncta_id"]) == [0, 1, 2]


def test_detection_all_filtered_returns_width_7():
    img = np.zeros((3, 12, 12), dtype=float)
    img[1, 2, 2] = 1000
    img[1, 8, 5] = 800
    mask_zero = np.zeros((3, 12, 12), dtype=np.int32)  # nothing inside a nucleus
    params = {"method": "Local Maxima", "min_distance": 2, "threshold_rel": 0.1,
              "sigma": 0.0, "nuclei_only": True}
    res = puncta.process_puncta_detection(img, mask_data=mask_zero, voxels=(1, 1, 1), params=params)
    assert res.shape == (0, 7)


def test_detection_before_filter_keeps_original_ids():
    # A spot OUTSIDE a nucleus is dropped; surviving ids must not be re-aranged
    # to 0..k-1 — they keep the position they had at detection.
    img = np.zeros((3, 16, 16), dtype=float)
    spots = [(1, 2, 2), (1, 2, 13), (1, 13, 2), (1, 13, 13)]
    for i, (z, y, x) in enumerate(spots):
        img[z, y, x] = 1000 - i  # distinct intensities for stable ordering

    # Run unfiltered to learn detection order / ids.
    params_all = {"method": "Local Maxima", "min_distance": 2, "threshold_rel": 0.1,
                  "sigma": 0.0, "nuclei_only": False}
    full = puncta.process_puncta_detection(img, mask_data=None, voxels=(1, 1, 1), params=params_all)
    n = full.shape[0]
    assert n == 4
    # Build a mask that keeps every detected spot EXCEPT the one with id 0.
    mask = np.zeros((3, 16, 16), dtype=np.int32)
    for row in full:
        z, y, x = row[:3].astype(int)
        if int(row[6]) != 0:           # drop id 0
            mask[z, y, x] = 1
    params_filt = dict(params_all, nuclei_only=True)
    filt = puncta.process_puncta_detection(img, mask_data=mask, voxels=(1, 1, 1), params=params_filt)
    surviving = set(filt[:, 6].astype(int))
    assert 0 not in surviving                      # the dropped id is gone
    assert surviving == {1, 2, 3}                  # others kept their original ids
    assert surviving != set(range(len(surviving))) # i.e. NOT re-aranged to 0..k-1


def _layer(name, coords, pids, nuc=None):
    d = {
        "name": name,
        "data": np.asarray(coords, dtype=float),
        "scale": np.ones(3),
        "translate": np.zeros(3),
        "puncta_id": np.asarray(pids),
    }
    if nuc is not None:
        d["nucleus_ids"] = np.asarray(nuc)
    return d


def test_analysis_emits_puncta_id_not_position():
    # puncta_id values are deliberately NOT 0..n-1, so positional emission would
    # be visibly wrong.
    A = _layer("A", [(0, 0, 0), (0, 0, 10)], [100, 101])
    B = _layer("B", [(0, 0, 0.1), (0, 0, 10.1)], [200, 201])

    dist = analysis.calculate_distances([A, B])
    a_rows = dist[dist["Source_Layer"] == "A"]
    assert set(a_rows["Source_ID"]) == {100, 101}
    # The A point at x=0 (id 100) is nearest the B point at x=0.1 (id 200).
    row0 = a_rows[a_rows["Source_ID"] == 100].iloc[0]
    assert int(row0["Target_ID"]) == 200

    coloc, _ = analysis.calculate_pairwise_colocalization(
        [A, B], [{"source": "A", "target": "B", "threshold": 1.0}]
    )
    assert set(coloc["ID_A"]) == {100, 101}
    assert int(coloc[coloc["ID_A"] == 100].iloc[0]["ID_B"]) == 200


def test_analysis_cross_sheet_id_consistency():
    # THE regression test for the reported bug: the same physical anchor punctum
    # must report the SAME id in Distances, Colocalization, and Tri sheets.
    A = _layer("A", [(0, 0, 0)], [100])
    B = _layer("B", [(0, 0, 0.2)], [200])
    C = _layer("C", [(0, 0, 0.3)], [300])

    dist = analysis.calculate_distances([A, B, C])
    src = dist[(dist["Source_Layer"] == "A") & (dist["Target_Layer"] == "B")].iloc[0]
    coloc, _ = analysis.calculate_pairwise_colocalization(
        [A, B, C], [{"source": "A", "target": "B", "threshold": 1.0}]
    )
    tri, _ = analysis.calculate_tri_colocalization(
        [A, B, C], [{"anchor": "A", "channel_a": "B", "channel_b": "C", "threshold": 1.0}]
    )
    assert len(tri) == 1
    assert int(src["Source_ID"]) == 100
    assert int(coloc.iloc[0]["ID_A"]) == 100
    assert int(tri.iloc[0]["Anchor_ID"]) == 100
    # All three sheets agree on the anchor's identity.
    assert int(src["Source_ID"]) == int(coloc.iloc[0]["ID_A"]) == int(tri.iloc[0]["Anchor_ID"])
    # Channel ids also stable.
    assert int(tri.iloc[0]["ChA_ID"]) == 200
    assert int(tri.iloc[0]["ChB_ID"]) == 300


def test_analysis_positional_fallback_when_no_puncta_id():
    A = {"name": "A", "data": np.array([[0.0, 0, 0], [0, 0, 10]]),
         "scale": np.ones(3), "translate": np.zeros(3)}  # no puncta_id
    B = {"name": "B", "data": np.array([[0.0, 0, 0.1], [0, 0, 10.1]]),
         "scale": np.ones(3), "translate": np.zeros(3)}
    dist = analysis.calculate_distances([A, B])
    a_rows = dist[dist["Source_Layer"] == "A"]
    # Falls back to positional indices 0..n-1.
    assert set(a_rows["Source_ID"]) == {0, 1}


def test_lookup_label_ids_respects_scale_and_translate():
    mask = np.zeros((1, 6, 6), dtype=int)
    mask[0, 3, 3] = 5
    ident, zero = (1, 1, 1), (0, 0, 0)

    # Identity frames: a point at data (0,3,3) hits the label.
    assert list(puncta.lookup_label_ids([[0, 3, 3]], ident, zero, mask, ident, zero)) == [5]

    # Non-zero points translate: data (0,0,0) + translate (0,3,3) -> world (0,3,3)
    # -> mask voxel (0,3,3) -> label 5. The OLD direct-index code (round(coords))
    # would have looked at (0,0,0) = background (0). This is the bug being fixed.
    assert list(puncta.lookup_label_ids([[0, 0, 0]], ident, (0, 3, 3), mask, ident, zero)) == [5]

    # Differing scales: points scale 0.5 -> data (0,6,6) maps to world (0,3,3).
    assert list(puncta.lookup_label_ids([[0, 6, 6]], (1, 0.5, 0.5), zero, mask, ident, zero)) == [5]

    # Out of bounds -> 0; empty input -> empty.
    assert list(puncta.lookup_label_ids([[0, 99, 99]], ident, zero, mask, ident, zero)) == [0]
    assert len(puncta.lookup_label_ids([], ident, zero, mask, ident, zero)) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
