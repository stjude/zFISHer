"""Headless tests for core analysis math (no napari/GUI).

Covers the per-channel summary statistics, the per-nucleus distribution bins
(including zero-padding to the consensus nucleus count), and the greedy
tri-colocalization matcher. These lock in current behavior so the planned
refactors/correctness fixes can be made safely.

Run:  pytest tests/test_analysis.py -q   (or: python tests/test_analysis.py)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher.core import analysis


def _per_nucleus(channel_counts, nuclei_ids):
    """Build a per-nucleus pivot-like frame: Nuclei_ID + one column per channel."""
    data = {"Nuclei_ID": list(nuclei_ids)}
    data.update(channel_counts)
    return pd.DataFrame(data)


def test_calculate_stats_mean_std_cv():
    df = _per_nucleus({"FITC": [2, 4, 6]}, [1, 2, 3])
    out = analysis.calculate_stats(df, total_nuclei=3)
    row = out[out["Channel"] == "FITC"].iloc[0]
    assert row["Total_Nuclei"] == 3
    assert row["Raw_Sum"] == 12
    assert abs(row["Mean_per_Nucleus"] - 4.0) < 1e-9          # 12 / 3
    # population std around mean 4 of [2,4,6]: sqrt(8/3)
    assert abs(row["StdDev_per_Nucleus"] - (8 / 3) ** 0.5) < 1e-3
    assert abs(row["CV_pct"] - ((8 / 3) ** 0.5 / 4 * 100)) < 1e-1


def test_calculate_stats_pads_to_total_nuclei():
    # Only 2 nuclei present, but the consensus mask has 4 → mean divides by 4
    # and the std is computed over [3, 5, 0, 0].
    df = _per_nucleus({"FITC": [3, 5]}, [1, 2])
    row = analysis.calculate_stats(df, total_nuclei=4).iloc[0]
    assert row["Raw_Sum"] == 8
    assert abs(row["Mean_per_Nucleus"] - 2.0) < 1e-9          # 8 / 4
    assert abs(row["StdDev_per_Nucleus"] - 4.5 ** 0.5) < 1e-3  # var = 18/4


def test_calculate_distribution_bins_and_zero_padding():
    df = _per_nucleus({"FITC": [0, 1, 2, 5, 12]}, [1, 2, 3, 4, 5])
    row = analysis.calculate_distribution(df, total_nuclei=7).iloc[0]
    # 5 present nuclei + 2 padded (7 - 5) → the 2 padded count as 0-puncta.
    assert row["Nuclei_0_puncta"] == 1 + 2
    assert row["Nuclei_1_puncta"] == 1
    assert row["Nuclei_2_puncta"] == 1
    assert row["Nuclei_3to10_puncta"] == 1   # the 5
    assert row["Nuclei_gt10_puncta"] == 1    # the 12


def test_calculate_stats_reports_extranuclear_and_total_puncta():
    df = _per_nucleus({"FITC": [2, 4, 6]}, [1, 2, 3])
    row = analysis.calculate_stats(df, total_nuclei=3, extranuclear={"FITC": 5}).iloc[0]
    assert row["Raw_Sum"] == 12
    assert row["Extranuclear_Puncta"] == 5
    assert row["Total_Puncta"] == 17
    assert abs(row["Mean_per_Nucleus"] - 4.0) < 1e-9   # extranuclear do not enter the mean
    # A channel missing from the dict counts 0.
    row0 = analysis.calculate_stats(df, total_nuclei=3, extranuclear={}).iloc[0]
    assert row0["Extranuclear_Puncta"] == 0 and row0["Total_Puncta"] == 12


def test_calculate_distribution_bins_sum_to_total_nuclei():
    df = _per_nucleus({"FITC": [0, 1, 2, 5, 12]}, [1, 2, 3, 4, 5])
    row = analysis.calculate_distribution(df, total_nuclei=7).iloc[0]
    bins = ["Nuclei_0_puncta", "Nuclei_1_puncta", "Nuclei_2_puncta",
            "Nuclei_3to10_puncta", "Nuclei_gt10_puncta"]
    assert row["Total_Nuclei"] == 7
    assert sum(int(row[b]) for b in bins) == 7


def _id_layer(name, nucleus_ids):
    n = len(nucleus_ids)
    return {"name": name, "data": np.zeros((n, 3)), "scale": np.ones(3),
            "translate": np.zeros(3), "nucleus_ids": np.asarray(nucleus_ids)}


def _no_nuclear_channel(fn):
    """Run fn with session.get_nuclear_channel returning a fixed name."""
    from zfisher.core import session
    orig = session.get_nuclear_channel
    session.get_nuclear_channel = lambda: "DAPI"
    try:
        return fn()
    finally:
        session.get_nuclear_channel = orig


def test_extranuclear_counts_per_channel_skips_non_puncta_layers():
    layers = [
        _id_layer("Aligned R1 - FITC_puncta", [1, 0, 2, 0, 0]),
        _id_layer("Warped R2 - Cy5_puncta", [3, 3]),
        _id_layer("Aligned R1 - DAPI_centroids", [0, 0]),   # skipped: nuclear + centroids
        {"name": "Aligned R1 - TRITC_puncta", "data": np.zeros((2, 3)),
         "scale": np.ones(3), "translate": np.zeros(3)},     # no nucleus_ids: skipped
    ]
    counts = _no_nuclear_channel(lambda: analysis.extranuclear_counts(layers))
    assert counts == {"Aligned R1 - FITC_puncta": 3, "Warped R2 - Cy5_puncta": 0}


def test_per_nucleus_counts_with_mask_labels_lists_every_nucleus():
    layers = [_id_layer("Aligned R1 - FITC_puncta", [1, 1, 3, 0])]
    # Mask has nuclei 1, 2, 3, 4; nucleus 2 and 4 have no puncta; ID 0 is excluded.
    pivot = _no_nuclear_channel(
        lambda: analysis.calculate_per_nucleus_counts(layers, nucleus_ids=[1, 2, 3, 4]))
    assert list(pivot["Nuclei_ID"]) == [1, 2, 3, 4]
    assert list(pivot["Aligned R1 - FITC_puncta"]) == [2, 0, 1, 0]
    # Without the label list, only nuclei with puncta appear (legacy behaviour).
    legacy = _no_nuclear_channel(lambda: analysis.calculate_per_nucleus_counts(layers))
    assert list(legacy["Nuclei_ID"]) == [1, 3]


def test_per_nucleus_counts_drops_ids_not_in_mask():
    # Nucleus 9 was deleted from the mask after assignment: its puncta are left out.
    layers = [_id_layer("Aligned R1 - FITC_puncta", [1, 9, 9, 2])]
    pivot = _no_nuclear_channel(
        lambda: analysis.calculate_per_nucleus_counts(layers, nucleus_ids=[1, 2]))
    assert list(pivot["Nuclei_ID"]) == [1, 2]
    assert 9 not in set(pivot["Nuclei_ID"])
    assert int(pivot["Aligned R1 - FITC_puncta"].sum()) == 2
    stats = analysis.calculate_stats(pivot, total_nuclei=2).iloc[0]
    assert stats["Raw_Sum"] == 2 and stats["Total_Nuclei"] == 2


def _layer(name, coords):
    return {"name": name, "data": np.asarray(coords, dtype=float),
            "scale": np.ones(3), "translate": np.zeros(3)}


def test_tri_colocalization_greedy_consumes_each_channel_once():
    anchor = _layer("A", [(0, 0, 0.0)])
    ch_a = _layer("B", [(0, 0, 0.1), (0, 0, 0.2)])
    ch_b = _layer("C", [(0, 0, 0.15), (0, 0, 0.25)])
    rules = [{"anchor": "A", "channel_a": "B", "channel_b": "C", "threshold": 1.0}]
    tri, _ = analysis.calculate_tri_colocalization([anchor, ch_a, ch_b], rules)

    assert len(tri) == 2                       # min(2 ChA, 2 ChB)
    # Each ChA and ChB punctum is used in exactly one triplet (greedy, no reuse).
    assert sorted(tri["ChA_ID"]) == [0, 1]
    assert sorted(tri["ChB_ID"]) == [0, 1]
    # Best (closest-sum) pairing is picked first: chA0+chB0 then chA1+chB1.
    first = tri.iloc[0]
    assert int(first["ChA_ID"]) == 0 and int(first["ChB_ID"]) == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
