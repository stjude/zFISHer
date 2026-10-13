"""The puncta ``Source`` column survives every path that writes or moves puncta.

``Source`` records how a punctum came to exist: ``auto`` (a detector),
``manual`` (placed by hand), or ``unknown`` (never recorded, as in CSVs from
before the column was kept). It used to be reset to ``auto`` by the aligned-space
transform, dropped by the GUI layer sink (which also renumbered ``puncta_id``),
and lost with every manual punctum when detection was re-run with Replace. The
core tests run without napari; the GUI tests use a headless ``ViewerModel``.

Run:  python -m pytest tests/test_puncta_source.py -q
"""
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import puncta

AUTO, MANUAL, UNKNOWN = (constants.PUNCTA_SOURCE_AUTO, constants.PUNCTA_SOURCE_MANUAL,
                         constants.PUNCTA_SOURCE_UNKNOWN)


# --- core -------------------------------------------------------------------

def test_table_is_canonical_and_marks_missing_sources_unknown():
    coords = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], float)
    feats = pd.DataFrame({"extra": [1, 2, 3], "SNR": [1.0, 2.0, 3.0], "puncta_id": [5, 6, 7],
                          "Source": [MANUAL, None, ""]})
    table = puncta.puncta_table(coords, feats)
    assert list(table.columns) == ["puncta_id", "Z", "Y", "X", "SNR", "Source", "extra"]
    assert list(table["Source"]) == [MANUAL, UNKNOWN, UNKNOWN]
    no_column = puncta.puncta_table(coords, feats.drop(columns="Source"))
    assert list(no_column["Source"]) == [UNKNOWN] * 3
    assert list(puncta.puncta_table(coords, None).columns) == ["Z", "Y", "X", "Source"]


def _raw(n, ids):
    coords = np.column_stack([np.arange(n), np.arange(n) * 2, np.arange(n) * 3]).astype(float)
    return np.column_stack([coords, np.zeros(n), np.full(n, 100.0), np.full(n, 5.0)]), np.asarray(ids)


def test_transform_carries_sources_through_the_nucleus_filter(tmp_path, monkeypatch):
    from zfisher.core import session
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    raw, ids = _raw(4, [10, 11, 12, 13])
    sources = np.array([MANUAL, AUTO, MANUAL, UNKNOWN], dtype=object)
    mask = np.zeros((10, 10, 12), np.int32)
    for z, y, x in raw[[0, 2, 3], :3].astype(int):      # row 1 lies outside every nucleus
        mask[z, y, x] = 1
    out = tmp_path / "a.csv"
    data = puncta.transform_puncta_to_aligned_space(
        raw, "R1", shift=np.zeros(3), canvas_offset=np.zeros(3), consensus_mask=mask,
        output_path=out, layer_name="a", puncta_ids=ids, sources=sources)
    assert list(data[:, 6]) == [10, 12, 13]
    csv = pd.read_csv(out)
    assert list(csv["puncta_id"]) == [10, 12, 13]
    assert list(csv["Source"]) == [MANUAL, MANUAL, UNKNOWN]
    assert list(puncta.sources_of(data, ids, sources)) == [MANUAL, MANUAL, UNKNOWN]
    with pytest.raises(ValueError):
        puncta.transform_puncta_to_aligned_space(
            raw, "R1", np.zeros(3), np.zeros(3), puncta_ids=ids, sources=sources[:2])


def test_detection_rows_are_auto(tmp_path, monkeypatch):
    from zfisher.core import session
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    raw, ids = _raw(3, [0, 1, 2])
    out = tmp_path / "b.csv"
    puncta._save_puncta_csv(np.column_stack([raw, ids]), out, "b")
    assert list(pd.read_csv(out)["Source"]) == [AUTO] * 3


def test_sources_of_an_unknown_id_is_unknown():
    data = np.zeros((2, 7))
    data[:, 6] = [3, 99]
    assert list(puncta.sources_of(data, [3], [MANUAL])) == [MANUAL, UNKNOWN]


# --- GUI sink, loader and autosave -------------------------------------------

@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(_qapp, monkeypatch):
    from napari.components import ViewerModel
    from zfisher.core import session
    from zfisher.ui import viewer_helpers
    tmp = Path(tempfile.mkdtemp())
    monkeypatch.setattr(session, "get_data",
                        lambda key=None, default=None: (str(tmp) if key == "output_dir" else default))
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    viewer = ViewerModel()
    source = types.SimpleNamespace(name="R1 - FITC", scale=(1, 1, 1), translate=(0, 0, 0))
    name = f"R1 - FITC{constants.PUNCTA_SUFFIX}"
    return types.SimpleNamespace(viewer=viewer, helpers=viewer_helpers, source=source, name=name,
                                 csv=constants.puncta_csv_path(tmp / constants.REPORTS_DIR, name))


def _rows(coords, ids):
    coords = np.asarray(coords, float)
    n = len(coords)
    return np.column_stack([coords, np.ones(n), np.full(n, 50.0), np.full(n, 4.0), np.asarray(ids, float)])


def _curated_layer(env):
    """A layer of three detected puncta (ids 0-2) and two placed by hand (7, 8)."""
    env.helpers.add_or_update_puncta_layer(
        env.viewer, env.source, _rows([[1, 1, 1], [1, 20, 20], [1, 40, 40], [2, 60, 60], [2, 80, 80]],
                                      [0, 1, 2, 7, 8]),
        sources=[AUTO, AUTO, AUTO, MANUAL, MANUAL])
    return env.viewer.layers[env.name]


def test_new_layer_keeps_puncta_id_and_source(env):
    layer = _curated_layer(env)
    assert list(layer.features["puncta_id"]) == [0, 1, 2, 7, 8]       # not renumbered
    assert list(layer.features["Source"]) == [AUTO] * 3 + [MANUAL] * 2
    csv = pd.read_csv(env.csv)
    assert list(csv.columns[:8]) == constants.PUNCTA_CSV_COLUMNS
    assert list(csv["Source"]) == [AUTO] * 3 + [MANUAL] * 2


def test_replace_keeps_manual_puncta(env):
    layer = _curated_layer(env)
    # A re-detection: one point lands on manual punctum 7, two are new.
    detected = _rows([[2, 61, 60], [1, 10, 10], [1, 30, 30]], [0, 1, 2])
    env.helpers.add_or_update_puncta_layer(env.viewer, env.source, detected, existing_action="replace")
    feats = layer.features
    assert list(feats["Source"]) == [MANUAL, MANUAL, AUTO, AUTO]
    assert list(feats["puncta_id"][:2]) == [7, 8]                      # manual identity kept
    assert feats["puncta_id"].is_unique
    np.testing.assert_array_equal(layer.data[:2], [[2, 60, 60], [2, 80, 80]])
    assert list(pd.read_csv(env.csv)["Source"]) == [MANUAL, MANUAL, AUTO, AUTO]


def test_replace_all_drops_manual_puncta(env):
    layer = _curated_layer(env)
    env.helpers.add_or_update_puncta_layer(
        env.viewer, env.source, _rows([[1, 10, 10]], [0]), existing_action="replace_all")
    assert list(layer.features["Source"]) == [AUTO]


def test_merge_marks_new_rows_auto_and_keeps_manual(env):
    layer = _curated_layer(env)
    env.helpers.add_or_update_puncta_layer(
        env.viewer, env.source, _rows([[3, 90, 90]], [0]), existing_action="merge")
    assert list(layer.features["Source"]) == [AUTO] * 3 + [MANUAL] * 2 + [AUTO]
    assert list(pd.read_csv(env.csv)["Source"])[-1] == AUTO


def test_merge_with_nothing_new_leaves_the_file_alone(env):
    """Detection no longer writes the CSV first, so an early return keeps the
    curated file intact."""
    _curated_layer(env)
    before = env.csv.read_text()
    env.helpers.add_or_update_puncta_layer(
        env.viewer, env.source, _rows([[2, 60, 60]], [0]), existing_action="merge")
    assert env.csv.read_text() == before


def test_aligned_space_transform_keeps_manual_puncta(env):
    """The canvas step: raw layer -> transform -> aligned layer and CSV."""
    raw_layer = _curated_layer(env)
    raw, pid, sources = env.helpers.raw_puncta_from_layer(raw_layer)
    out = env.csv.with_name("Aligned.csv")
    transformed = puncta.transform_puncta_to_aligned_space(
        raw, "R1", np.zeros(3), np.array([-1.0, 0, 0]), output_path=out, layer_name="x",
        puncta_ids=pid, sources=sources)
    aligned = types.SimpleNamespace(name="Aligned R1 - FITC", scale=(1, 1, 1), translate=(0, 0, 0))
    env.helpers.add_or_update_puncta_layer(env.viewer, aligned, transformed,
                                           sources=puncta.sources_of(transformed, pid, sources))
    layer = env.viewer.layers[f"Aligned R1 - FITC{constants.PUNCTA_SUFFIX}"]
    assert list(layer.features["puncta_id"]) == [0, 1, 2, 7, 8]
    assert list(layer.features["Source"]) == [AUTO] * 3 + [MANUAL] * 2
    assert list(pd.read_csv(out)["Source"]) == [AUTO] * 3 + [MANUAL] * 2


def test_legacy_csv_without_source_loads_as_unknown(env, caplog):
    path = env.csv.parent / "legacy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"puncta_id": [0, 1], "Z": [1, 2], "Y": [3, 4], "X": [5, 6]}).to_csv(path, index=False)
    with caplog.at_level("WARNING"):
        env.helpers._load_points_layer(env.viewer, "legacy_puncta", path, (1, 1, 1), {}, (0, 0, 0))
    layer = env.viewer.layers["legacy_puncta"]
    assert list(layer.features["Source"]) == [UNKNOWN, UNKNOWN]
    assert any("no Source column" in r.getMessage() for r in caplog.records)


def test_autosave_writes_the_canonical_table(env):
    from zfisher.ui import events
    layer = _curated_layer(env)
    layer.features = layer.features.drop(columns="Source")
    events.save_puncta_layer(layer)
    csv = pd.read_csv(env.csv)
    assert list(csv.columns[:8]) == constants.PUNCTA_CSV_COLUMNS
    assert list(csv["Source"]) == [UNKNOWN] * 5


@pytest.mark.parametrize("ids", [[0, 1, 2, np.nan, 8], [0, 1, 2, np.nan, np.nan]])
def test_replace_gives_kept_manual_puncta_without_an_id_one(env, ids):
    layer = _curated_layer(env)
    feats = layer.features.copy()
    feats["puncta_id"] = ids
    layer.features = feats
    env.helpers.add_or_update_puncta_layer(
        env.viewer, env.source, _rows([[1, 10, 10], [1, 30, 30]], [0, 1]), existing_action="replace")
    out = layer.features["puncta_id"]
    assert out.notna().all() and out.is_unique and len(out) == 4
    if not np.isnan(ids[-1]):
        assert out[1] == 8                          # a kept id is not changed
    assert list(layer.features["Source"]) == [MANUAL, MANUAL, AUTO, AUTO]
    assert pd.read_csv(env.csv)["puncta_id"].notna().all()
