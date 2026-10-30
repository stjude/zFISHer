"""Nucleus IDs read from the saved mask, for checks, repair and the report.

The 19 Oct 2026 lab report: puncta CSVs whose Nucleus_ID column no longer
matched the saved consensus mask, while every other column, coordinates and
intensity included, was right, and zFISHer's own check said nothing changed.
``puncta.check_nucleus_ids`` and ``repair_nucleus_ids`` read each punctum's
nucleus from the mask on disk by the rule the GUI uses; ``zfisher-cli
nucleus-ids`` runs them over an output folder, writing repaired copies beside
the originals; export refuses to go out beside puncta files that contradict
its consensus mask.

Run:  python -m pytest tests/test_nucleus_ids_from_files.py -q
"""
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from zfisher import cli, constants
from zfisher.core import io, puncta

from test_stages import finished, reopened  # noqa: F401  (fixtures)


def _mask():
    mask = np.zeros((4, 30, 30), np.int32)
    mask[:, 2:12, 2:12], mask[:, 15:28, 2:12], mask[:, 2:12, 15:28] = 1, 2, 3
    return mask


def _scrambled_csv(path):
    """The note's pattern: right coordinates and intensities, IDs of other rows."""
    coords = np.array([[1.2, 5.4, 6.0], [2.0, 20.6, 5.0], [0.0, 6.0, 20.4], [3.0, 7.0, 7.0], [1.0, 14.0, 14.0]])
    right = [1, 2, 3, 1, 0]
    features = pd.DataFrame({"puncta_id": range(5), "Nucleus_ID": [3, 1, 2, 2, 1],
                             "Intensity": [10.0, 20.0, 30.0, 40.0, 50.0], "SNR": 1.0, "Source": "auto"})
    puncta.write_puncta_csv(path, coords, features)
    return right


def test_a_scrambled_column_is_found_and_repaired_beside_the_original(tmp_path):
    csv, out = tmp_path / "a.csv", tmp_path / "a.repaired.csv"
    right = _scrambled_csv(csv)
    before = csv.read_bytes()

    check = puncta.check_nucleus_ids(csv, _mask())
    assert (check["rows"], check["agree"], check["zero_but_inside"]) == (5, 0, 0)
    puncta.repair_nucleus_ids(csv, _mask(), out)

    assert csv.read_bytes() == before
    repaired, original = pd.read_csv(out), pd.read_csv(csv)
    assert list(repaired["Nucleus_ID"]) == right
    pd.testing.assert_frame_equal(repaired.drop(columns="Nucleus_ID"), original.drop(columns="Nucleus_ID"))
    assert puncta.check_nucleus_ids(out, _mask())["fraction"] == 1.0


def test_the_rule_is_the_nearest_voxel_as_in_the_gui():
    mask = _mask()
    coords = np.array([[1.0, 11.6, 5.0], [1.0, 11.4, 5.0]])      # 11.6 rounds out of nucleus 1, 11.4 stays
    assert list(puncta.nucleus_ids_from_mask(coords, mask)) == [0, 1]
    np.testing.assert_array_equal(puncta.nucleus_ids_from_mask(coords, mask),
                                  puncta.lookup_label_ids(coords, (0.3, 0.1, 0.1), (0, 2, 3), mask,
                                                          (0.3, 0.1, 0.1), (0, 2, 3)))


def test_the_cli_checks_a_batch_folder_and_repairs_a_scrambled_file(reopened, capsys):
    assert cli.main(["nucleus-ids", str(reopened.out)]) == 0      # what batch writes agrees with its mask
    first = puncta.aligned_puncta_csvs(reopened.reports)[0]
    table = pd.read_csv(first)
    table["Nucleus_ID"] = np.roll(table["Nucleus_ID"].to_numpy(), 1)
    table.to_csv(first, index=False)
    before = first.read_bytes()

    assert cli.main(["nucleus-ids", str(reopened.out)]) == 2
    assert cli.main(["nucleus-ids", str(reopened.out), "--write"]) == 0

    out = capsys.readouterr().out
    assert f"{first.name}:" in out and "Originals are unchanged" in out
    assert first.read_bytes() == before
    mask = io.read_label_tif(io.find_mask(reopened.out / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME))
    repaired = first.with_name(first.stem + ".nucleus_ids_repaired.csv")
    assert puncta.check_nucleus_ids(repaired, mask)["fraction"] == 1.0


@pytest.fixture
def export(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.core import session
    from zfisher.ui import events, viewer_helpers
    data = {"output_dir": str(tmp_path), "processed_files": {}}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "set_processed_file", lambda *a, **k: None)
    model = ViewerModel()
    monkeypatch.setattr(napari, "current_viewer", lambda: model)
    consensus = model.add_labels(_mask(), name=constants.CONSENSUS_MASKS_NAME)
    name = "Aligned R1 - FITC_puncta"
    csv = constants.puncta_csv_path(tmp_path / constants.REPORTS_DIR, name)
    csv.parent.mkdir()
    right = _scrambled_csv(csv)                                    # the file on disk is stale ...
    table = pd.read_csv(csv).assign(Nucleus_ID=right)
    layer = model.add_points(table[["Z", "Y", "X"]].to_numpy(float), name=name,
                             features=table.drop(columns=["Z", "Y", "X"]))   # ... the layer is right
    yield types.SimpleNamespace(helpers=viewer_helpers, events=events, layer=layer, consensus=consensus,
                                csv=csv, right=right)


def test_export_saves_a_lagging_file_again(export):
    export.helpers.verify_puncta_files_for_report([export.layer], export.consensus)
    assert list(pd.read_csv(export.csv)["Nucleus_ID"]) == export.right


def test_export_stops_when_a_file_still_disagrees(export, monkeypatch):
    monkeypatch.setattr(export.events, "save_puncta_layer", lambda layer, report=True: False)
    with pytest.raises(ValueError, match="5 of 5 nucleus IDs differ from the consensus mask"):
        export.helpers.verify_puncta_files_for_report([export.layer], export.consensus)
