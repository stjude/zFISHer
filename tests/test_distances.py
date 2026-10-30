"""Report sidecars, distance queries and configurable Distribution bins (plan item C1).

Export writes the report's tables as CSV beside it (``reports/distances.csv``
and ``reports/report_sheets/``); ``query_distances`` answers "how many source
puncta have a target within d µm, binned" from them, in the API, the CLI and
the GUI, without running the analysis again; the Distribution sheet's bins
can be set, with defaults that give the columns it always had.

Run:  python -m pytest tests/test_distances.py -q
"""
import logging
import os
import shutil
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pandas as pd
import pytest

from zfisher import cli, constants
from zfisher.core import analysis, api, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)

RULE = [{"source": f"R1 - {CHANNEL}", "target": f"R2 - {CHANNEL}", "threshold": 1.0}]


def test_the_default_bins_are_the_columns_the_sheet_always_had():
    assert [label for label, _, _ in analysis.distribution_bins()] == [
        "Nuclei_0_puncta", "Nuclei_1_puncta", "Nuclei_2_puncta", "Nuclei_3to10_puncta", "Nuclei_gt10_puncta"]
    assert [label for label, _, _ in analysis.distribution_bins([0, 1, 5, 20])] == [
        "Nuclei_0_puncta", "Nuclei_1to4_puncta", "Nuclei_5to19_puncta", "Nuclei_gt19_puncta"]
    with pytest.raises(ValueError, match="start at 0"):
        analysis.distribution_bins([1, 5])


def test_custom_bins_count_every_nucleus_once():
    per_nucleus = pd.DataFrame({"Nuclei_ID": [1, 2, 3, 4], "FITC": [0, 3, 7, 25]})
    table = analysis.calculate_distribution(per_nucleus, total_nuclei=6, bins=[0, 1, 5, 20])
    row = table.iloc[0]
    assert (row["Nuclei_0_puncta"], row["Nuclei_1to4_puncta"], row["Nuclei_5to19_puncta"],
            row["Nuclei_gt19_puncta"]) == (3, 1, 1, 1)                  # two nuclei with no row count as 0
    assert row[[c for c in table.columns if c.startswith("Nuclei_")]].sum() == 6


@pytest.fixture
def exported(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    s = api.open_session(out, read_only=False)
    report = api.export_report(s, rules=RULE)
    yield types.SimpleNamespace(out=out, s=s, report=report)
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_export_writes_the_tables_beside_the_report(exported):
    reports = exported.out / constants.REPORTS_DIR
    sidecar = pd.read_csv(reports / analysis.DISTANCES_SIDECAR)
    sheet = pd.read_excel(exported.report.path, sheet_name=constants.DISTANCES_SHEET)
    assert len(sidecar) == len(sheet) and list(sidecar.columns) == list(sheet.columns)
    np.testing.assert_allclose(sidecar["Distance_um"], sheet["Distance_um"])
    names = {p.name for p in (reports / analysis.SHEETS_DIR).glob("*.csv")}
    assert {"Distances.csv", "Stats.csv", "Distribution.csv", "ROI_per_Nuclei.csv"} <= names
    assert api.query_report(exported.s, constants.STATS_SHEET)


def test_the_query_counts_nearest_neighbours_within_the_distance(exported):
    q = api.query_distances(exported.s, f"R1 - {CHANNEL}", f"R2 - {CHANNEL}", 0.8, 0.2)
    table = pd.read_csv(exported.out / constants.REPORTS_DIR / analysis.DISTANCES_SIDECAR)
    rows = table[(table["Source_Layer"] == q.source) & (table["Target_Layer"] == q.target)]
    assert q.n_source == len(rows) and q.n_within == int((rows["Distance_um"] <= 0.8).sum())
    assert [b[:2] for b in q.bins] == [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8]]
    assert sum(b[2] for b in q.bins) == q.n_within and "nearest" in q.semantics
    with pytest.raises(ValueError, match="could mean"):
        api.query_distances(exported.s, CHANNEL, f"R2 - {CHANNEL}", 0.5)


def test_the_cli_and_the_gui_answer_from_the_sidecar(exported, capsys, monkeypatch):
    assert cli.main(["query-distances", str(exported.out), f"R1 - {CHANNEL}", f"R2 - {CHANNEL}",
                     "--within", "0.5"]) == 0
    assert "within 0.5 µm" in capsys.readouterr().out
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    monkeypatch.setattr(napari, "current_viewer", lambda: types.SimpleNamespace(
        status="", window=types.SimpleNamespace(_qt_window=None)))
    from zfisher.ui.widgets import colocalization_widget as cw
    session.load_session_file(next((exported.out / constants.SESSIONS_DIR).glob("zfisher_session_2.json")))
    assert set(cw._distance_layers()) == {f"Aligned R1 - {CHANNEL}_puncta", f"Warped R2 - {CHANNEL}_puncta"}
    cw._distance_query(source=f"Aligned R1 - {CHANNEL}_puncta", target=f"Warped R2 - {CHANNEL}_puncta",
                       within=0.5, bin_um=0.25)
    text = cw._distance_result.text()
    assert "puncta (" in text and "within 0.5 µm" in text and "0–0.25 µm" in text


def test_bins_from_the_template_reach_the_report(tmp_path):
    from zfisher.core import generate_batch_template as gbt
    from synthetic_rounds import write_rounds
    write_rounds(tmp_path)
    sheets = gbt.build_template_sheets()
    book = tmp_path / "b.xlsx"
    rows = [{"Name": "ok", "R1": str(tmp_path / "R1.ome.tif"), "R2": str(tmp_path / "R2.ome.tif"),
             "Distribution_Bins": "0, 2, 4"},
            {"Name": "bad", "R1": str(tmp_path / "R1.ome.tif"), "R2": str(tmp_path / "R2.ome.tif"),
             "Distribution_Bins": "1, x"}]
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=sheets["Datasets"].columns).to_excel(writer, sheet_name="Datasets", index=False)
    config, error = gbt.parse_batch_config(book)
    assert config is None and "Distribution_Bins '1, x'" in error
    rows.pop()
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        pd.DataFrame(rows, columns=sheets["Datasets"].columns).to_excel(writer, sheet_name="Datasets", index=False)
    config, error = gbt.parse_batch_config(book)
    assert error is None and config.datasets[0].distribution_bins == [0, 2, 4]
    assert gbt.pipeline_kwargs(config.datasets[0])["distribution_bins"] == [0, 2, 4]
