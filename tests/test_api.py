"""The core API (``zfisher.core.api``): the operations every front end calls.

Run on a small batch session (``test_stages``): a read-only open writes
nothing and refuses changes; the queries report the session's state; detect,
warp and export work on a writable session and never touch the default (GUI)
session; a report rule that could mean two layers is refused.

Run:  python -m pytest tests/test_api.py -q
"""
import json
import logging
import shutil

import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import api, provenance, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


@pytest.fixture
def folder(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    session.clear_session()
    yield out
    session.clear_session()
    logging.setLogRecordFactory(factory)


def _session_files(out):
    return sorted(p.name for p in (out / constants.SESSIONS_DIR).iterdir())


def test_a_read_only_open_writes_nothing_and_refuses_changes(folder):
    before = _session_files(folder)
    s = api.open_session(folder)
    info = api.session_info(s)
    assert info.read_only and info.session_file == "zfisher_session_1.json"
    assert info.channels == [f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"]
    assert info.has_consensus_mask and info.has_saved_registration and not info.has_report
    assert info.output_dir == str(folder) and info.schema_version == 1
    with pytest.raises(PermissionError, match="read-only"):
        s.set("shift", [0, 0, 0])
    assert _session_files(folder) == before


def test_the_queries(folder):
    s = api.open_session(folder)
    states = api.get_status(s)
    assert {a.kind for a in states} == {"aligned_puncta", "report"}
    assert all(a.state == "fresh" for a in states if a.kind == "aligned_puncta")
    params = api.get_parameters(s)
    assert set(params.detection) == {f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"}
    assert params.registration["rigid"]["method"]
    checks = api.check_nucleus_ids(s)
    assert checks and all(c.fraction == 1.0 for c in checks)
    with pytest.raises(FileNotFoundError, match="no exported report"):
        api.query_report(s, "Stats")
    assert set(api.schema()) >= {"SessionInfo", "DetectionResult", "ReportResult"}


def test_writing_operations_leave_the_default_session_alone(folder):
    session.update_data("nuclear_channel", "Hoechst")              # the GUI's session, in memory
    s = api.open_session(folder, read_only=False)
    assert api.session_info(s).session_file == "zfisher_session_2.json"
    result = api.detect_channel(s, "R1", CHANNEL, {"method": "Local Maxima", "sensitivity": 0.05,
                                                   "min_distance": 2, "z_scale": 3.0}, existing="replace_all")
    assert result.n_puncta > 0 and result.n_by_source == {"auto": result.n_puncta}
    assert result.parameters["threshold_rel"] == 0.05 and result.parameters["sigma"] == 0.0
    raw = folder / constants.REPORTS_DIR / f"R1_-_{CHANNEL}_puncta.csv"
    with session.use(s):                                             # provenance is read from s's folder
        assert provenance.latest(raw)["run_by"] == "api"
    assert api.get_status(s)[0].state == "stale"                     # R1's aligned puncta are now behind
    warped = api.transform_channel(s, "R1", CHANNEL)
    assert warped.n_puncta > 0 and api.get_status(s)[0].state == "fresh"
    with pytest.raises(ValueError, match="could mean"):
        api.export_report(s, rules=[{"source": CHANNEL, "target": f"R2 - {CHANNEL}", "threshold": 1.0}])
    report = api.export_report(s, rules=[{"source": f"R1 - {CHANNEL}", "target": f"R2 - {CHANNEL}",
                                          "threshold": 1.0}])
    assert report.sheets["Colocalization"] > 0 and api.session_info(s).has_report
    rows = api.query_report(s, "Colocalization", limit=3)
    assert len(rows) == 3 and isinstance(rows[0], dict)
    assert session.get_data("nuclear_channel") == "Hoechst" and session.get_data("output_dir") is None
    saved = json.loads((folder / constants.SESSIONS_DIR / "zfisher_session_2.json").read_text())
    assert "Master_Analysis_Report" in saved["processed_files"]


def test_the_report_counts_puncta_per_nucleus_from_the_mask(folder):
    """A Nucleus_ID column scrambled on disk does not reach the report."""
    s = api.open_session(folder, read_only=False)
    rule = [{"source": f"R1 - {CHANNEL}", "target": f"R2 - {CHANNEL}", "threshold": 1.0}]
    first = pd.read_excel(api.export_report(s, rules=rule).path, sheet_name=constants.PER_NUCLEI_SHEET)
    csv = folder / constants.REPORTS_DIR / f"Aligned_R1_-_{CHANNEL}_puncta.csv"
    table = pd.read_csv(csv)
    table["Nucleus_ID"] = table["Nucleus_ID"].sample(frac=1, random_state=1).to_numpy()
    table.to_csv(csv, index=False)
    second = pd.read_excel(api.export_report(s, rules=rule).path, sheet_name=constants.PER_NUCLEI_SHEET)
    pd.testing.assert_frame_equal(first, second)
