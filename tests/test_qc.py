"""Which datasets of a batch need a look (``qc.flag_batch``, plan item C3).

A batch of five copies of one synthetic dataset, with outliers planted in
two: a registration that fell back to the rough shift, and a channel that lost
most of its puncta. Only those two are flagged, each for its reason; with too
few datasets the cohort is not compared at all; thresholds come from the
template's QC sheet; ``qc_report.md`` is written.

Run:  python -m pytest tests/test_qc.py -q
"""
import json
import logging
import shutil

import pandas as pd
import pytest

from zfisher import cli, constants
from zfisher.core import api, session
from zfisher.core.params import QCThresholds

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


@pytest.fixture
def batch_root(finished, tmp_path):
    for i in range(1, 6):
        shutil.copytree(finished / "out", tmp_path / f"ds{i}")
    first = tmp_path / "ds2" / constants.SESSIONS_DIR / constants.SESSION_FILENAME
    data = json.loads(first.read_text())
    data["registration"]["rigid"].update(method="rough_shift", fallback_used=True, fallback_reason="ransac_failed")
    first.write_text(json.dumps(data))
    csv = tmp_path / "ds3" / constants.REPORTS_DIR / f"Aligned_R1_-_{CHANNEL}_puncta.csv"
    table = pd.read_csv(csv)
    table.iloc[: max(1, len(table) // 10)].to_csv(csv, index=False)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    yield tmp_path
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_only_the_planted_outliers_are_flagged(batch_root):
    result = api.flag_batch(batch_root)
    by_dataset = {}
    for f in result.flags:
        by_dataset.setdefault(f["dataset"], []).append(f)
    assert set(by_dataset) == {"ds2", "ds3"} and result.n_datasets == 5
    assert any(f["metric"] == "registration.method" and "rough shift" in f["reason"] for f in by_dataset["ds2"])
    metrics = {f["metric"] for f in by_dataset["ds3"]}
    assert {f"R1 - {CHANNEL}.n_puncta", f"R1 - {CHANNEL}.puncta_per_nucleus"} <= metrics
    assert all(m.startswith(f"R1 - {CHANNEL}.") for m in metrics)         # only the channel that changed
    count = next(f for f in by_dataset["ds3"] if f["metric"] == f"R1 - {CHANNEL}.n_puncta")
    assert "low for this batch" in count["reason"]
    report = (batch_root / "qc_report.md").read_text()
    assert "| ds2 |" in report and "| ds3 |" in report and "## Batch medians" in report


def test_a_small_batch_is_not_compared_and_says_so(batch_root, capsys):
    assert api.flag_batch(batch_root, write_report=False).compared_with_batch
    assert api.flag_batch(batch_root, write_report=False).note is None
    for name in ("ds3", "ds4", "ds5"):
        shutil.rmtree(batch_root / name)
    result = api.flag_batch(batch_root, write_report=False)
    assert {f["dataset"] for f in result.flags} == {"ds2"}              # absolute rules still apply
    assert not result.compared_with_batch and "at least 4" in result.note and "Only 2 dataset" in result.note
    assert not (batch_root / "qc_report.md").exists()
    api.flag_batch(batch_root)
    assert "not compared with each other" in (batch_root / "qc_report.md").read_text()
    assert cli.main(["flag-batch", str(batch_root)]) == 0
    assert "Not compared with each other: Only 2 dataset(s)" in capsys.readouterr().out


def test_thresholds_come_from_the_template_and_the_cli_reports(batch_root, capsys):
    from zfisher.core import generate_batch_template as gbt
    sheets = gbt.build_template_sheets()
    assert list(sheets["QC"]["Setting"]) == list(QCThresholds.model_fields)
    flags = api.flag_batch(batch_root, thresholds={"z_threshold": 3.5, "min_cohort": 6}, write_report=False).flags
    assert {f["dataset"] for f in flags} == {"ds2"}                     # 5 datasets < min_cohort 6
    assert cli.main(["flag-batch", str(batch_root)]) == 0
    assert "5 datasets, 2 flagged." in capsys.readouterr().out
