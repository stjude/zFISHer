"""Running a batch's datasets one after another or in parallel (``core.batch``).

Each dataset runs through the whole pipeline into its own folder; with
``workers`` above 1 they run in separate processes. Results come back per
dataset, failures included, in the template's order; parallel runs write the
same files as sequential ones.

Run:  python -m pytest tests/test_batch_runner.py -q
"""
import hashlib
import logging
import shutil
import time

import pytest

from zfisher.core import batch, session
from zfisher.core.params import BatchConfig, DatasetSpec

from synthetic_rounds import write_rounds


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    pytest.importorskip("SimpleITK")
    root = tmp_path_factory.mktemp("rounds")
    write_rounds(root)
    return root


def _config(rounds, *names, broken=()):
    specs = [DatasetSpec(name=n, r1=rounds / ("missing.tif" if n in broken else "R1.ome.tif"),
                         r2=rounds / "R2.ome.tif") for n in names]
    return BatchConfig(datasets=specs)


@pytest.fixture(autouse=True)
def plain_log_records():
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    levels = {n: logging.getLogger(n).level for n in ("zfisher", "stdout", "stderr")}
    yield
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    session.clear_session()


def _digest(folder):
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*.csv"))}


def test_datasets_run_in_order_and_failures_are_reported(rounds, tmp_path):
    seen = []
    results = batch.run_batch(_config(rounds, "a", "b", "c", broken={"b"}), tmp_path, on_result=seen.append)
    assert [r.name for r in results] == ["a", "b", "c"] and [r.name for r in seen] == ["a", "b", "c"]
    assert [r.ok for r in results] == [True, False, True]
    assert "missing.tif" in results[1].error and results[0].stages_run[-1] == "transform R2 - FITC"
    assert (tmp_path / "c" / "reports").exists()


def test_parallel_workers_write_what_a_sequential_run_writes(rounds, tmp_path):
    start = time.monotonic()
    batch.run_batch(_config(rounds, "a", "b"), tmp_path / "seq")
    sequential = time.monotonic() - start
    start = time.monotonic()
    results = batch.run_batch(_config(rounds, "a", "b"), tmp_path / "par", workers=2)
    parallel = time.monotonic() - start
    print(f"\nsynthetic batch of 2: sequential {sequential:.1f} s, 2 workers {parallel:.1f} s")
    assert all(r.ok for r in results)
    for name in ("a", "b"):
        assert _digest(tmp_path / "seq" / name) == _digest(tmp_path / "par" / name)


def test_the_cli_runs_a_workbook_with_workers(rounds, tmp_path, capsys):
    import pandas as pd
    from zfisher import cli
    from zfisher.core import generate_batch_template as gbt
    sheets = gbt.build_template_sheets()
    book = tmp_path / "b.xlsx"
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        sheets["Instructions"].to_excel(writer, sheet_name="Instructions", index=False)
        pd.DataFrame([{"Name": n, "R1": str(rounds / "R1.ome.tif"), "R2": str(rounds / "R2.ome.tif")}
                      for n in ("x", "y")], columns=sheets["Datasets"].columns).to_excel(
            writer, sheet_name="Datasets", index=False)
    assert cli.main(["batch", str(book), "-o", str(tmp_path / "out"), "--workers", "2"]) == 0
    text = capsys.readouterr().out
    assert "2 at a time" in text and "x: OK" in text and "y: OK" in text
    shutil.rmtree(tmp_path / "out")
