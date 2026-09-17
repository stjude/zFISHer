"""Headless tests for report export naming (no napari/GUI).

``export_report`` used to refuse to write when the target file existed, which
left stale reports in place and made batch re-runs into the same folder fail.
It now writes a numbered copy by default; overwrite and error remain available.

Run:  pytest tests/test_report_export.py -q   (or: python tests/test_report_export.py)
"""
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher.core import report, session


def _with_stub_session(fn):
    orig = session.get_data
    session.get_data = lambda key=None, default=None: default
    try:
        return fn()
    finally:
        session.get_data = orig


def _df():
    return pd.DataFrame({"Source_Layer": ["A"], "Target_Layer": ["B"], "Distance_um": [0.5]})


def test_next_free_path_numbers_from_two():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.xlsx"
        assert report.next_free_path(p) == p
        p.write_bytes(b"")
        assert report.next_free_path(p) == Path(d) / "r_2.xlsx"
        (Path(d) / "r_2.xlsx").write_bytes(b"")
        assert report.next_free_path(p) == Path(d) / "r_3.xlsx"


def test_export_versions_existing_report_by_default():
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "report.xlsx"
        p1 = _with_stub_session(lambda: report.export_report(_df(), target))
        p2 = _with_stub_session(lambda: report.export_report(_df(), target))
        p3 = _with_stub_session(lambda: report.export_report(_df(), target))
        assert p1 == target and p1.exists()
        assert p2 == Path(d) / "report_2.xlsx" and p2.exists()
        assert p3 == Path(d) / "report_3.xlsx" and p3.exists()


def test_export_overwrite_and_error_modes():
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "report.xlsx"
        _with_stub_session(lambda: report.export_report(_df(), target))
        first_size = target.stat().st_size
        bigger = pd.concat([_df()] * 50, ignore_index=True)
        p = _with_stub_session(lambda: report.export_report(bigger, target, on_exists="overwrite"))
        assert p == target
        assert target.stat().st_size != first_size
        assert not (Path(d) / "report_2.xlsx").exists()
        raised = False
        try:
            _with_stub_session(lambda: report.export_report(_df(), target, on_exists="error"))
        except FileExistsError:
            raised = True
        assert raised


def test_export_rejects_unknown_mode():
    with tempfile.TemporaryDirectory() as d:
        raised = False
        try:
            report.export_report(_df(), Path(d) / "x.xlsx", on_exists="maybe")
        except ValueError:
            raised = True
        assert raised


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
