"""Every analysis says which zFISHer produced it.

The session file records the version that created it (``created_with_zfisher``)
and the version that last saved it (``zfisher_version``); the report's Metadata
sheet has a "zFISHer Version" row. The welcome screen and Home widget show the
same string, which for anything but a tagged release includes the commit
(``2.1.0-dev+96a0ac9``), so unreleased code can be told from a release.

Run:  python -m pytest tests/test_version_records.py -q
"""
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from zfisher import version
from zfisher.core import provenance, session


def test_software_version_is_the_full_version_computed_once(monkeypatch):
    calls = []
    monkeypatch.setattr(provenance, "_software_version", None)
    monkeypatch.setattr(version, "get_full_version", lambda: (calls.append(1), "9.9.9-dev+abc1234")[1])
    assert provenance.software_version() == "9.9.9-dev+abc1234"
    assert provenance.software_version() == "9.9.9-dev+abc1234" and calls == [1]


def test_session_file_records_the_version(monkeypatch):
    monkeypatch.setattr(provenance, "_software_version", "9.9.9-dev+abc1234")
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    out = Path(tempfile.mkdtemp())
    try:
        assert session.initialize_new_session(out, out / "R1.tif", out / "R2.tif")
        files = list(out.rglob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["created_with_zfisher"] == "9.9.9-dev+abc1234"
        assert data["zfisher_version"] == "9.9.9-dev+abc1234"
        monkeypatch.setattr(provenance, "_software_version", "10.0.0")
        session.update_data("anything", 1)                     # any later save
        data = json.loads(files[0].read_text())
        assert data["zfisher_version"] == "10.0.0"
        assert data["created_with_zfisher"] == "9.9.9-dev+abc1234"
    finally:
        session.clear_session()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


def test_report_metadata_has_the_version(tmp_path, monkeypatch):
    pytest.importorskip("openpyxl")
    from zfisher.core import report
    monkeypatch.setattr(provenance, "_software_version", "9.9.9-dev+abc1234")
    path = report.export_report(pd.DataFrame({"Distance": [1.0]}), tmp_path / "r.xlsx")
    meta = pd.read_excel(path, sheet_name="Metadata")
    assert dict(zip(meta["Key"], meta["Value"]))["zFISHer Version"] == "9.9.9-dev+abc1234"
