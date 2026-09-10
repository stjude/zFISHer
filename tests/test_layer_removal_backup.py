"""Regression test for the data-safety net: removing a layer must BACK UP its
file (rename to ``*.bak``) rather than delete it, so an accidental removal or a
crash-time teardown can't destroy a manually-curated puncta CSV.

``events`` imports napari/qtpy, so this is skipped where napari isn't installed
(the core CI environment). The matching recovery-on-load behaviour is covered
without napari in ``test_session_portability.py``.

Run (in an env with napari):  python -m pytest tests/test_layer_removal_backup.py -q
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from zfisher import constants
from zfisher.core import session


def test_layer_removal_backs_up_instead_of_deleting(tmp_path):
    from zfisher.ui import events  # imports napari

    session.clear_session()
    try:
        reports = tmp_path / constants.REPORTS_DIR
        reports.mkdir(parents=True, exist_ok=True)
        csv = reports / "R1_-_FITC_puncta.csv"
        csv.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")

        session.update_data("output_dir", str(tmp_path))
        session.set_processed_file("R1 - FITC_puncta", str(csv), layer_type="points")

        events._remove_layer_and_file("R1 - FITC_puncta")

        assert not csv.exists(), "original file should be moved, not left in place"
        assert (reports / "R1_-_FITC_puncta.csv.bak").exists(), "backup must exist"
        # Session entry is dropped on removal.
        assert "R1 - FITC_puncta" not in session.get_data("processed_files", default={})
    finally:
        session.clear_session()
