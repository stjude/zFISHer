"""The Review Queue page (``zfisher.ui.widgets.review_widget``).

Proposals an assistant logged for the open session are listed until a person
accepts or dismisses them, and the answer is logged as the person's; the
batch check lists how many datasets of a batch folder stand out.

Run:  python -m pytest tests/test_review_widget.py -q
"""
import contextlib
import os
import shutil
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from zfisher.core import api, decisions, session

from test_stages import CHANNEL, finished, reopened  # noqa: F401  (fixtures)


class _Dialog(contextlib.nullcontext):
    def __init__(self, *a, **k):
        super().__init__(self)

    def update_progress(self, *a):
        pass


@pytest.fixture
def page(reopened, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from zfisher.ui import popups
    from zfisher.ui.widgets.review_widget import ReviewQueueWidget
    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    viewer = types.SimpleNamespace(window=types.SimpleNamespace(_qt_window=None), status="")
    with decisions.use_actor("agent:test", via="mcp:pytest"):
        api.propose(session.default(), "threshold_rel", 0.08, scope={"round": "R1", "channel": CHANNEL},
                    justification="SNR is low")
        api.propose(session.default(), "remove_border_nuclei", True, scope={"stage": "consensus"})
    yield types.SimpleNamespace(widget=ReviewQueueWidget(viewer), viewer=viewer, out=reopened.out)


def test_proposals_are_listed_until_a_person_answers(page):
    w = page.widget
    assert w._proposals.count() == 2 and "2 proposal(s) waiting" in w._proposals_note.value
    first = w._proposals.item(0).text()
    assert f"R1 - {CHANNEL}: threshold_rel → 0.08" in first and "SNR is low" in first and "agent:test via mcp:pytest" in first
    w._accept_btn.native.click()                                 # nothing selected
    assert w._proposals_note.value == "Select a proposal first."
    with decisions.use_actor(decisions.HUMAN):
        w._proposals.setCurrentRow(0)
        w._note.value = "agreed"
        w._accept_btn.native.click()
        assert w._proposals.count() == 1 and "Accepted" in page.viewer.status
        w._proposals.setCurrentRow(0)
        w._dismiss_btn.native.click()
    assert w._proposals.count() == 0 and w._proposals_note.value == "No proposals are waiting."
    answers = [r for r in api.get_decisions(session.default(), "proposal_review")]
    assert [(r["status"], r["actor"], r["justification"]) for r in answers] == [
        ("accepted", "human", "agreed"), ("dismissed", "human", None)]


def test_no_open_session_says_so(page):
    session.clear_session()
    page.widget.refresh()
    assert page.widget._proposals.count() == 0 and page.widget._proposals_note.value == "No session is open."


def test_the_batch_check_counts_the_datasets(page, tmp_path):
    batch = tmp_path / "batch"
    for name in ("d1", "d2"):
        shutil.copytree(page.out, batch / name)
    page.widget.check_batch(str(batch))
    assert "2 dataset(s)" in page.widget._batch_note.value
    assert page.widget._flags.count() == len(api.flag_batch(batch, write_report=False).flags)
    assert not (batch / "qc_report.md").exists()
