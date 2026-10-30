"""The decisions log: who set which parameter, from what to what.

``decisions.jsonl`` gets one record per parameter change or stage run, from the
GUI, the batch pipeline and the core API alike. The actor is stamped by the
front end (``human`` for the GUI and the CLI, ``default`` for a batch channel
left to built-in settings, an agent's name under ``use_actor``), never taken
from a caller's argument.

Run:  python -m pytest tests/test_decisions.py -q
"""
import json
import logging
import os
import shutil

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from zfisher.core import api, decisions, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


def _log(folder):
    path = folder / decisions.DECISIONS_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_batch_run_records_every_stage_with_its_actor(tmp_path):
    from synthetic_rounds import run_batch, write_rounds
    write_rounds(tmp_path)
    levels = {n: logging.getLogger(n).level for n in ("zfisher", "stdout", "stderr")}
    try:
        with decisions.use_actor(decisions.HUMAN, user="lab-member"):
            run_batch(tmp_path, tmp_path / "out")
    finally:
        session.clear_session()
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
    log = _log(tmp_path / "out")
    names = [(e["name"], e["scope"].get("round")) for e in log]
    assert ("segmentation", "R1") in names and ("segmentation", "R2") in names
    assert {"registration", "canvas", "consensus"} <= {e["name"] for e in log}
    detection = [e for e in log if e["name"] == "detection"]
    assert {e["scope"]["round"] for e in detection} == {"R1", "R2"}
    # The synthetic batch has no Puncta sheet: nobody chose these settings.
    assert {e["actor"] for e in detection} == {decisions.DEFAULT}
    assert detection[0]["new"]["sigma"] == 1.0 and detection[0]["old"] is None
    assert {e["actor"] for e in log if e["name"] != "detection"} == {decisions.HUMAN}
    assert {e["user"] for e in log} == {"lab-member"}
    registration = next(e for e in log if e["name"] == "registration")
    assert len(registration["metrics_after"]["shift"]) == 3


@pytest.fixture
def folder(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    (out / decisions.DECISIONS_FILE).unlink(missing_ok=True)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    yield out
    session.clear_session()
    logging.setLogRecordFactory(factory)


def test_an_api_call_records_the_agent_it_was_made_for(folder):
    s = api.open_session(folder, read_only=False)
    with decisions.use_actor("agent:test-model"):
        api.detect_channel(s, "R2", CHANNEL, {"method": "Local Maxima", "sensitivity": 0.2, "min_distance": 3})
    (entry,) = api.get_decisions(s, "detection")
    assert entry["actor"] == "agent:test-model" and entry["scope"] == {"round": "R2", "channel": CHANNEL}
    assert entry["old"]["sigma"] == 1.0 and entry["new"]["sensitivity"] == 0.2
    assert entry["metrics_after"]["num_puncta"] >= 0


def test_a_read_only_session_records_nothing(folder):
    s = api.open_session(folder)
    with session.use(s):
        assert decisions.record("anything", 1) is None
    assert not (folder / decisions.DECISIONS_FILE).exists()


def test_adding_and_clearing_a_rule_in_the_gui_is_recorded(tmp_path, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui.widgets import colocalization_widget as cw
    model = ViewerModel()
    monkeypatch.setattr(napari, "current_viewer", lambda: model)
    session.clear_session()
    session.update_data("output_dir", str(tmp_path))
    a = model.add_points(np.zeros((1, 3)), name="Aligned R1 - FITC_puncta")
    b = model.add_points(np.zeros((1, 3)), name="Warped R2 - Cy5_puncta")
    decisions.set_process_actor(decisions.HUMAN)
    try:
        cw._rule_builder(source_layer=a, target_layer=b, cutoff=1.5)
        cw._on_clear_rules()
    finally:
        decisions.set_process_actor(decisions.DEFAULT)
        session.clear_session()
    added, cleared = _log(tmp_path)
    assert added["name"] == "colocalization_rules" and added["actor"] == decisions.HUMAN
    assert added["old"] == [] and added["new"][0]["threshold"] == 1.5 and added["new"][0]["source"] == a.name
    assert cleared["old"] == added["new"] and cleared["new"] == []


def test_the_actor_comes_from_the_front_end_not_a_caller():
    assert decisions.current_actor() == decisions.DEFAULT
    with decisions.use_actor("agent:x", user="someone"):
        assert (decisions.current_actor(), decisions.current_user()) == ("agent:x", "someone")
    assert decisions.current_actor() == decisions.DEFAULT and decisions.current_user() != "someone"
    import inspect
    assert "actor" not in inspect.signature(decisions.record).parameters
