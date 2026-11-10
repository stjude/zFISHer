"""Planning a pipeline run: ``pipeline.plan_full_zfisher_pipeline`` names the
stages a run with given settings would execute in a finished folder, without
running any or writing anything.

Each test changes one setting (or one file) of a copy of a small synthetic
batch session, asks for the plan, then runs the pipeline with the same
settings and checks the stages it ran are the ones planned.

Run:  python -m pytest tests/test_pipeline_plan.py -q
"""
import json
import logging
import shutil

import pytest

from zfisher import constants
from zfisher.core import pipeline, session

from test_stages import CHANNEL, finished  # noqa: F401  (fixture)


@pytest.fixture
def run_folder(finished, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    data = json.loads(sorted((out / constants.SESSIONS_DIR).glob("zfisher_session_*.json"))[-1].read_text())
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)       # see synthetic_rounds.run_batch
    yield out, data["r1_path"], data["r2_path"], dict(data["pipeline_settings"])
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    session.clear_session()


def _files(folder):
    return {p.relative_to(folder).as_posix(): p.stat().st_mtime_ns for p in folder.rglob("*") if p.is_file()}


def _plan_then_run(run_folder, **changes):
    out, r1, r2, settings = run_folder
    settings.update(changes)
    before = _files(out)
    planned = pipeline.plan_full_zfisher_pipeline(r1, r2, out, **settings)
    assert _files(out) == before                       # planning writes nothing
    ran = pipeline.run_full_zfisher_pipeline(r1, r2, out, **settings)
    return planned, ran


def test_nothing_changed_plans_nothing(run_folder):
    assert _plan_then_run(run_folder) == ([], [])


def test_new_detection_settings_redetect_and_rewarp_that_channel(run_folder):
    config = {CHANNEL: {"method": "Local Maxima", "threshold_rel": 0.2, "min_distance": 2, "nuclei_only": True}}
    planned, ran = _plan_then_run(run_folder, puncta_config=config)
    assert planned == ran == [f"detect R1 - {CHANNEL}", f"detect R2 - {CHANNEL}",
                              f"transform R1 - {CHANNEL}", f"transform R2 - {CHANNEL}"]


def test_a_detection_limited_to_one_round(run_folder):
    config = {CHANNEL: {"method": "Local Maxima", "threshold_rel": 0.2, "min_distance": 2, "round": "R2"}}
    planned, ran = _plan_then_run(run_folder, puncta_config=config)
    assert planned == ran == [f"detect R2 - {CHANNEL}", f"transform R2 - {CHANNEL}"]


def test_a_registration_setting_reruns_everything_after_it(run_folder):
    planned, ran = _plan_then_run(run_folder, max_ransac_distance=7)
    assert planned == ran
    assert planned[:3] == ["register", "canvas", "consensus"] and len(planned) == 5


def test_consensus_and_transform_settings(run_folder):
    planned, ran = _plan_then_run(run_folder, overlap_method="Union")
    assert planned == ran == ["consensus", f"transform R1 - {CHANNEL}", f"transform R2 - {CHANNEL}"]


def test_a_first_report_and_then_new_bins(run_folder):
    rules = [{"source": f"R1 - {CHANNEL}", "target": f"R2 - {CHANNEL}", "threshold": 1.0}]
    assert _plan_then_run(run_folder, pairwise_rules=rules) == (["report"], ["report"])
    out, r1, r2, settings = run_folder
    settings["pairwise_rules"] = rules
    assert _plan_then_run((out, r1, r2, settings), distribution_bins=[0, 2, 5]) == (["report"], ["report"])


def test_a_deleted_output_is_made_again(run_folder):
    out = run_folder[0]
    (out / constants.REPORTS_DIR / f"Aligned_R1_-_{CHANNEL}_puncta.csv").unlink()
    planned, ran = _plan_then_run(run_folder)
    assert planned == ran == [f"transform R1 - {CHANNEL}"]


def test_a_folder_without_a_run_cannot_be_planned(tmp_path, run_folder):
    _, r1, r2, settings = run_folder
    with pytest.raises(LookupError, match="every stage"):
        pipeline.plan_full_zfisher_pipeline(r1, r2, tmp_path / "empty", **settings)


# --- Changing a setting for the next update (api.set_param) ---------------------------

from zfisher.core import api, decisions  # noqa: E402


def _session_files(out):
    return sorted(p.name for p in (out / constants.SESSIONS_DIR).iterdir())


def test_a_dry_run_says_what_would_run_and_changes_nothing(run_folder):
    out = run_folder[0]
    before = _files(out)
    change = api.set_param(api.open_session(out), "detection", "sensitivity", 0.2, channel=CHANNEL, dry_run=True)
    assert (change.name, change.old, change.new, change.applied) == ("threshold_rel", 0.05, 0.2, False)
    assert change.rounds == ["R1", "R2"] and not change.unchanged      # one setting for both rounds
    assert change.would_run == [f"detect R1 - {CHANNEL}", f"detect R2 - {CHANNEL}",
                                f"transform R1 - {CHANNEL}", f"transform R2 - {CHANNEL}"]
    assert _files(out) == before


def test_a_change_is_stored_and_logged_and_update_runs_what_was_planned(run_folder):
    out = run_folder[0]
    s = api.open_session(out, read_only=False)
    with decisions.use_actor("agent:test", via="mcp:pytest"):
        change = api.set_param(s, "registration", "max_ransac_distance", 7, justification="few inliers")
    assert change.applied and (change.old, change.new) == (0, 7)
    assert s.get("pipeline_settings")["max_ransac_distance"] == 7
    entry = api.get_decisions(s, "max_ransac_distance")[-1]
    assert (entry["actor"], entry["via"], entry["old"], entry["new"], entry["justification"]) == (
        "agent:test", "mcp:pytest", 0, 7, "few inliers")
    assert entry["scope"] == {"stage": "registration"}
    assert api.update(s) == change.would_run


def test_values_outside_a_settings_range_or_choices_are_refused(run_folder):
    s = api.open_session(run_folder[0])
    for scope, name, value, channel, message in (
            ("registration", "max_ransac_distance", 500, None, "less than or equal"),
            ("consensus", "overlap_method", "Sideways", None, "Intersection"),
            ("detection", "sensitivity", 2.0, CHANNEL, "less than or equal"),
            ("detection", "min_distance", 2, None, "needs the channel"),
            ("detection", "min_distance", 2, "Cy7", "channels are FITC"),
            ("detection", "brightness", 2, CHANNEL, "No detection setting 'brightness'"),
            ("registration", "warp_harder", True, None, "they are max_ransac_distance, apply_warp"),
            ("report", "distribution_bins", [1, 5], None, "start at 0"),
            ("cosmic", "x", 1, None, "scopes are segmentation")):
        with pytest.raises(ValueError, match=message):
            api.set_param(s, scope, name, value, channel=channel, dry_run=True)


def test_writing_back_the_built_in_detection_settings_changes_nothing(run_folder):
    s = api.open_session(run_folder[0], read_only=False)
    for name, value in (("min_distance", 2), ("z_edge_exclude", 0)):
        change = api.set_param(s, "detection", name, value, channel=CHANNEL)
        assert not change.applied and change.unchanged and change.would_run == []
    assert api.get_decisions(s, "z_edge_exclude") == []


def test_what_an_operation_would_write_and_leave_out_of_date(run_folder):
    out = run_folder[0]
    s = api.open_session(out)
    reports = out / constants.REPORTS_DIR
    detect = api.plan_operation(s, "detect", "R1", CHANNEL)
    assert detect.writes == [str(reports / f"R1_-_{CHANNEL}_puncta.csv")]
    assert detect.makes_stale == [str(reports / f"Aligned_R1_-_{CHANNEL}_puncta.csv")]   # no report yet
    assert api.plan_operation(s, "transform", "R2", CHANNEL).writes[0].endswith(f"R2_-_{CHANNEL}_puncta.csv")
    report = api.plan_operation(s, "report")
    assert report.writes[0].endswith("Master_Analysis_Report.xlsx") and report.makes_stale == []
    with pytest.raises(ValueError, match="which round and channel"):
        api.plan_operation(s, "detect")


def test_a_channel_may_name_its_round_and_must_exist_to_plan_it(run_folder):
    s = api.open_session(run_folder[0])
    assert api.plan_operation(s, "detect", "R1", f"R1 - {CHANNEL}") == api.plan_operation(s, "detect", "R1", CHANNEL)
    assert api.plan_operation(s, "transform", "R2", f"R2 - {CHANNEL}").writes[0].endswith(
        f"R2_-_{CHANNEL}_puncta.csv")
    with pytest.raises(ValueError, match=f"'R2 - {CHANNEL}' is a channel of R2, not of R1"):
        api.plan_operation(s, "detect", "R1", f"R2 - {CHANNEL}")
    with pytest.raises(ValueError, match="R1's image has no channel 'Cy5'; its channels are DAPI, FITC"):
        api.plan_operation(s, "detect", "R1", "Cy5")
    with pytest.raises(FileNotFoundError, match="detect the channel first"):
        api.plan_operation(s, "transform", "R1", "DAPI")
    change = api.set_param(s, "detection", "sensitivity", 0.2, channel=f"R2 - {CHANNEL}", dry_run=True)
    assert change.channel == CHANNEL and change.new == 0.2
    with decisions.use_actor("agent:test", via="mcp:pytest"):
        entry = api.propose(s, "threshold_rel", 0.08, scope={"channel": f"R2 - {CHANNEL}"}, justification="x")
    assert entry["scope"] == {"channel": CHANNEL, "round": "R2"}
    assert api.round_channel(None, f"R1 - {CHANNEL}") == api.round_channel("R1", CHANNEL) == CHANNEL


def test_a_proposal_is_logged_without_changing_the_session(run_folder):
    out = run_folder[0]
    before = _session_files(out)
    s = api.open_session(out)
    with decisions.use_actor("agent:test", via="mcp:pytest"):
        entry = api.propose(s, "threshold_rel", 0.08, scope={"channel": CHANNEL}, justification="SNR is low")
    assert entry["status"] == "proposed" and entry["via"] == "mcp:pytest" and entry["actor"] == "agent:test"
    assert api.get_decisions(s)[-1]["justification"] == "SNR is low"
    assert _session_files(out) == before


def test_proposals_wait_in_the_review_queue_until_a_person_answers(run_folder):
    out = run_folder[0]
    s = api.open_session(out)
    with decisions.use_actor("agent:test", via="mcp:pytest"):
        first = api.propose(s, "threshold_rel", 0.08, scope={"round": "R1", "channel": CHANNEL}, justification="low SNR")
        second = api.propose(s, "remove_border_nuclei", True, scope={"stage": "consensus"}, justification="cut nuclei")
    queue = api.review_queue(s)
    assert [p.id for p in queue] == [first["id"], second["id"]] and len(first["id"]) == 12
    assert (queue[0].name, queue[0].value, queue[0].actor, queue[0].via) == ("threshold_rel", 0.08, "agent:test",
                                                                             "mcp:pytest")
    with decisions.use_actor(decisions.HUMAN):
        accepted = api.review_proposal(s, first["id"], accept=True, note="agreed")
        dismissed = api.review_proposal(s, second["id"], accept=False)
    assert (accepted["status"], accepted["actor"], accepted["justification"]) == ("accepted", "human", "agreed")
    assert accepted["new"] == {"id": first["id"], "name": "threshold_rel", "value": 0.08}
    assert dismissed["status"] == "dismissed" and api.review_queue(s) == []
    with pytest.raises(LookupError, match="No open proposal"):
        api.review_proposal(s, first["id"], accept=False)


def test_the_outcome_of_a_run_in_brief(run_folder):
    out = run_folder[0]
    held = api.outcome(api.open_session(out))
    assert held.n_nuclei > 0 and set(held.channels) == {f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"}
    assert all(c["n_puncta"] > 0 and c["median_snr"] > 1 for c in held.channels.values())
    assert held.registration["method"] == "ransac" and held.warnings == []
    assert held.missing == ["Master_Analysis_Report"]                   # this run had no colocalization rules
    newest = api.session_files(out)[-1]
    data = json.loads(newest.read_text())
    data["registration"]["rigid"].update(method="rough_shift", fallback_used=True, fallback_reason="too_few_pairs")
    newest.write_text(json.dumps(data))
    held = api.outcome(api.open_session(out))
    assert held.registration["fallback_used"] and any("rough shift" in w for w in held.warnings)


def test_a_detection_previewed_writes_nothing_and_matches_the_run(run_folder):
    out = run_folder[0]
    before = _files(out)
    s = api.open_session(out)
    same = api.preview_detection(s, "R1", f"R1 - {CHANNEL}")
    assert same.channel == f"R1 - {CHANNEL}" and same.n_nuclei > 0
    assert same.preview["n_puncta"] == same.current["n_puncta"] > 0          # the recorded settings, as run
    assert same.preview["snr"] == pytest.approx(same.current["snr"]) and same.params["threshold_rel"] == 0.05
    strict = api.preview_detection(s, "R1", CHANNEL, {"method": "Local Maxima", "threshold_rel": 0.9,
                                                      "min_distance": 2})
    assert strict.preview["n_puncta"] < same.preview["n_puncta"] and strict.current == same.current
    assert _files(out) == before and s.read_only
    with pytest.raises(ValueError, match="no channel 'Cy5'"):
        api.preview_detection(s, "R1", "Cy5")
