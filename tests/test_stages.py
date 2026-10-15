"""Re-detecting one channel of a finished session and warping it again.

``stages.detect_channel`` detects a channel on its raw image, read from disk,
into the channel's raw-space CSV; ``stages.transform_channel`` takes that CSV
through the registration saved at canvas generation into the aligned CSV. The
tests run the batch pipeline once on small synthetic rounds (about 2 s), and
each test reopens its own copy of the output folder the way Load Session does.

Run:  python -m pytest tests/test_stages.py -q
"""
import logging
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import io, provenance, puncta, session, stages

from synthetic_rounds import run_batch, write_rounds

CHANNEL = "FITC"


@pytest.fixture(scope="module")
def finished():
    """A batch output folder, made once."""
    pytest.importorskip("SimpleITK")
    root = Path(tempfile.mkdtemp())
    write_rounds(root)
    run_batch(root, root / "out")
    session.clear_session()
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def reopened(finished, tmp_path):
    """A copy of the batch output, opened with ``session.load_session_file``."""
    out = tmp_path / "out"
    shutil.copytree(finished / "out", out)
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)       # see synthetic_rounds.run_batch
    session.load_session_file(next(out.rglob("zfisher_session_1.json")))
    yield types.SimpleNamespace(out=out, reports=out / constants.REPORTS_DIR,
                                original=finished / "out" / constants.REPORTS_DIR)
    session.clear_session()
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


def _csv(directory, name):
    return pd.read_csv(constants.puncta_csv_path(directory, name))


def _raw(directory, round_id):
    return _csv(directory, stages.raw_puncta_name(round_id, CHANNEL))


def _aligned_name(round_id):
    prefix = constants.WARPED_PREFIX if round_id == "R2" else constants.ALIGNED_PREFIX
    return f"{prefix} {round_id} - {CHANNEL}{constants.PUNCTA_SUFFIX}"


@pytest.mark.parametrize("round_id", ["R1", "R2"])
def test_transform_reproduces_the_batch_aligned_csv(reopened, round_id):
    path = stages.transform_channel(round_id, CHANNEL)
    assert path == constants.puncta_csv_path(reopened.reports, _aligned_name(round_id))
    expected = constants.puncta_csv_path(reopened.original, _aligned_name(round_id))
    assert path.read_bytes() == expected.read_bytes()
    record = provenance.latest(path)
    assert record["event"] == "transformed" and record["space"] == "aligned" and record["run_by"] == "gui"
    assert record["derived_from"] == f"{round_id}_-_{CHANNEL}_puncta.csv"
    assert record["remove_extranuclear"] is True                 # as batch made it
    assert record["n_carried_manual"] == 0


@pytest.mark.parametrize("round_id", ["R1", "R2"])
def test_redetect_with_the_recorded_settings_finds_the_same_puncta_under_new_ids(reopened, round_id):
    before = _raw(reopened.reports, round_id)
    stages.detect_channel(round_id, CHANNEL)
    after = _raw(reopened.reports, round_id)
    np.testing.assert_array_equal(after[["Z", "Y", "X"]], before[["Z", "Y", "X"]])
    assert list(after.columns) == constants.PUNCTA_CSV_COLUMNS
    assert after["puncta_id"].min() > before["puncta_id"].max()
    # Detection's own numbering, shifted: the gaps left by the nucleus filter stay.
    np.testing.assert_array_equal(np.diff(after["puncta_id"]), np.diff(before["puncta_id"]))
    record = provenance.latest(constants.puncta_csv_path(reopened.reports, stages.raw_puncta_name(round_id, CHANNEL)))
    assert record["event"] == "detected" and record["detected_in"] == "raw" and record["existing_action"] == "replace"
    assert record["mask"]["path"].endswith(f"{round_id} - DAPI_masks.ome.tif")
    assert record["source_image"]["path"].endswith(f"{round_id}.ome.tif")
    row = puncta.detection_settings()[f"{round_id} - {CHANNEL}"]
    assert row["detected_in"] == "raw" and row["num_puncta"] == len(after)
    assert session.get_data("processed_files")[stages.raw_puncta_name(round_id, CHANNEL)]["space"] == "raw"


def test_changed_settings_reach_the_aligned_csv(reopened):
    params = stages.recorded_detection_parameters("R2", CHANNEL)
    batch_aligned = _csv(reopened.original, _aligned_name("R2"))
    stages.detect_channel("R2", CHANNEL, params=dict(params, threshold_rel=0.6))
    raw = _raw(reopened.reports, "R2")
    aligned = pd.read_csv(stages.transform_channel("R2", CHANNEL))
    assert 0 < len(raw) < len(_raw(reopened.original, "R2"))
    assert set(aligned["puncta_id"]) <= set(raw["puncta_id"])
    assert len(aligned) < len(batch_aligned)
    assert puncta.detection_settings()[f"R2 - {CHANNEL}"]["sensitivity"] == 0.6


def _append(path, rows):
    table = pd.read_csv(path)
    pd.concat([table, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def test_puncta_placed_by_hand_survive_a_redetect(reopened):
    raw_csv = constants.puncta_csv_path(reopened.reports, stages.raw_puncta_name("R2", CHANNEL))
    aligned_csv = constants.puncta_csv_path(reopened.reports, _aligned_name("R2"))
    # One picked on the raw image, inside a nucleus; one picked on the aligned layer.
    _append(raw_csv, [{"puncta_id": 900, "Z": 16.0, "Y": 43.0, "X": 38.0, "Nucleus_ID": 0, "Intensity": 0,
                       "SNR": 0, "Source": "manual"}])
    _append(aligned_csv, [{"puncta_id": 901, "Z": 15.5, "Y": 60.25, "X": 70.75, "Nucleus_ID": 0,
                           "Intensity": 0, "SNR": 0, "Source": "manual"}])
    stages.detect_channel("R2", CHANNEL)
    raw = _raw(reopened.reports, "R2")
    assert list(raw.loc[raw["Source"] == "manual", "puncta_id"]) == [900]
    aligned = pd.read_csv(stages.transform_channel("R2", CHANNEL))
    manual = aligned[aligned["Source"] == "manual"].set_index("puncta_id")
    assert set(manual.index) == {900, 901}
    assert list(manual.loc[901, ["Z", "Y", "X"]]) == [15.5, 60.25, 70.75]       # not moved
    assert provenance.latest(aligned_csv)["n_carried_manual"] == 1
    assert aligned["puncta_id"].is_unique

    stages.detect_channel("R2", CHANNEL, existing="replace_all")
    assert set(_raw(reopened.reports, "R2")["Source"]) == {"auto"}
    aligned = pd.read_csv(stages.transform_channel("R2", CHANNEL, keep_aligned_manual=False))
    assert set(aligned["Source"]) == {"auto"}


def test_a_carried_punctum_whose_id_is_taken_gets_a_new_one(reopened):
    aligned_csv = stages.transform_channel("R1", CHANNEL)
    taken = int(pd.read_csv(aligned_csv)["puncta_id"].iloc[0])
    _append(aligned_csv, [{"puncta_id": taken, "Z": 15.0, "Y": 50.0, "X": 50.0, "Nucleus_ID": 0,
                           "Intensity": 0, "SNR": 0, "Source": "manual"}])
    aligned = pd.read_csv(stages.transform_channel("R1", CHANNEL))
    assert aligned["puncta_id"].is_unique
    carried = aligned[(aligned["Source"] == "manual")]
    assert len(carried) == 1 and carried["puncta_id"].iloc[0] != taken


def test_merge_keeps_every_row_and_adds_no_duplicates(reopened):
    before = _raw(reopened.reports, "R1")
    stages.detect_channel("R1", CHANNEL, existing="merge")
    after = _raw(reopened.reports, "R1")
    assert list(after["puncta_id"]) == list(before["puncta_id"])


def test_a_missing_raw_image_names_the_round(reopened):
    session.update_data("r1_path", str(reopened.out / "moved" / "R1.ome.tif"))
    with pytest.raises(io.RawImageNotFound, match="R1"):
        stages.detect_channel("R1", CHANNEL)


def test_detecting_inside_nuclei_needs_the_round_mask(reopened):
    files = dict(session.get_data("processed_files"))
    del files["R2 - DAPI_masks"]
    session.update_data("processed_files", files)
    with pytest.raises(FileNotFoundError, match="R2 has no nuclei mask"):
        stages.detect_channel("R2", CHANNEL)
    stages.detect_channel("R2", CHANNEL, params=dict(stages.recorded_detection_parameters("R2", CHANNEL),
                                                       nuclei_only=False))
    record = provenance.latest(constants.puncta_csv_path(reopened.reports, stages.raw_puncta_name("R2", CHANNEL)))
    assert record["mask"] == "none"


def test_a_session_without_a_saved_registration_cannot_be_rewarped(reopened):
    session.update_data("registration", {})
    with pytest.raises(LookupError, match="no saved registration"):
        stages.transform_channel("R2", CHANNEL)


def test_settings_must_be_given_when_none_are_recorded(reopened):
    with pytest.raises(LookupError, match="No detection settings"):
        stages.detect_channel("R1", "DAPI")
    with pytest.raises(ValueError, match="existing must be"):
        stages.detect_channel("R1", CHANNEL, existing="keep")


def test_finding_nothing_empties_the_csv_or_writes_none(reopened):
    """A re-detection that finds nothing replaces the old puncta; a channel
    with no CSV yet gets none, as batch writes none for an empty channel."""
    params = dict(stages.recorded_detection_parameters("R1", CHANNEL), threshold_rel=1.01)
    raw_csv = stages.detect_channel("R1", CHANNEL, params=params)
    assert len(pd.read_csv(raw_csv)) == 0 and list(pd.read_csv(raw_csv).columns) == constants.PUNCTA_CSV_COLUMNS
    aligned_csv = stages.transform_channel("R1", CHANNEL)
    assert len(pd.read_csv(aligned_csv)) == 0                    # the batch rows are gone too
    raw_csv.unlink()
    aligned_csv.unlink()
    assert stages.detect_channel("R1", CHANNEL, params=params) is None
    assert not raw_csv.exists()
    assert puncta.detection_settings()[f"R1 - {CHANNEL}"]["num_puncta"] == 0


# --- what is out of date --------------------------------------------------------

def _states(items=None):
    return {item.name: (item.state, item.edited) for item in (items or stages.status())}


def _export_report(reopened):
    from zfisher.core import analysis
    layers = []
    for name in (_aligned_name("R1"), _aligned_name("R2")):
        table = _csv(reopened.reports, name)
        layers.append({"name": name, "data": table[["Z", "Y", "X"]].to_numpy(float), "scale": np.ones(3),
                       "translate": np.zeros(3), "nucleus_ids": table["Nucleus_ID"].to_numpy(),
                       "puncta_id": table["puncta_id"].to_numpy()})
    return analysis.run_colocalization_analysis(layers, [], "report.xlsx", None, None, reopened.reports)


def test_a_finished_batch_session_is_up_to_date(reopened):
    states = _states()
    assert states[_aligned_name("R1")] == ("fresh", False) and states[_aligned_name("R2")] == ("fresh", False)
    assert states[stages.REPORT_NAME] == ("missing", False)          # no rules, so batch wrote no report


def test_a_redetected_channel_is_stale_until_warped_again(reopened):
    report = _export_report(reopened)
    assert _states()[report.name] == ("fresh", False)
    stages.detect_channel("R2", CHANNEL, params=dict(stages.recorded_detection_parameters("R2", CHANNEL),
                                                       threshold_rel=0.6))
    items = {item.name: item for item in stages.status()}
    r2 = items[_aligned_name("R2")]
    assert (r2.state, r2.edited, r2.round, r2.channel) == ("stale", False, "R2", CHANNEL)
    assert "R2_-_FITC_puncta.csv changed since it was warped" in r2.reasons
    assert items[_aligned_name("R1")].state == "fresh"
    assert items[report.name].state == "stale"                      # downstream of the stale channel
    stages.transform_channel("R2", CHANNEL)
    states = _states()
    assert states[_aligned_name("R2")] == ("fresh", False)
    assert states[report.name] == ("stale", False)                  # its R2 puncta changed
    report = _export_report(reopened)
    assert _states()[report.name] == ("fresh", False)


def test_hand_edits_are_told_apart_from_nucleus_id_updates(reopened):
    aligned_csv = constants.puncta_csv_path(reopened.reports, _aligned_name("R1"))
    table = pd.read_csv(aligned_csv)
    table["Nucleus_ID"] = 0                                          # as a reload or export re-derives them
    puncta.write_puncta_csv(aligned_csv, table[["Z", "Y", "X"]].to_numpy(float), table.drop(columns=["Z", "Y", "X"]))
    assert _states()[_aligned_name("R1")] == ("fresh", False)
    _append(aligned_csv, [{"puncta_id": 5000, "Z": 15.0, "Y": 50.0, "X": 50.0, "Nucleus_ID": 0,
                           "Intensity": 0, "SNR": 0, "Source": "manual"}])
    item = {i.name: i for i in stages.status()}[_aligned_name("R1")]
    assert (item.state, item.edited) == ("fresh", True)
    stages.detect_channel("R1", CHANNEL)
    assert _states()[_aligned_name("R1")] == ("stale", True)        # a re-warp would have to keep the edit


def test_a_consensus_mask_edit_makes_the_report_and_filtered_puncta_stale(reopened):
    report = _export_report(reopened)
    consensus = io.find_mask(reopened.out / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME)
    mask = io.read_label_tif(consensus)
    mask[mask == mask.max()] = 0                                      # delete one nucleus
    io.write_label_tif(consensus, mask)
    items = {item.name: item for item in stages.status()}
    assert items[report.name].state == "stale"
    assert "the consensus mask changed since it was exported" in items[report.name].reasons
    # Batch removed puncta outside nuclei with the old mask.
    assert items[_aligned_name("R1")].state == "stale"


def test_aligned_detection_and_sessions_without_a_saved_registration(reopened):
    from zfisher.core import provenance as prov
    aligned_csv = constants.puncta_csv_path(reopened.reports, _aligned_name("R2"))
    puncta.record_puncta(aligned_csv, "detected", detected_in="aligned", space="aligned", run_by="gui")
    assert _states()[_aligned_name("R2")] == ("detected_on_aligned", False)
    session.update_data("registration", {})
    assert [item.kind for item in stages.status()] == ["report"]     # cannot be warped again: not judged
    assert prov.latest(aligned_csv)["event"] == "detected"


def test_sessions_made_before_fingerprints_were_kept(reopened):
    """2.1.x recorded neither the raw CSV's hash nor the puncta fingerprint:
    a re-detection is seen from the record times, and a hand edit from the
    counts by Source."""
    import json
    log = reopened.out / "provenance.jsonl"
    records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    for record in records:
        record.pop("derived_from_sha256", None)
        record.pop("points_sha256", None)
    log.write_text("".join(json.dumps(r) + "\n" for r in records))
    assert _states()[_aligned_name("R2")] == ("fresh", False)
    aligned_csv = constants.puncta_csv_path(reopened.reports, _aligned_name("R2"))
    _append(aligned_csv, [{"puncta_id": 5000, "Z": 15.0, "Y": 50.0, "X": 50.0, "Nucleus_ID": 0,
                           "Intensity": 0, "SNR": 0, "Source": "manual"}])
    puncta.record_puncta(aligned_csv, "saved")                      # as the GUI's autosave records it
    assert _states()[_aligned_name("R2")] == ("fresh", True)
    import time
    time.sleep(1.1)                                                  # record times are to the second
    raw_csv = constants.puncta_csv_path(reopened.reports, stages.raw_puncta_name("R2", CHANNEL))
    puncta.record_puncta(raw_csv, "detected")
    assert _states()[_aligned_name("R2")] == ("stale", True)


# --- re-detecting one channel -----------------------------------------------------

def test_redetect_channel_detects_and_warps_and_says_the_report_is_out_of_date(reopened):
    report = _export_report(reopened)
    assert stages.report_note() is None
    params = dict(stages.recorded_detection_parameters("R2", CHANNEL), threshold_rel=0.6)
    raw_csv, aligned_csv = stages.redetect_channel("R2", CHANNEL, params)
    assert raw_csv.name == "R2_-_FITC_puncta.csv" and aligned_csv.name == "Warped_R2_-_FITC_puncta.csv"
    assert _states()[_aligned_name("R2")] == ("fresh", False)
    note = stages.report_note()
    assert note.startswith(f"The report {report.name} is out of date: the puncta in Warped_R2_-_FITC_puncta.csv")
    assert note.endswith("Export it again to update it.")


def test_can_rewarp_only_with_a_saved_registration(reopened):
    assert stages.can_rewarp() == (True, None)
    session.update_data("registration", {})
    ok, reason = stages.can_rewarp()
    assert not ok and "no saved registration" in reason
