"""Raw-space puncta are kept, recorded, and can be transformed again.

Batch detected puncta on the raw images and discarded them after transforming
them to aligned space; the GUI canvas step renamed each raw puncta CSV to .bak
and unregistered it. Now both keep one raw CSV per round and channel, registered
with ``space='raw'``. Every puncta CSV zFISHer writes is recorded in the
session's ``provenance.jsonl`` and logged as an ``ARTIFACT`` line: its SHA-256,
round, channel, space, image and mask (each with its SHA-256), detection
parameters, who ran it (batch or gui) and the software version. Aligned CSVs
name the raw CSV and the registration that placed them. A reopened session with
aligned data does not load raw puncta as layers.

The batch test runs the whole pipeline on small synthetic OME-TIFF rounds
(about 2 s). The GUI tests use a headless ``ViewerModel``.

Run:  python -m pytest tests/test_raw_puncta.py -q
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import provenance, puncta

from synthetic_rounds import run_batch, write_rounds


# --- core provenance --------------------------------------------------------------

@pytest.fixture
def open_session(tmp_path, monkeypatch):
    """An in-memory session whose output folder is ``tmp_path``."""
    from zfisher.core import session
    data = {"output_dir": str(tmp_path)}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default))
    monkeypatch.setattr(session, "update_data", lambda key, value: data.__setitem__(key, value))
    return data


def test_records_carry_fields_forward_and_count_sources(open_session, tmp_path, caplog):
    csv = tmp_path / "reports" / "R1_-_FITC_puncta.csv"
    csv.parent.mkdir()
    puncta.write_puncta_csv(csv, np.zeros((3, 3)), pd.DataFrame({
        "puncta_id": [0, 1, 2], "Source": ["auto", "manual", None]}))
    assert provenance.records() == []                          # no record yet, so the write added none
    with caplog.at_level("INFO"):
        puncta.record_puncta(csv, "detected", round="R1", detection_parameters={"method": "LoG"}, run_by="gui")
    later = puncta.record_puncta(csv, "kept_raw", space="raw:R1", round=None)
    assert later["path"] == "reports/R1_-_FITC_puncta.csv"     # relative to the output folder
    assert later["round"] == "R1" and later["detection_parameters"] == {"method": "LoG"}
    assert later["space"] == "raw:R1" and later["event"] == "kept_raw" and later["run_by"] == "gui"
    assert later["n_puncta"] == 3 and later["n_by_source"] == {"auto": 1, "manual": 1, "unknown": 1}
    assert later["sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert later["software"]["zfisher"]
    assert [r["event"] for r in provenance.records(csv)] == ["detected", "kept_raw"]
    lines = (tmp_path / provenance.PROVENANCE_FILE).read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[-1]) == later
    assert any(r.getMessage().startswith("ARTIFACT path=reports/R1_-_FITC_puncta.csv event=detected")
               for r in caplog.records)


def test_without_a_session_nothing_is_recorded(tmp_path):
    csv = tmp_path / "x.csv"
    puncta.write_puncta_csv(csv, np.zeros((1, 3)))
    assert puncta.record_puncta(csv, "detected") is None and provenance.verify(csv) is None


def test_file_identity_hashes_and_notes_a_missing_file(open_session, tmp_path):
    f = tmp_path / "img.tif"
    f.write_bytes(b"1234")
    identity = provenance.file_identity(f)
    assert identity["size_bytes"] == 4 and identity["path"] == "img.tif"
    assert identity["sha256"] == hashlib.sha256(b"1234").hexdigest()
    assert provenance.file_identity(tmp_path / "gone.tif") == {"path": "gone.tif", "missing": True}


def test_verify_detects_a_csv_changed_outside_zfisher(open_session, tmp_path):
    csv = tmp_path / "R1_-_FITC_puncta.csv"
    assert provenance.verify(csv) is None                       # no record yet
    puncta.write_puncta_csv(csv, np.zeros((2, 3)), pd.DataFrame({"puncta_id": [0, 1]}))
    puncta.record_puncta(csv, "detected", round="R1")
    assert provenance.verify(csv) is True
    puncta.write_puncta_csv(csv, np.ones((3, 3)), pd.DataFrame({"puncta_id": [0, 1, 2]}))
    assert provenance.verify(csv) is True                       # zFISHer's own writes are recorded
    latest = provenance.latest(csv)
    assert latest["event"] == "saved" and latest["n_puncta"] == 3 and latest["round"] == "R1"
    with open(csv, "a") as f:
        f.write("9,9,9,9,,,,manual\n")                          # an edit in another program
    assert provenance.verify(csv) is False


def test_input_hashes_are_cached_per_session(open_session, tmp_path, monkeypatch):
    calls = []
    real = provenance._sha256
    monkeypatch.setattr(provenance, "_sha256", lambda p: (calls.append(p), real(p))[1])
    image = tmp_path / "R1.nd2"
    image.write_bytes(b"x" * 100)
    first = provenance.file_sha256(image)
    assert provenance.file_sha256(image) == first and len(calls) == 1   # second call uses the cache
    assert open_session["file_hashes"][str(image.resolve())]["sha256"] == first
    image.write_bytes(b"y" * 101)                                       # a different file at the same path
    assert provenance.file_sha256(image) != first and len(calls) == 2


# --- batch, end to end ----------------------------------------------------------

@pytest.fixture(scope="module")
def batch():
    pytest.importorskip("SimpleITK")
    from zfisher.core import session
    root = Path(tempfile.mkdtemp())
    write_rounds(root)
    out = root / "out"
    run_batch(root, out)
    yield types.SimpleNamespace(root=root, out=out, reports=out / constants.REPORTS_DIR,
                                processed=dict(session.get_data("processed_files", default={})),
                                hashes=dict(session.get_data("file_hashes", default={})),
                                session=session)
    session.clear_session()
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("round_id", ["R1", "R2"])
def test_batch_keeps_one_annotated_raw_csv_per_channel(batch, round_id):
    raw = batch.reports / f"{round_id}_-_FITC_puncta.csv"
    assert raw.exists() and len(pd.read_csv(raw)) > 0
    entry = batch.processed[f"{round_id} - FITC_puncta"]
    assert entry["space"] == "raw" and entry["round"] == round_id
    assert not list(batch.reports.glob("*.provenance.json"))       # no sidecars
    record = provenance.latest(raw)
    assert record["event"] == "detected" and record["path"] == f"reports/{raw.name}"
    assert record["space"] == f"raw:{round_id}" and record["detected_in"] == "raw"
    assert record["channel"] == "FITC" and record["run_by"] == "batch"
    assert record["source_image"]["path"].endswith(f"{round_id}.ome.tif")
    assert record["source_image"]["sha256"] == provenance._sha256(batch.root / f"{round_id}.ome.tif")
    assert record["mask"]["path"].endswith("DAPI_masks.ome.tif") and len(record["mask"]["sha256"]) == 64
    assert provenance.verify(raw) is True
    image_key = str((batch.root / f"{round_id}.ome.tif").resolve())
    assert batch.hashes[image_key]["sha256"] == record["source_image"]["sha256"]   # hashed once, cached
    assert "method" in record["detection_parameters"]
    assert record["detection_parameters"]["z_scale"] == pytest.approx(3.0)   # filled in by detection
    assert record["n_by_source"] == {"auto": record["n_puncta"]}


@pytest.mark.parametrize("round_id, prefix", [("R1", "Aligned"), ("R2", "Warped")])
def test_raw_csv_transformed_from_disk_reproduces_the_batch_result(batch, round_id, prefix):
    from zfisher.core import io, registration
    aligned_csv = batch.reports / f"{prefix}_{round_id}_-_FITC_puncta.csv"
    record = provenance.latest(aligned_csv)
    assert record["event"] == "transformed" and record["derived_from"] == f"{round_id}_-_FITC_puncta.csv"
    assert provenance.verify(aligned_csv) is True
    assert len(record["consensus_mask"]["sha256"]) == 64
    assert record["registration"] == registration.registration_fingerprint()
    consensus = io.read_label_tif(io.find_mask(batch.out / constants.SEGMENTATION_DIR,
                                               constants.CONSENSUS_MASKS_NAME))
    data, sources = puncta.transform_puncta_csv(
        batch.reports / record["derived_from"], round_id, registration.load_registration(),
        consensus_mask=consensus, remove_extranuclear=True)
    expected = pd.read_csv(aligned_csv)
    np.testing.assert_array_equal(data[:, 6], expected["puncta_id"])
    np.testing.assert_allclose(data[:, :3], expected[["Z", "Y", "X"]], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(data[:, 3], expected["Nucleus_ID"])
    assert list(sources) == list(expected["Source"])


# --- GUI -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def gui(_qapp, monkeypatch):
    """A headless viewer whose layer removals go through zFISHer's handler,
    with an in-memory session."""
    from napari.components import ViewerModel
    from zfisher.core import session
    from zfisher.ui import events, viewer_helpers
    out = Path(tempfile.mkdtemp())
    data = {"output_dir": str(out), "processed_files": {}, "r1_path": str(out / "R1.nd2")}

    def set_processed_file(name, path, layer_type, metadata=None):
        data["processed_files"][name] = {"path": str(path), "type": layer_type, **(metadata or {})}

    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "update_data", lambda key, value: data.__setitem__(key, value))
    monkeypatch.setattr(session, "set_processed_file", set_processed_file)
    monkeypatch.setattr(session, "remove_processed_file", lambda name: data["processed_files"].pop(name, None))
    monkeypatch.setattr(events, "QTimer", types.SimpleNamespace(singleShot=lambda _ms, fn: None))
    viewer = ViewerModel()
    monkeypatch.setattr(events.napari, "current_viewer", lambda: viewer)
    viewer.layers.events.removed.connect(lambda e: events.on_layer_removed(e, {}))
    yield types.SimpleNamespace(viewer=viewer, helpers=viewer_helpers, data=data,
                                reports=out / constants.REPORTS_DIR)
    shutil.rmtree(out, ignore_errors=True)


def _raw_layer(gui, name="R1 - FITC_puncta"):
    source = types.SimpleNamespace(name=name.replace(constants.PUNCTA_SUFFIX, ""),
                                   scale=(1, 1, 1), translate=(0, 0, 0))
    rows = np.column_stack([np.ones((3, 3)) * [[1], [2], [3]], np.zeros(3), np.full(3, 9.0),
                            np.full(3, 2.0), [0, 1, 2]])
    gui.helpers.add_or_update_puncta_layer(gui.viewer, source, rows, sources=["auto", "auto", "manual"])
    return gui.viewer.layers[name]


def test_canvas_step_keeps_the_raw_csv_registered(gui):
    layer = _raw_layer(gui)
    layer.features.loc[0, "Source"] = "manual"          # an edit made before the canvas step
    gui.helpers.retire_raw_puncta_layer(gui.viewer, layer)
    csv = gui.reports / "R1_-_FITC_puncta.csv"
    assert "R1 - FITC_puncta" not in gui.viewer.layers
    assert csv.exists() and not csv.with_name(csv.name + ".bak").exists()
    assert list(pd.read_csv(csv)["Source"]) == ["manual", "auto", "manual"]
    entry = gui.data["processed_files"]["R1 - FITC_puncta"]
    assert entry["space"] == "raw" and entry["round"] == "R1"
    record = provenance.latest(csv)
    assert record["space"] == "raw:R1" and record["run_by"] == "gui" and record["event"] == "kept_raw"
    assert record["n_by_source"] == {"manual": 2, "auto": 1}


def test_other_removals_still_back_the_file_up(gui):
    layer = _raw_layer(gui)
    gui.viewer.layers.remove(layer)
    csv = gui.reports / "R1_-_FITC_puncta.csv"
    assert not csv.exists() and csv.with_name(csv.name + ".bak").exists()
    assert "R1 - FITC_puncta" not in gui.data["processed_files"]


def test_reload_of_an_aligned_session_skips_raw_puncta(gui):
    layer = _raw_layer(gui)
    gui.helpers.retire_raw_puncta_layer(gui.viewer, layer)
    raw_entry = dict(gui.data["processed_files"]["R1 - FITC_puncta"])
    aligned_csv = gui.reports / "Aligned_R1_-_FITC_puncta.csv"
    shutil.copy(gui.reports / "R1_-_FITC_puncta.csv", aligned_csv)
    files = {"R1 - FITC_puncta": raw_entry,
             "Aligned R1 - FITC_puncta": {"path": str(aligned_csv), "type": "points"}}
    gui.helpers.restore_processed_layers(gui.viewer, files, (1, 1, 1))
    assert "Aligned R1 - FITC_puncta" in gui.viewer.layers
    assert "R1 - FITC_puncta" not in gui.viewer.layers
    # Before the canvas step there is nothing aligned, and raw puncta load as layers.
    gui.helpers.restore_processed_layers(gui.viewer, {"R1 - FITC_puncta": raw_entry}, (1, 1, 1))
    assert "R1 - FITC_puncta" in gui.viewer.layers


def test_detection_on_an_aligned_image_is_flagged(gui):
    image = gui.viewer.add_image(np.zeros((4, 8, 8)), name="Warped R2 - FITC", scale=(0.3, 0.1, 0.1))
    gui.data["processed_files"]["Warped R2 - FITC"] = {"path": str(gui.reports / "w.tif"), "type": "image"}
    _raw_layer(gui, "Warped R2 - FITC_puncta")
    gui.helpers.record_detection_provenance(image, None, {"method": "LoG"}, existing_action="replace")
    record = provenance.latest(gui.reports / "Warped_R2_-_FITC_puncta.csv")
    assert record["detected_in"] == "aligned" and record["space"] == "aligned"
    assert record["round"] == "R2" and record["existing_action"] == "replace"
    assert record["source_image"]["path"].endswith("w.tif") and record["event"] == "detected"


def test_detection_on_a_raw_image_names_the_round_file(gui):
    image = gui.viewer.add_image(np.zeros((4, 8, 8)), name="R1 - FITC")
    _raw_layer(gui)
    gui.helpers.record_detection_provenance(image, None, {"method": "LoG"})
    record = provenance.latest(gui.reports / "R1_-_FITC_puncta.csv")
    assert record["detected_in"] == "raw" and record["space"] == "raw:R1"
    assert record["source_image"] == {"path": "R1.nd2", "missing": True}   # inside the output folder here
