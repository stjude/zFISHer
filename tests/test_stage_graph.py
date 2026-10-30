"""Stage records and resumable pipeline runs (restructuring plan 2f).

Every pipeline stage records a fingerprint of its settings, inputs and the
stages it used, and the state of the files it wrote. Running the pipeline
again over a folder it finished runs nothing; changing one channel's settings
runs only what depends on them; a file edited by hand is kept, and what was
made from it is made again. A folder whose session no pipeline with stage
records made (a GUI session, an older batch) is refused, as before.

Run:  python -m pytest tests/test_stage_graph.py -q
"""
import logging
import shutil

import numpy as np
import pytest

from zfisher import constants
from zfisher.core import api, io, pipeline, session, stages

from synthetic_rounds import write_rounds

CHANNEL = "FITC"
CONFIG = {CHANNEL: {"method": "Local Maxima", "threshold_rel": 0.05, "min_distance": 2, "sigma": 0.0,
                    "nuclei_only": True, "use_tophat": False, "tophat_radius": 15, "round": "Both"}}
RULES = [{"source": "R1 - FITC", "target": "R2 - FITC", "threshold": 1.0}]


def _run(root, out, config=CONFIG):
    factory = logging.getLogRecordFactory()
    levels = {n: logging.getLogger(n).level for n in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    try:
        return pipeline.run_full_zfisher_pipeline(root / "R1.ome.tif", root / "R2.ome.tif", out,
                                                  seg_method="Classical", puncta_config=config,
                                                  pairwise_rules=RULES)
    finally:
        logging.setLogRecordFactory(factory)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    pytest.importorskip("SimpleITK")
    root = tmp_path_factory.mktemp("rounds")
    write_rounds(root)
    ran = _run(root, root / "out")
    yield root, ran
    session.clear_session()


@pytest.fixture
def copy(finished, tmp_path):
    root, _ = finished
    shutil.copytree(root / "out", tmp_path / "out")
    for name in ("R1.ome.tif", "R2.ome.tif"):
        shutil.copy2(root / name, tmp_path / name)
    yield tmp_path
    session.clear_session()


def _states(out):
    return {p.relative_to(out): stages.file_state(p) for p in out.rglob("*")
            if p.is_file() and p.suffix in (".csv", ".tif", ".xlsx", ".npy", ".tfm")}


def test_a_first_run_records_every_stage(finished):
    _, ran = finished
    assert ran[0] == "segment" and ran[-1] == "report"
    assert {"register", "canvas", "consensus", f"detect R1 - {CHANNEL}", f"transform R2 - {CHANNEL}"} <= set(ran)


def test_a_second_run_over_a_finished_folder_runs_no_stage(copy):
    before = _states(copy / "out")
    assert _run(copy, copy / "out") == []
    assert _states(copy / "out") == before
    s = api.open_session(copy / "out")
    with session.use(s):
        assert {st["state"] for st in stages.stage_status()} == {stages.FRESH}


def test_changing_one_channels_settings_runs_only_what_depends_on_them(copy):
    ran = _run(copy, copy / "out", {CHANNEL: dict(CONFIG[CHANNEL], threshold_rel=0.08)})
    assert set(ran) == {f"detect R1 - {CHANNEL}", f"detect R2 - {CHANNEL}", f"transform R1 - {CHANNEL}",
                        f"transform R2 - {CHANNEL}", "report"}
    assert "segment" not in ran and "canvas" not in ran                     # nothing upstream again


def test_a_hand_edited_consensus_mask_is_kept_and_what_follows_is_redone(copy):
    out = copy / "out"
    path = io.find_mask(out / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME)
    mask = io.read_label_tif(path)
    label = int(np.unique(mask)[1])
    mask[mask == label] = 0                                         # a nucleus deleted by hand
    io.write_label_tif(path, mask)
    s = api.open_session(out)
    with session.use(s):
        assert {st["stage"]: st["state"] for st in stages.stage_status()}["consensus"] == "edited"
    ran = _run(copy, out)
    assert "consensus" not in ran and "canvas" not in ran
    assert {f"transform R1 - {CHANNEL}", f"transform R2 - {CHANNEL}", "report"} <= set(ran)
    assert label not in np.unique(io.read_label_tif(path))           # the edit is still there


def test_a_folder_no_pipeline_run_made_is_refused(copy):
    out = copy / "out"
    for f in (out / constants.SESSIONS_DIR).glob("*.json"):
        f.unlink()
    session.clear_session()
    session.update_data("output_dir", str(out))                      # a session made by hand, in the GUI
    session.clear_session()
    with pytest.raises(RuntimeError, match="not made by a pipeline run that can be resumed"):
        _run(copy, out)
