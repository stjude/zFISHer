"""Removing nuclei cut by the edge of the imaged region (plan item C4).

The region both rounds imaged is the intersection of their footprints on the
aligned canvas, from the registration's canvas record; a nucleus whose
bounding box reaches its edge in Y or X is cut by the field of view. The
option is off by default, set in the GUI, the template or ``zfisher-cli run
--remove-border-nuclei``, and part of the consensus stage's fingerprint, so
turning it on re-runs the consensus and what follows, not the rest.

Run:  python -m pytest tests/test_border_nuclei.py -q
"""
import logging

import numpy as np
import pytest

from zfisher import constants
from zfisher.core import io, pipeline, segmentation, session

from synthetic_rounds import write_rounds

CANVAS = {"canvas_shape": [32, 195, 194], "round_shapes": {"R1": [32, 192, 192], "R2": [32, 192, 192]},
          "raw_to_canvas": {"R1": [0.0, 3.0, 0.0], "R2": [0.0, 0.17, 2.0]}}


def test_the_imaged_region_is_the_rounds_overlap_rounded_inwards():
    start, end = segmentation.imaged_xy_box(CANVAS, (32, 195, 194))
    assert list(start) == [3, 2] and list(end) == [192, 192]
    start, end = segmentation.imaged_xy_box({}, (4, 50, 60))
    assert list(start) == [0, 0] and list(end) == [50, 60]


def _mask():
    mask = np.zeros((4, 60, 60), np.int32)
    mask[1:3, 20:30, 20:30] = 1          # inside
    mask[1:3, 5:12, 25:32] = 2           # reaches Y = 5, the region's edge
    mask[1:3, 30:40, 50:58] = 3          # reaches X = 57, past the edge at 55
    mask[1:3, 6:14, 6:14] = 4            # close to the edge, not on it
    return mask


def test_nuclei_reaching_the_edge_are_found_and_removed(tmp_path):
    box = (np.array([5, 5]), np.array([55, 55]))
    assert segmentation.border_nuclei(_mask(), box) == [2, 3]
    session.clear_session()
    try:
        session.update_data("output_dir", str(tmp_path))
        session.update_data("registration", {"canvas": {"round_shapes": {"R1": [4, 50, 50], "R2": [4, 50, 50]},
                                                        "raw_to_canvas": {"R1": [0, 5, 5], "R2": [0, 5, 5]}}})
        merged, _ = segmentation.process_consensus_nuclei(_mask(), _mask(), tmp_path, method="Intersection",
                                                          remove_border=True, voxel_size=(0.3, 0.1, 0.1))
        assert set(np.unique(merged)) - {0} == {1, 4}
        saved = io.read_label_tif(io.find_mask(tmp_path / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME))
        assert set(np.unique(saved)) - {0} == {1, 4}
    finally:
        session.clear_session()


def test_turning_it_on_reruns_the_consensus_and_what_follows(tmp_path):
    pytest.importorskip("SimpleITK")
    write_rounds(tmp_path)
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)
    levels = {n: logging.getLogger(n).level for n in ("zfisher", "stdout", "stderr")}
    try:
        args = (tmp_path / "R1.ome.tif", tmp_path / "R2.ome.tif", tmp_path / "out")
        first = pipeline.run_full_zfisher_pipeline(*args, seg_method="Classical")
        again = pipeline.run_full_zfisher_pipeline(*args, seg_method="Classical", remove_border_nuclei=True)
    finally:
        logging.setLogRecordFactory(factory)
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        session.clear_session()
    assert "consensus" in first and "consensus" in again
    assert "segment" not in again and "canvas" not in again and any(s.startswith("transform") for s in again)


def test_the_template_sets_it(tmp_path):
    import pandas as pd
    from zfisher.core import generate_batch_template as gbt
    write_rounds(tmp_path)
    sheets = gbt.build_template_sheets()
    book = tmp_path / "b.xlsx"
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        pd.DataFrame([{"Name": "x", "R1": str(tmp_path / "R1.ome.tif"), "R2": str(tmp_path / "R2.ome.tif"),
                       "Remove_Border_Nuclei": True}], columns=sheets["Datasets"].columns).to_excel(
            writer, sheet_name="Datasets", index=False)
    config, error = gbt.parse_batch_config(book)
    assert error is None and config.datasets[0].consensus.remove_border_nuclei is True
    assert gbt.pipeline_kwargs(config.datasets[0])["remove_border_nuclei"] is True
