"""Dropping puncta at the ends of the Z stack (plan item C5).

``z_edge_exclude`` drops puncta within that many planes of the first and last
plane, where spots are cut by the stack's ends. It is a detection setting
like the others: in the model, the widget, the template's Puncta sheet and
the settings the report lists. 0, the default, changes nothing.

Run:  python -m pytest tests/test_z_edge.py -q
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from zfisher.core import puncta
from zfisher.core.params import DetectionParams


def _volume():
    image = np.full((12, 30, 30), 10.0)
    for z, y, x in [(0, 5, 5), (1, 5, 20), (5, 15, 15), (10, 25, 5), (11, 25, 25)]:
        image[z, y - 1:y + 2, x - 1:x + 2] = 500.0
    return image


def test_spots_near_the_ends_are_dropped_and_ids_kept():
    params = {"method": "Local Maxima", "threshold_rel": 0.5, "min_distance": 2, "nuclei_only": False}
    every = puncta.process_puncta_detection(_volume(), voxels=(0.3, 0.1, 0.1), params=dict(params))
    trimmed = puncta.process_puncta_detection(_volume(), voxels=(0.3, 0.1, 0.1),
                                              params=dict(params, z_edge_exclude=2))
    assert sorted(every[:, 0].astype(int)) == [0, 1, 5, 10, 11]
    assert trimmed[:, 0].astype(int).tolist() == [5]
    kept_id = trimmed[0, 6]
    assert kept_id in every[:, 6] and every[every[:, 6] == kept_id][0, 0] == 5   # the same id as without the rule


def test_off_by_default_and_recorded_only_when_set():
    assert DetectionParams().z_edge_exclude == 0
    assert "z_edge_exclude" not in DetectionParams().detection_kwargs()
    assert DetectionParams(z_edge_exclude=5).detection_kwargs()["z_edge_exclude"] == 5
    assert "z_edge_exclude" not in puncta.detection_settings_row({"method": "Local Maxima"}, 3, "raw")
    assert puncta.detection_settings_row({"z_edge_exclude": 5}, 3, "raw")["z_edge_exclude"] == 5


def test_the_template_column_reaches_the_settings(tmp_path):
    import pandas as pd
    from zfisher.core import generate_batch_template as gbt
    from synthetic_rounds import write_rounds
    write_rounds(tmp_path)
    sheets = gbt.build_template_sheets()
    book = tmp_path / "b.xlsx"
    with pd.ExcelWriter(book, engine="openpyxl") as writer:
        pd.DataFrame([{"Name": "x", "R1": str(tmp_path / "R1.ome.tif"), "R2": str(tmp_path / "R2.ome.tif")}],
                     columns=sheets["Datasets"].columns).to_excel(writer, sheet_name="Datasets", index=False)
        pd.DataFrame([{"Channel": "FITC", "Z_Edge_Exclude": 5}, {"Channel": "Cy5", "Z_Edge_Exclude": 50}],
                     columns=sheets["Puncta"].columns).to_excel(writer, sheet_name="Puncta", index=False)
    config, error = gbt.parse_batch_config(book)
    assert error is None
    fitc, cy5 = (gbt.resolve_puncta_for_dataset("x", config.puncta_rules)[c] for c in ("FITC", "Cy5"))
    assert fitc["z_edge_exclude"] == 5 and cy5["z_edge_exclude"] == 20       # clamped to the range
    assert any("Z_Edge_Exclude" in w and "Clamped" in w for w in config.warnings)


def test_the_widget_has_the_control_and_its_sections_are_in_place():
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from zfisher.ui.widgets import puncta_widget as pw
    control = pw._puncta_widget.z_edge_exclude
    assert (control.value, control.min, control.max) == (0, 0, 20)
    layout = pw._puncta_widget.native.layout()
    tophat = layout.indexOf(pw._puncta_widget.use_tophat.native)
    assert layout.indexOf(pw._bg_header) == tophat - 2 and layout.indexOf(pw._bg_desc) == tophat - 1
    assert layout.indexOf(control.native) < layout.indexOf(pw._bg_header)    # with the detection settings
