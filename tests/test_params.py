"""The typed setting models (``core.params``) and their parity with the GUI.

Every user-facing setting is stated once, in ``core.params``. These tests hold
each model field to the control the GUI shows for it (value, minimum, maximum,
choices), parse every spelling of the segmentation method, accept the
session's ``sensitivity`` key, and keep the JSON schema, from which other
front ends' tool signatures are generated, equal to a checked-in snapshot.

Run:  python -m pytest tests/test_params.py -q
Regenerate the snapshot after a deliberate change:
      python -c "from zfisher.core import params; import json; print(json.dumps(params.schema(), indent=1))" \
        > tests/snapshots/params_schema.json
"""
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from pydantic import ValidationError

from zfisher.core import params, session
from zfisher.core.params import (ConsensusParams, DetectionParams, RegistrationParams, SegMethod)

SNAPSHOT = Path(__file__).parent / "snapshots" / "params_schema.json"

# (model, field) -> (widget module, function gui name, widget parameter)
GUI = {
    (DetectionParams, "method"): ("puncta_widget", "_puncta_widget", "method"),
    (DetectionParams, "threshold_rel"): ("puncta_widget", "_puncta_widget", "threshold"),
    (DetectionParams, "min_distance"): ("puncta_widget", "_puncta_widget", "min_distance"),
    (DetectionParams, "sigma"): ("puncta_widget", "_puncta_widget", "sigma"),
    (DetectionParams, "z_scale"): ("puncta_widget", "_puncta_widget", "z_scale"),
    (DetectionParams, "use_tophat"): ("puncta_widget", "_puncta_widget", "use_tophat"),
    (DetectionParams, "tophat_radius"): ("puncta_widget", "_puncta_widget", "tophat_radius"),
    (DetectionParams, "nuclei_only"): ("puncta_widget", "_puncta_widget", "nuclei_only"),
    (RegistrationParams, "max_ransac_distance"): ("registration_widget", "_registration_widget", "max_distance"),
    (ConsensusParams, "overlap_method"): ("nuclei_matching_widget", "_nuclei_matching_widget", "method"),
    (ConsensusParams, "match_threshold"): ("nuclei_matching_widget", "_nuclei_matching_widget", "match_threshold"),
}
AUTOMATED = {  # the automated preprocessing widget shows the same settings again
    (RegistrationParams, "max_ransac_distance"): "max_distance",
    (RegistrationParams, "apply_warp"): "apply_warp",
    (ConsensusParams, "overlap_method"): "overlap_method",
    (ConsensusParams, "match_threshold"): "match_threshold",
    (ConsensusParams, "remove_extranuclear_puncta"): "remove_outliers",
}


@pytest.fixture(scope="module")
def widgets():
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import importlib
    return lambda module: importlib.import_module(f"zfisher.ui.widgets.{module}")


def _same(control, model, name):
    option = params.magicgui_options(model, [name])[name]
    value = control.value.value if hasattr(control.value, "value") else control.value
    assert value == option["value"], (name, value, option["value"])
    for key in ("min", "max"):
        if key in option:
            assert getattr(control, key) == option[key], (name, key)
    if "choices" in option:
        assert list(control.choices) == option["choices"], name


@pytest.mark.parametrize("key", list(GUI), ids=lambda k: f"{k[0].__name__}.{k[1]}")
def test_each_gui_control_matches_its_model_field(widgets, key):
    module, gui, param = GUI[key]
    _same(getattr(getattr(widgets(module), gui), param), *key)


@pytest.mark.parametrize("key", list(AUTOMATED), ids=lambda k: f"{k[0].__name__}.{k[1]}")
def test_the_automated_preprocessing_controls_match_too(widgets, key):
    gui = widgets("automated_preprocessing_widget")._automated_preprocessing_magic_widget
    _same(getattr(gui, AUTOMATED[key]), *key)


def test_the_colocalization_cutoffs_start_at_the_model_default(widgets):
    module = widgets("colocalization_widget")
    for gui in (module._rule_builder, module._tri_rule_builder):
        assert gui.cutoff.value == params.default(params.ColocRule, "cutoff_um")
        assert gui.cutoff.min == params.field_range(params.ColocRule, "cutoff_um")[0]


@pytest.mark.parametrize("text, method", [
    ("Classical", SegMethod.CLASSICAL), ("classical", SegMethod.CLASSICAL),
    ("Classical (Fast)", SegMethod.CLASSICAL), ("Cellpose", SegMethod.CELLPOSE),
    ("cellpose", SegMethod.CELLPOSE), ("Deep Learning (Cellpose)", SegMethod.CELLPOSE)])
def test_every_spelling_of_the_segmentation_method(text, method):
    assert SegMethod.parse(text) is method


def test_an_unknown_segmentation_method_is_refused():
    with pytest.raises(ValueError, match="Classical or Cellpose"):
        SegMethod.parse("watershed")


def test_the_sessions_sensitivity_key_and_the_ranges():
    p = DetectionParams.model_validate({"sensitivity": 0.2, "method": "Radial Symmetry"})
    assert p.threshold_rel == 0.2 and p.method is params.DetectionMethod.RADIAL_SYMMETRY
    assert "z_scale" not in p.detection_kwargs() and p.detection_kwargs()["method"] == "Radial Symmetry"
    with pytest.raises(ValidationError):
        DetectionParams(threshold_rel=0.0)
    with pytest.raises(ValidationError):
        DetectionParams(unknown=1)


def test_the_schema_matches_the_snapshot():
    assert params.schema() == json.loads(SNAPSHOT.read_text()), (
        "core.params changed its schema; if that was deliberate, regenerate tests/snapshots/params_schema.json "
        "(see this module's docstring)")


def test_sessions_record_their_format_and_refuse_a_newer_one(tmp_path):
    out = tmp_path / "out"
    try:
        session.clear_session()
        assert session.initialize_new_session(out, tmp_path / "r1.tif", tmp_path / "r2.tif")
        assert session.get_data("schema_version") == params.SCHEMA_VERSION
        newer = tmp_path / "newer" / "sessions" / "zfisher_session_1.json"
        newer.parent.mkdir(parents=True)
        newer.write_text(json.dumps({"output_dir": str(newer.parent.parent), "schema_version": 99}))
        with pytest.raises(ValueError, match="newer zFISHer"):
            session.load_session_file(newer)
        older = tmp_path / "older" / "sessions" / "zfisher_session_1.json"
        older.parent.mkdir(parents=True)
        older.write_text(json.dumps({"output_dir": str(older.parent.parent)}))
        session.load_session_file(older)
        assert session.get_data("schema_version") == 1
    finally:
        session.clear_session()
