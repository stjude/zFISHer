"""Reset, Load Session and New Session clear the canvas without touching the
open session's files.

Removing a layer by hand backs its file up to ``.bak`` and drops it from the
session JSON (``test_layer_removal_backup.py``). Clearing the canvas to start
over went through the same handler: Home's Reset renamed every file of the
open session and emptied its JSON, Load Session did the same to the session
it was leaving, and loading the open session again found its JSON already
emptied and came back without its processed layers. The files are the
session's and stay as they are; only a layer the user deletes is backed up.

Headless: a napari ``ViewerModel`` whose removals go through zFISHer's
handler, the session module as it is, files in a scratch folder. The last
test opens a small batch session (``test_stages``) through the real Load
Session widget.

Run:  python -m pytest tests/test_reset_keeps_files.py -q
"""
import contextlib
import json
import logging
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from zfisher import constants
from zfisher.core import session

from test_stages import finished, reopened  # noqa: F401  (fixtures)

MASK = "R1 - DAPI_masks"
CENTROIDS = f"{MASK}{constants.CENTROIDS_SUFFIX}"
PUNCTA = "R1 - FITC_puncta"


def _headless_viewer(monkeypatch):
    """A napari ``ViewerModel`` behind the attributes the widgets use, with
    zFISHer's layer lock and removal handler wired as the app wires them."""
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import events

    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

        def __setattr__(self, name, value):
            setattr(model, name, value)

    viewer = _Viewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(events, "QTimer", types.SimpleNamespace(singleShot=lambda _ms, fn: None))
    events.install_layer_lock(model)
    model.layers.events.removed.connect(lambda e: events.on_layer_removed(e, {}))
    return viewer, model


@pytest.fixture
def headless(monkeypatch):
    viewer, model = _headless_viewer(monkeypatch)
    from zfisher.ui import viewer as viewer_module
    suppressed = viewer_module._suppress_custom_controls
    # Loading a session attaches its log, which opens the zfisher loggers up
    # to DEBUG and is not undone by detaching; leave logging as it was found.
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    yield types.SimpleNamespace(viewer=viewer, model=model)
    viewer_module._suppress_custom_controls = suppressed
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture
def canvas(headless, tmp_path):
    """An open session with a mask, its ID and centroid layers and a puncta
    layer: files on disk, registered in the session JSON, shown on the
    headless viewer, the non-puncta layers locked as in the app."""
    from zfisher.ui import events

    out = tmp_path / "session"
    seg = out / constants.SEGMENTATION_DIR
    reports = out / constants.REPORTS_DIR
    seg.mkdir(parents=True)
    reports.mkdir()
    mask_file = seg / f"{MASK}.ome.tif"
    mask_file.write_bytes(b"mask")                     # only its existence matters
    centroids_file = constants.puncta_csv_path(reports, CENTROIDS)
    centroids_file.write_text("id,Z,Y,X\n1,1,2,3\n")
    puncta_file = constants.puncta_csv_path(reports, PUNCTA)
    puncta_file.write_text("puncta_id,Z,Y,X\n0,1,2,3\n")

    session.clear_session()
    session.update_data("output_dir", str(out))
    session.set_processed_file(MASK, str(mask_file), "labels", {"subtype": "mask"})
    session.set_processed_file(f"{MASK}_IDs", "", "points", {"subtype": "computed_ids"})
    session.set_processed_file(CENTROIDS, str(centroids_file), "points")
    session.set_processed_file(PUNCTA, str(puncta_file), "points", {"subtype": "puncta_csv"})

    model = headless.model
    point = np.array([[1.0, 2.0, 3.0]])
    events.lock_layer(model.add_labels(np.zeros((2, 8, 8), np.int32), name=MASK))
    events.lock_layer(model.add_points(point, name=f"{MASK}_IDs"))
    events.lock_layer(model.add_points(point, name=CENTROIDS))
    model.add_points(point, name=PUNCTA)                # puncta stay editable, so deletable
    yield types.SimpleNamespace(
        viewer=headless.viewer, model=model, events=events, out=out,
        files={MASK: mask_file, CENTROIDS: centroids_file, PUNCTA: puncta_file},
        names={MASK, f"{MASK}_IDs", CENTROIDS, PUNCTA},
        json=out / constants.SESSIONS_DIR / constants.SESSION_FILENAME)
    session.clear_session()


def _registered(canvas):
    """The layers the session JSON on disk lists."""
    return set(json.loads(canvas.json.read_text())["processed_files"])


def _untouched(canvas):
    return all(f.exists() and not f.with_name(f.name + ".bak").exists() for f in canvas.files.values())


def test_clearing_the_canvas_keeps_every_file_and_session_entry(canvas):
    canvas.events.clear_layers_keeping_files(canvas.viewer)

    assert len(canvas.model.layers) == 0
    assert _untouched(canvas)
    assert set(session.get_data("processed_files")) == canvas.names
    assert _registered(canvas) == canvas.names
    assert canvas.events._programmatic_removal is False


def test_clearing_leaves_the_flag_as_it_found_it(canvas):
    canvas.events._set_programmatic_removal(True)
    try:
        canvas.events.clear_layers_keeping_files(canvas.viewer)
        assert canvas.events._programmatic_removal is True
    finally:
        canvas.events._set_programmatic_removal(False)


def test_a_layer_the_user_deletes_is_still_backed_up(canvas):
    canvas.model.layers.remove(canvas.model.layers[PUNCTA])

    csv = canvas.files[PUNCTA]
    assert not csv.exists() and csv.with_name(csv.name + ".bak").exists()
    assert PUNCTA not in session.get_data("processed_files") and PUNCTA not in _registered(canvas)
    assert canvas.files[MASK].exists()                  # the other layers are not involved


def test_a_removal_while_a_session_is_being_read_keeps_the_file(canvas):
    session.set_loading(True)
    try:
        canvas.model.layers.remove(canvas.model.layers[PUNCTA])
    finally:
        session.set_loading(False)
    assert _untouched(canvas) and _registered(canvas) == canvas.names


def test_home_reset_keeps_the_open_sessions_files(canvas, monkeypatch):
    from zfisher.ui import popups
    from zfisher.ui.widgets.home_widget import HomeWidget
    monkeypatch.setattr(popups, "show_yes_no_popup", lambda parent, title, text: True)

    HomeWidget(canvas.viewer)._on_reset()

    assert len(canvas.model.layers) == 0
    assert session.get_data("output_dir") is None       # closed in memory
    assert _untouched(canvas) and _registered(canvas) == canvas.names   # whole on disk


class _Dialog(contextlib.nullcontext):
    """Stands in for the progress dialog, which needs a window."""

    def __init__(self, *args, **kwargs):
        super().__init__(self)

    def update_progress(self, *args):
        pass

    def freeze_canvas(self):
        pass


def _load_widget(viewer, monkeypatch, session_file):
    """The real Load Session widget pointed at ``session_file``, its dialogs
    stood in for; returns it with the list of error pop-ups it shows."""
    from zfisher.ui import popups
    from zfisher.ui.widgets import load_session_widget
    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    errors = []
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: errors.append((title, text)))
    monkeypatch.setattr(load_session_widget, "refresh_rules_display", lambda: None)
    monkeypatch.setattr(load_session_widget, "refresh_report_note", lambda: None)
    widget = load_session_widget.LoadSessionWidget(viewer)
    widget._load_session_file.value = session_file
    return widget, errors


def test_load_session_leaves_the_session_it_closes_as_it_was(canvas, monkeypatch, tmp_path):
    other = tmp_path / "other"
    for sub in (constants.SESSIONS_DIR, constants.LOGS_DIR):
        (other / sub).mkdir(parents=True)
    other_file = other / constants.SESSIONS_DIR / constants.SESSION_FILENAME
    other_file.write_text(json.dumps({"output_dir": str(other), "processed_files": {}}))
    widget, errors = _load_widget(canvas.viewer, monkeypatch, other_file)

    widget._on_load_session()

    assert errors == []
    assert session.get_data("output_dir") == str(other) and len(canvas.model.layers) == 0
    assert _untouched(canvas) and _registered(canvas) == canvas.names


def test_loading_the_open_session_again_brings_its_layers_back(headless, reopened, monkeypatch):
    """Load Session on the session that is open, from the file the app is
    writing: the clear before the load used to empty that file, so the
    session came back without its processed layers, their files renamed."""
    from zfisher.ui import viewer_helpers
    registry = session.get_data("processed_files")
    viewer_helpers.restore_processed_layers(headless.viewer, registry, tuple(session.get_data("canvas_scale")),
                                            session.get_data("canvas_offset_pixels"))
    before = sorted(layer.name for layer in headless.model.layers)
    assert before, "the batch session should restore some layers"
    current = reopened.out / constants.SESSIONS_DIR / session.get_data("session_filename")
    widget, errors = _load_widget(headless.viewer, monkeypatch, current)

    widget._on_load_session()

    assert errors == []
    assert sorted(layer.name for layer in headless.model.layers) == before
    assert set(session.get_data("processed_files")) == set(registry)
    assert list(reopened.out.rglob("*.bak")) == []
