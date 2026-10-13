"""Tests for the GUI latency instruments that need Qt.

The stall monitor is a 50 ms ``QTimer``; a block on the main thread delays the
next tick, and the gap becomes one ``STALL`` line naming what was running.
Also covered: ``error_handler`` writes a timed ``ACTION`` line, and the timing
wrapper still receives the arguments napari events and Qt buttons send.

Run (in an env with napari):  python -m pytest tests/test_stall_monitor.py -q
"""
import io
import logging
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from zfisher.core import timing


@pytest.fixture(scope="module")
def _qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def records():
    """Timing records written by any zfisher logger during the test."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    log = logging.getLogger("zfisher")
    log.addHandler(handler)
    try:
        yield lambda kind: [f for k, f in (timing.parse_timing_line(l) or (None, None)
                                           for l in stream.getvalue().splitlines()) if k == kind]
    finally:
        log.removeHandler(handler)


def _pump(app, seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.processEvents()
        time.sleep(0.005)


def _monitor(viewer=None):
    from zfisher.ui.stall_monitor import StallMonitor
    return StallMonitor(viewer)


def test_block_on_main_thread_writes_one_stall(_qapp, records):
    mon = _monitor()
    mon.start()
    try:
        _pump(_qapp, 0.2)
        with timing.action_timer("mask_ids_refresh"):
            time.sleep(0.3)
        _pump(_qapp, 0.2)
    finally:
        mon.stop()
    (stall,) = records("STALL")
    assert float(stall["blocked_ms"]) >= 250
    assert stall["during"] == "mask_ids_refresh"


def test_idle_loop_writes_no_stall(_qapp, records):
    mon = _monitor()
    mon.start()
    try:
        _pump(_qapp, 0.5)
    finally:
        mon.stop()
    assert records("STALL") == []


def test_untimed_block_is_reported_as_untracked_with_view_fields(_qapp, records):
    viewer = types.SimpleNamespace(
        dims=types.SimpleNamespace(ndisplay=3),
        layers=[types.SimpleNamespace(visible=True), types.SimpleNamespace(visible=False)],
    )
    mon = _monitor(viewer)
    mon.start()
    try:
        _pump(_qapp, 0.1)
        time.sleep(0.3)
        _pump(_qapp, 0.1)
    finally:
        mon.stop()
    (stall,) = records("STALL")
    assert stall["during"] == "untracked"
    assert stall["ndisplay"] == "3" and stall["visible_layers"] == "1"
    assert stall["app_active"] in ("0", "1")


def test_error_handler_writes_a_timed_action(_qapp, records, monkeypatch):
    import napari
    from zfisher.ui.decorators import error_handler
    monkeypatch.setattr(napari, "current_viewer", lambda: None)

    @error_handler("Registration Failed")
    def ok():
        return 5

    @error_handler("Canvas Generation Failed")
    def fails():
        raise ValueError("bad")

    assert ok() == 5
    assert fails() is None  # still swallowed, as before
    first, second = records("ACTION")
    assert (first["action"], first["status"]) == ("widget:registration", "ok")
    assert (second["action"], second["status"], second["error"]) == (
        "widget:canvas_generation", "error", "ValueError")


def test_napari_events_still_reach_a_timed_handler(_qapp, records):
    from napari.utils.events import EventEmitter
    seen = []

    @timing.timed_action("controls_rebuild")
    def handler(event=None):
        seen.append(event)

    emitter = EventEmitter(source=None, type_name="changed")
    emitter.connect(handler)
    emitter(value=1)
    assert len(seen) == 1 and seen[0] is not None and seen[0].value == 1
    assert records("ACTION")[0]["action"] == "controls_rebuild"


def test_mask_paint_toggle_still_receives_checked(_qapp, records, monkeypatch):
    """PySide6 passes a *args wrapper no arguments; the toggles are connected
    through a lambda so the timed handler still gets ``checked``."""
    import zfisher.ui.widgets.mask_editor_widget as me
    shim = types.SimpleNamespace(mask_layer=types.SimpleNamespace(value=None))
    monkeypatch.setattr(me, "_mask_editor_widget", shim)
    me._paint_toggle_btn.click()  # no layer selected: handler unchecks and returns
    me._erase_toggle_btn.click()
    actions = {f["action"]: f for f in records("ACTION")}
    assert actions["mask_paint_toggle"]["status"] == "ok"
    assert actions["mask_paint_toggle"]["on"] == "True"
    assert actions["mask_erase_toggle"]["status"] == "ok"
    assert not me._paint_toggle_btn.isChecked()
