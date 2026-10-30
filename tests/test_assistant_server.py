"""The MCP server started from Home (``zfisher.ui.assistant_server``).

A client on another thread talks to the server over HTTP, as a client on
the same computer would, while the main thread runs the GUI. A change to the
open session runs on the GUI's own session (no second session file), with the
puncta layer on screen updated from the file the change wrote; a change that
would rewrite the session is refused for the open session.

Headless: a napari ``ViewerModel`` behind the attributes the GUI uses, on a
small batch session (``test_stages``).

Run:  python -m pytest tests/test_assistant_server.py -q
"""
import asyncio
import contextlib
import os
import threading
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from zfisher import constants
from zfisher.core import api, io, session, stages

from test_stages import CHANNEL, finished, reopened  # noqa: F401  (fixtures)

pytest.importorskip("zfisher_mcp")

R1_PUNCTA = stages.raw_puncta_name("R1", CHANNEL)


class _Dialog(contextlib.nullcontext):
    def __init__(self, *a, **k):
        super().__init__(self)

    def update_progress(self, *a):
        pass


@pytest.fixture
def gui(reopened, monkeypatch):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import numpy as np
    from napari.components import ViewerModel
    from zfisher.ui import popups, viewer_helpers

    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

        def __setattr__(self, name, value):
            setattr(model, name, value)

    viewer = _Viewer()
    monkeypatch.setattr(popups, "ProgressDialog", _Dialog)
    voxels = io.image_identity(io.resolve_raw_path("R1"))["voxels"]
    image = model.add_image(np.zeros((4, 8, 8), np.uint16), name=f"R1 - {CHANNEL}", scale=voxels)
    csv = constants.puncta_csv_path(reopened.reports, R1_PUNCTA)
    viewer_helpers.show_puncta_csv(viewer, R1_PUNCTA, csv, image)
    yield types.SimpleNamespace(viewer=viewer, model=model, app=app, out=reopened.out, csv=csv)


def _client(url, token, calls):
    """Run ``calls`` (name, arguments) over HTTP on a thread of its own."""
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    from mcp.types import Implementation

    async def go():
        transport = streamable_http_client(url, http_client=create_mcp_http_client(
            headers={"Authorization": f"Bearer {token}"}))
        async with Client(transport, client_info=Implementation(name="test-client", version="0")) as client:
            return [await client.call_tool(name, arguments) for name, arguments in calls]
    return asyncio.run(go())


def _while_the_gui_runs(app, work):
    """``work()`` on another thread while this one processes GUI events."""
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("value", work()), daemon=True)
    thread.start()
    while thread.is_alive():
        app.processEvents()
        thread.join(timeout=0.02)
    return out["value"]


def test_a_client_changes_the_open_session_through_the_gui(gui):
    from zfisher.ui import assistant_server
    sessions = gui.out / constants.SESSIONS_DIR
    before_files = sorted(p.name for p in sessions.iterdir())
    before_points = len(gui.model.layers[R1_PUNCTA].data)
    server = assistant_server.AssistantServer(gui.viewer)
    url, token = server.start(gui.out.parent, "operate")
    try:
        info, detect, refused = _while_the_gui_runs(gui.app, lambda: _client(url, token, [
            ("get_session_info", {"session": "out"}),
            ("detect_channel", {"session": "out", "round": "R1", "channel": CHANNEL, "dry_run": False,
                                "params": {"method": "Local Maxima", "threshold_rel": 0.5, "min_distance": 2}}),
            ("update", {"session": "out", "dry_run": False})]))
    finally:
        server.stop()
    assert not server.running
    assert info.structured_content["session_file"] == session.default().get("session_filename")
    assert not detect.is_error, detect.content[0].text
    written = pd.read_csv(gui.csv)
    assert detect.structured_content["result"]["n_puncta"] == len(written)
    assert len(gui.model.layers[R1_PUNCTA].data) == len(written) != before_points     # the layer shows the file
    assert sorted(p.name for p in sessions.iterdir()) == before_files                 # the GUI's session, no copy
    recorded = stages.recorded_detection_parameters("R1", CHANNEL)
    assert recorded["threshold_rel"] == 0.5
    entry = api.get_decisions(session.default(), "detection")[-1]
    assert (entry["actor"], entry["via"], entry["new"]["sensitivity"]) == ("agent", "mcp:test-client", 0.5)
    assert refused.is_error and "would rewrite the session open in zFISHer" in refused.content[0].text
    assert "The assistant ran detect R1" in gui.model.status


def test_a_mode_must_be_one_of_the_three(gui):
    from zfisher.ui import assistant_server
    assert assistant_server.available()
    with pytest.raises(ValueError, match="query, triage, operate"):
        assistant_server.AssistantServer(gui.viewer).start(gui.out.parent, "admin")
