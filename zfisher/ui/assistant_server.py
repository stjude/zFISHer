"""The MCP server run from inside zFISHer, for a client on this machine.

Home's Assistant Server section starts ``zfisher_mcp``'s server (the
separate ``packages/zfisher-mcp`` package; the section says so when it is not
installed) on ``http://127.0.0.1:<port>/mcp`` behind a token made fresh for
each start. Its folder is the one holding the open session, so the open
session is one of the sessions a client sees.

``GuiHost`` is the server's way into this process. The open session is
served from memory, never as a second copy read from disk. A change a client
makes to it (in operate mode) is handed to the main thread, which saves any
edits waiting to be saved, runs the change on a worker thread under a
progress dialog (so nothing else writes meanwhile), and then shows the puncta
files the change rewrote. Changes that would rewrite the session itself
(``update``, ``run_batch``) are refused for the open session. Other sessions
are read from disk and changed directly, as the headless server does.
"""
import concurrent.futures
import importlib.util
import logging
import threading
from pathlib import Path

from qtpy.QtCore import QObject, Qt, Signal

from ..core import session
from .workers import run_on_worker

logger = logging.getLogger(__name__)

MODES = ("query", "triage", "operate")
REFUSED_ON_OPEN = ("update", "run_batch")


def available():
    """True when the ``zfisher_mcp`` package is installed."""
    return importlib.util.find_spec("zfisher_mcp") is not None


class GuiHost(QObject):
    """The server's host (``zfisher_mcp.host``) inside zFISHer."""

    _job = Signal(object)

    def __init__(self, viewer):
        super().__init__()
        from zfisher_mcp.host import DiskHost
        self.viewer = viewer
        self._disk = DiskHost()
        self._job.connect(self._run_job, Qt.QueuedConnection)

    def open_folder(self):
        out = session.default().get("output_dir")
        return Path(out).resolve() if out else None

    def _is_open(self, folder):
        return self.open_folder() is not None and Path(folder).resolve() == self.open_folder()

    def session(self, folder, writable=False):
        if self._is_open(folder):
            return session.default()
        return self._disk.session(folder, writable)

    def run_write(self, what, folder, fn):
        if not self._is_open(folder):
            return fn()
        if what.split()[0] in REFUSED_ON_OPEN:
            raise ValueError(f"{what.split()[0]} would rewrite the session open in zFISHer; close it there "
                             "(Home, Reset) or run it on a copy.")
        if threading.current_thread() is threading.main_thread():
            raise RuntimeError("A change to the open session was asked for on the main thread.")
        future = concurrent.futures.Future()
        self._job.emit((what, fn, future))
        return future.result()

    # --- on the main thread ---------------------------------------------------------

    def _run_job(self, job):
        what, fn, future = job
        try:
            future.set_result(self._run_here(what, fn))
        except BaseException as exc:          # the client hears of it; zFISHer carries on
            future.set_exception(exc)

    def _run_here(self, what, fn):
        from . import events, popups
        from .widgets.mask_editor_widget import flush_pending_mask_saves
        import napari
        from .. import constants
        for layer in list(self.viewer.layers):         # the files must hold what is on screen
            if isinstance(layer, napari.layers.Points) and layer.name.endswith(constants.PUNCTA_SUFFIX):
                events.save_puncta_layer(layer, report=False)
        flush_pending_mask_saves()
        before = self._puncta_files()
        with popups.ProgressDialog(self.viewer.window._qt_window, f"Assistant: {what}...") as dialog:
            dialog.update_progress(10, f"Assistant: {what}...")
            result = run_on_worker(fn)
            dialog.update_progress(90, "Showing the files it changed...")
            shown = self._show_changed(before)
        self.viewer.status = f"The assistant ran {what}" + (f"; updated {', '.join(shown)}." if shown else ".")
        logger.info("ASSISTANT %s ran; layers updated: %s", what, ", ".join(shown) or "none")
        return result

    def _puncta_files(self):
        from ..core import stages
        files = session.default().get("processed_files") or {}
        return {name: stages.file_state(info["path"]) for name, info in files.items()
                if isinstance(info, dict) and info.get("subtype") == "puncta_csv" and info.get("path")}

    def _show_changed(self, before):
        """Reload every puncta layer on screen whose CSV the change rewrote."""
        from . import viewer_helpers
        files = session.default().get("processed_files") or {}
        shown = []
        for name, state in self._puncta_files().items():
            if state == before.get(name) or name not in self.viewer.layers:
                continue
            layer = self.viewer.layers[name]
            viewer_helpers.show_puncta_csv(self.viewer, name, files[name]["path"], layer)
            shown.append(name)
        return shown


class AssistantServer:
    """Start and stop the server for ``viewer``."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.host = None
        self.http = None
        self.root = None
        self.mode = None

    @property
    def running(self):
        return self.http is not None and self.http.running

    def start(self, root, mode="query"):
        """Serve the sessions under ``root`` in ``mode``; returns ``(url, token)``."""
        from zfisher_mcp.httpserver import HttpServer
        from zfisher_mcp.paths import Roots
        from zfisher_mcp.server import ServerConfig, build_server
        if mode not in MODES:
            raise ValueError(f"No mode '{mode}'; the modes are {', '.join(MODES)}.")
        self.stop()
        self.host = self.host or GuiHost(self.viewer)
        server = build_server(ServerConfig(roots=Roots({"": root}), mode=mode, actor="agent", host=self.host))
        self.http = HttpServer(server).start()
        self.root, self.mode = Path(root), mode
        logger.info("ASSISTANT server started in %s mode at %s for %s", mode, self.http.url, root)
        return self.http.url, self.http.token

    def stop(self):
        if self.http is not None:
            self.http.stop()
            logger.info("ASSISTANT server stopped")
        self.http = None
