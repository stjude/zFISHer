"""
Main-thread stall monitor.

A ``QTimer`` ticks every 50 ms on the Qt main thread. A timer cannot fire
while the thread is blocked, so a gap between ticks longer than the interval
plus ``threshold_ms`` means the event loop was stuck for that long. Each such
gap writes one ``STALL`` line to the session log, naming the stage or action
that was running (from ``zfisher.core.timing``) or ``untracked`` when the
block came from something no timer wraps: a napari re-slice, a texture
upload, garbage collection.

A hang that never returns is never observed here; the gap is only measured at
the first tick after the block ends.
"""

import logging
import time

from qtpy.QtCore import QObject, Qt, QTimer
from qtpy.QtWidgets import QApplication

from ..core import timing

logger = logging.getLogger(__name__)

INTERVAL_MS = 50
THRESHOLD_MS = 100


class StallMonitor(QObject):
    def __init__(self, viewer=None, interval_ms=INTERVAL_MS, threshold_ms=THRESHOLD_MS, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._interval_ms = interval_ms
        self._threshold_ms = threshold_ms
        self._last_tick = None
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._last_tick = time.perf_counter()
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def is_running(self):
        return self._timer.isActive()

    def _during(self, block_start):
        current = timing.current_activity()
        if current:
            return current
        last = timing.last_activity()
        # The work that blocked the loop has usually finished by the time the
        # late tick runs; it counts if it ended inside the gap.
        if last and last[1] >= block_start:
            return last[0]
        return "untracked"

    def _view_fields(self):
        fields = {}
        viewer = self._viewer
        if viewer is None:
            return fields
        try:
            fields["ndisplay"] = viewer.dims.ndisplay
            fields["visible_layers"] = sum(1 for layer in viewer.layers if layer.visible)
        except Exception:
            pass
        return fields

    def _tick(self):
        now = time.perf_counter()
        previous, self._last_tick = self._last_tick, now
        if previous is None:
            return
        blocked_ms = (now - previous) * 1000 - self._interval_ms
        if blocked_ms <= self._threshold_ms:
            return
        try:
            fields = [("blocked_ms", f"{blocked_ms:.0f}"), ("during", self._during(previous))]
            fields.extend(self._view_fields().items())
            app = QApplication.instance()
            if app is not None:
                # macOS throttles timers of a backgrounded app, which reads as
                # a stall that the user never saw.
                fields.append(("app_active", int(app.applicationState() == Qt.ApplicationState.ApplicationActive)))
            logger.info(timing.format_timing_line("STALL", fields))
        except Exception:
            logger.debug("Stall line failed", exc_info=True)


_monitor = None


def start(viewer=None):
    """Start the process-wide monitor (idempotent) and stop it when the app quits."""
    global _monitor
    if _monitor is None:
        _monitor = StallMonitor(viewer)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(stop)
    _monitor.start()
    return _monitor


def stop():
    if _monitor is not None:
        _monitor.stop()
