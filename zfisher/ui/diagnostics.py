"""Keep the errors napari catches.

napari catches exceptions raised in its event callbacks and in Qt overrides
(through ``sys.excepthook``) and turns each into a notification. zFISHer
blocks napari's notification toasts, so those errors used to vanish. They are
now written, with their tracebacks, to the log (the session log once one is
open); the toasts stay blocked.
"""
import logging

logger = logging.getLogger(__name__)



def log_caught_errors(manager=None):
    """Log every error and warning ``manager`` (napari's notification
    manager) receives, before its blocked notification. Safe to call twice."""
    if manager is None:
        from napari.utils.notifications import notification_manager as manager
    if getattr(manager.dispatch, "_zfisher_logs", False):
        return
    original = manager.dispatch

    def dispatch(notification):
        exc = getattr(notification, "exception", None)
        if isinstance(exc, BaseException):
            logger.error("Error caught by napari: %s: %s", type(exc).__name__, exc,
                         exc_info=(type(exc), exc, exc.__traceback__))
        else:
            logger.warning("napari: %s", getattr(notification, "message", notification))
        return original(notification)
    dispatch._zfisher_logs = True
    manager.dispatch = dispatch
