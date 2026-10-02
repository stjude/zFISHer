import logging
from functools import wraps
import napari
from ..core import session, timing
from . import popups

logger = logging.getLogger(__name__)

def require_active_session(message="Please start or load a session first."):
    """
    A decorator that checks if a zFISHer session is active before executing a function.

    If no session is active (i.e., no output_dir is set), it shows an error
    popup and stops the function from running.

    This is useful for any widget function that operates on session data.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not session.get_data("output_dir"):
                try:
                    viewer = napari.current_viewer()
                    popups.show_error_popup(viewer.window._qt_window, "No Active Session", message)
                except Exception as e:
                    logger.error("Could not show 'No Active Session' popup: %s", e)
                return None  # Stop execution
            return func(*args, **kwargs)
        return wrapper
    return decorator

def error_handler(title="An Error Occurred"):
    """
    A decorator to catch exceptions in widget functions and show an error popup.

    The call is timed as one ``ACTION`` line in the session log, named after
    ``title`` without its "Failed" suffix (``widget:registration``).
    """
    action = "widget:" + title.removesuffix(" Failed").strip().lower().replace(" ", "_")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                logger.debug("Starting %s", action)
                with timing.action_timer(action):
                    return func(*args, **kwargs)
            except Exception as e:
                logger.error("Error in %s: %s", func.__name__, e, exc_info=True)
                try:
                    viewer = napari.current_viewer()
                    if viewer:
                        viewer.status = f"Error: {e}"
                        popups.show_error_popup(viewer.window._qt_window, title, str(e))
                except Exception as popup_e:
                    logger.error("Could not show error popup: %s", popup_e)
        return wrapper
    return decorator