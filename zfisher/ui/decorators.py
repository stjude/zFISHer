from functools import wraps
import napari
from zfisher.core import session
from . import popups

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
                    print(f"Could not show 'No Active Session' popup: {e}")
                return None  # Stop execution
            return func(*args, **kwargs)
        return wrapper
    return decorator