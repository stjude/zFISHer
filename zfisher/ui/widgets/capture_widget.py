
import napari
from pathlib import Path
from magicgui import magicgui

import zfisher.core.session as session
from .. import popups

@magicgui(
    call_button="Capture View",
    output_filename={"label": "Filename (.png)", "value": "capture.png"}
)
def capture_widget(output_filename: str = "capture.png"):
    """Captures the current viewer canvas to a file."""
    viewer = napari.current_viewer()
    
    if not viewer:
        print("Error: No napari viewer found.")
        return

    dialog = popups.ProgressDialog(viewer.window._qt_window, "Capturing View...")

    try:
        output_dir = session.get_data("output_dir")
        if not output_dir:
            output_dir = Path.home()
        save_path = Path(output_dir) / output_filename
        
        viewer.screenshot(str(save_path))
        
        print(f"Saved screenshot to {save_path}")
        viewer.status = f"Saved screenshot: {save_path.name}"
        
        popups.show_info_popup(
            viewer.window._qt_window,
            "Capture Complete",
            f"Screenshot saved successfully.\n\nFile: {save_path.name}\nPath: {save_path}"
        )
                 
    except Exception as e:
        print(f"Capture failed: {e}")
        viewer.status = "Capture failed (check console)."
        popups.show_error_popup(
            viewer.window._qt_window,
            "Capture Failed",
            f"An error occurred during capture.\n\nError: {e}"
        )
    finally:
        dialog.close()
