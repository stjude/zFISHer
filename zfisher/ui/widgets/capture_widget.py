import napari
from pathlib import Path
from magicgui import magicgui, widgets

import zfisher.core.session as session
from .. import popups

# --- State for auto-incrementing filename ---
capture_count = 1

def _get_next_filename():
    """Returns the next available 'captureX.png' filename, or None if no session."""
    global capture_count
    
    output_dir = session.get_data("output_dir")
    if not output_dir:
        return None
    
    captures_dir = Path(output_dir) / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    while True:
        filename = f"capture{capture_count}.png"
        if not (captures_dir / filename).exists():
            return filename
        capture_count += 1

def _capture_view(output_filename: str):
    """Core logic to capture the view."""
    viewer = napari.current_viewer()
    
    if not viewer:
        print("Error: No napari viewer found.")
        return

    try:
        output_dir = session.get_data("output_dir")
        if not output_dir:
            popups.show_error_popup(
                viewer.window._qt_window,
                "No Active Session",
                "Please start or load a session to enable captures."
            )
            return
            
        captures_dir = Path(output_dir) / "captures"
        captures_dir.mkdir(parents=True, exist_ok=True)
            
        save_path = captures_dir / output_filename
        
        if save_path.exists():
            next_name = _get_next_filename()
            popups.show_error_popup(
                viewer.window._qt_window,
                "File Exists",
                f"The file '{output_filename}' already exists. The filename has been updated to '{next_name}'. Please try again."
            )
            capture_widget.output_filename.value = next_name
            return

        viewer.screenshot(str(save_path))
        
        print(f"Saved screenshot to {save_path}")
        viewer.status = f"Saved screenshot: {save_path.name}"
        
        # Update filename for the next capture
        global capture_count
        capture_count += 1
        capture_widget.output_filename.value = _get_next_filename()
        
    except Exception as e:
        print(f"Capture failed: {e}")
        viewer.status = "Capture failed (check console)."
        popups.show_error_popup(
            viewer.window._qt_window,
            "Capture Failed",
            f"An error occurred during capture.\n\nError: {e}"
        )

@magicgui(
    call_button="Capture View",
    layout="vertical",
    output_filename={"label": "Filename:"}
)
def capture_widget(output_filename: str):
    """Magicgui widget to capture the current viewer canvas."""
    _capture_view(output_filename)

# --- Hotkey setup ---
def capture_with_hotkey(viewer: napari.Viewer):
    """Wrapper to call capture from a hotkey."""
    # Use the filename currently in the widget's textbox
    filename = capture_widget.output_filename.value
    _capture_view(filename)

# Add hotkey information and initialize filename
capture_widget.insert(0, widgets.Label(value="Hotkey: P (press in canvas)"))
initial_filename = _get_next_filename()
capture_widget.output_filename.value = initial_filename if initial_filename else "capture1.png"
