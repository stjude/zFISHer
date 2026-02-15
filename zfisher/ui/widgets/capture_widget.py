import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui, widgets
from packaging.version import parse as parse_version

import zfisher.core.session as session
from .. import popups
from ..decorators import require_active_session

# --- Arrow Drawing ---

class ArrowDrawer:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.start_pos = None
        self.arrows_layer = None
        self._is_active = False
        self._save_callback = self._create_save_callback()

    def _create_save_callback(self):
        def _save_arrows_data(event=None):
            out_dir = session.get_data("output_dir")
            if out_dir and self.arrows_layer:
                seg_dir = Path(out_dir) / "segmentation"
                seg_dir.mkdir(exist_ok=True, parents=True)
                arrows_path = seg_dir / f"{self.arrows_layer.name}.npy"
                np.save(arrows_path, self.arrows_layer.data)
                session.set_processed_file(self.arrows_layer.name, str(arrows_path), layer_type='vectors')
        return _save_arrows_data

    def _get_or_create_layer(self):
        if self.arrows_layer and self.arrows_layer.name in self.viewer.layers:
            return self.arrows_layer
        
        for layer in self.viewer.layers:
            if layer.name == "Arrows" and isinstance(layer, napari.layers.Vectors):
                self.arrows_layer = layer
                # Ensure listener is attached
                if self._save_callback not in self.arrows_layer.events.data.callbacks:
                    self.arrows_layer.events.data.connect(self._save_callback)
                return layer

        vector_params = {
            'data': np.empty((0, 2, self.viewer.dims.ndim)),
            'name': "Arrows",
            'opacity': 1.0,
            'edge_width': 1,
            'length': 1,
            'edge_color': 'white',
        }
        
        # Defensively add new arrow head styling parameters based on napari version
        # to ensure consistent appearance with loaded arrows.
        if parse_version(napari.__version__) >= parse_version("0.7.0"):
            # Use smaller values to prevent the head from overpowering the tail
            vector_params['head_width'] = 4
            vector_params['head_length'] = 6

        self.arrows_layer = self.viewer.add_vectors(**vector_params)
        self.arrows_layer.events.data.connect(self._save_callback)
        return self.arrows_layer

    def on_mouse_drag(self, viewer, event):
        """A generator function to handle mouse dragging for drawing arrows."""
        if not self._is_active:
            return

        # On mouse press, handle right-click to delete. This does not require a modifier.
        if event.type == 'mouse_press' and event.button == 2:
            layer = self._get_or_create_layer()
            if len(layer.data) > 0:
                layer.data = layer.data[:-1]
                self.viewer.status = "Removed last arrow."
            # We yield to consume the event and prevent a context menu from appearing.
            yield
            return

        # For drawing, require Shift + Left-Click to avoid conflict with default panning.
        # If the modifier is not present, we return immediately, allowing panning to occur.
        if 'Shift' not in event.modifiers or event.button != 1:
            return

        # --- on mouse press (event.type == 'mouse_press') ---
        layer = self._get_or_create_layer()
        cursor_pos = np.array(event.position)
        self.start_pos = cursor_pos

        # A new arrow is a zero-length vector at the start position.
        # The data format is [start_point, vector_from_start].
        new_arrow_data = np.array([self.start_pos, np.zeros_like(self.start_pos)])
        layer.data = np.vstack([layer.data, [new_arrow_data]])
        self.viewer.status = "Release mouse to finish arrow."
        yield

        # --- on mouse move ---
        while event.type == 'mouse_move':
            cursor_pos = np.array(event.position)
            # Update the vector component of the arrow data.
            vector = cursor_pos - self.start_pos
            layer.data[-1, 1, :] = vector
            # Force a refresh to ensure the view updates.
            layer.refresh()
            yield

        # --- on mouse release ---
        # The data is already in the correct [start, vector] format.
        # We just need to finalize the state.
        self.start_pos = None
        self.viewer.status = "Arrow drawn."

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            self._get_or_create_layer()
            self.viewer.status = "Arrow drawing ON. Hold Shift & drag to draw, right-click to remove last."
            # Insert the callback at the beginning of the list to capture the event
            # before napari's default pan/zoom handlers can.
            if self.on_mouse_drag not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.insert(0, self.on_mouse_drag)
        else:
            # If we were in the middle of drawing, cancel it
            if self.start_pos is not None:
                # Remove the temporary arrow that was being drawn
                if self.arrows_layer and len(self.arrows_layer.data) > 0:
                    self.arrows_layer.data = self.arrows_layer.data[:-1]
                self.start_pos = None
            
            self.viewer.status = "Arrow drawing OFF."
            if self.on_mouse_drag in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self.on_mouse_drag)

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

@require_active_session("Please start or load a session to enable captures.")
def _capture_view(viewer: napari.Viewer, output_filename: str):
    """Core logic to capture the view."""
    
    if not viewer:
        print("Error: No napari viewer found.")
        return

    try:
        output_dir = session.get_data("output_dir") # We know this exists due to the decorator
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

        # Use Qt's grab method to include custom widgets like the scale bar
        canvas_qwidget = viewer.window.qt_viewer.canvas.native
        pixmap = canvas_qwidget.grab()
        pixmap.save(str(save_path))
        
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
    viewer = napari.current_viewer()
    _capture_view(viewer, output_filename)

# --- Hotkey setup ---
def capture_with_hotkey(viewer: napari.Viewer):
    """Wrapper to call capture from a hotkey."""
    # Use the filename currently in the widget's textbox
    filename = capture_widget.output_filename.value
    _capture_view(viewer, filename)

# --- Widget Setup ---
# Add hotkey information and initialize filename
hotkey_container = widgets.Container(layout="horizontal", labels=False)
hotkey_container.append(widgets.Label(value="Hotkey: Shift+P (press in canvas)"))
capture_widget.insert(0, hotkey_container)

initial_filename = _get_next_filename()
capture_widget.output_filename.value = initial_filename if initial_filename else "capture1.png"

# Add Arrow drawing tool
arrow_container = widgets.Container(layout="horizontal", labels=False)
arrow_chk = widgets.CheckBox(text="Draw Arrows")
arrow_container.append(arrow_chk)
capture_widget.append(arrow_container)

# Add Scale Bar Options
sb_container = widgets.Container(layout="horizontal", labels=False)
sb_label = widgets.Label(value="<b>Scalebar:</b>")
sb_lock = widgets.CheckBox(text="Lock")
sb_pixels = widgets.CheckBox(text="Show Pixels")

sb_container.extend([sb_label, sb_lock, sb_pixels])
capture_widget.append(sb_container)

@sb_lock.changed.connect
def _on_sb_lock(state: bool):
    viewer = napari.current_viewer()
    if viewer and hasattr(viewer.window, 'custom_scale_bar'):
        viewer.window.custom_scale_bar.locked = state
        if not state:
            viewer.status = "Scale Bar Unlocked: Hold Right-Click to drag."
        else:
            viewer.status = "Scale Bar Locked."

@sb_pixels.changed.connect
def _on_sb_pixels(state: bool):
    viewer = napari.current_viewer()
    if viewer and hasattr(viewer.window, 'custom_scale_bar'):
        viewer.window.custom_scale_bar.show_pixels = state
        viewer.window.custom_scale_bar.recalculate() # Recalculate text

# This needs a viewer instance, so we can't do it until the viewer is created.
# A bit of a hack: we'll check for the viewer when the checkbox is clicked.
arrow_drawer = None

@arrow_chk.changed.connect
def _on_arrow_draw_toggled(state: bool):
    global arrow_drawer
    viewer = napari.current_viewer()
    if not viewer:
        arrow_chk.value = False
        print("Cannot activate arrow drawing without a viewer.")
        return
    
    if arrow_drawer is None:
        arrow_drawer = ArrowDrawer(viewer)
        
    arrow_drawer.set_active(state)
