import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui, widgets

import zfisher.core.session as session
from .. import popups

# --- Arrow Drawing ---

class SafeShapes(napari.layers.Shapes):
    """A Shapes layer that suppresses errors during status updates (e.g. 3D ray intersection bugs)."""
    def get_status(self, position, *args, **kwargs):
        try:
            return super().get_status(position, *args, **kwargs)
        except Exception:
            # Return None or empty dict to avoid crashing the status checker
            return None

class ArrowDrawer:
    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.start_pos = None
        self.arrows_layer = None
        self._is_active = False

    def _calculate_arrow_geometry(self, start, end):
        vector = end - start
        length = np.linalg.norm(vector)
        if length == 0:
            return None, None
            
        direction = vector / length
        
        # Head is 25% of length
        head_len = length * 0.25
        head_width = head_len * 0.5
        
        base = end - direction * head_len
        
        perp = np.zeros_like(vector)
        if self.viewer.dims.ndisplay == 3:
            view_dir = np.array(self.viewer.camera.view_direction)
            cross = np.cross(direction, view_dir)
            norm_cross = np.linalg.norm(cross)
            if norm_cross > 1e-6:
                perp = cross / norm_cross
            else:
                perp = np.zeros_like(vector)
                perp[0] = 1 
        else:
            # 2D mode: rotate 90 degrees in the last 2 dimensions
            perp[-2] = -direction[-1]
            perp[-1] = direction[-2]
            
        wing1 = base + perp * head_width
        wing2 = base - perp * head_width
        
        # Shaft: Start -> Base
        shaft = np.array([start, base])
        # Head: End -> Wing1 -> Wing2 (Triangle)
        head = np.array([end, wing1, wing2])
        
        return shaft, head

    def _get_or_create_layer(self):
        if self.arrows_layer and self.arrows_layer.name in self.viewer.layers:
            return self.arrows_layer
        
        for layer in self.viewer.layers:
            if layer.name == "Arrows" and isinstance(layer, napari.layers.Shapes):
                self.arrows_layer = layer
                return layer

        # Use SafeShapes instead of viewer.add_shapes to prevent 3D status crashes
        self.arrows_layer = SafeShapes(
            ndim=self.viewer.dims.ndim,
            name="Arrows",
            edge_width=2,
            edge_color='white',
            face_color='white',
            opacity=1.0
        )
        self.viewer.add_layer(self.arrows_layer)
        return self.arrows_layer

    def on_mouse_click(self, viewer, event):
        if not self._is_active:
            return

        # Right-click to remove last arrow
        if event.button == 2:
            if self.arrows_layer and len(self.arrows_layer.data) > 0:
                if len(self.arrows_layer.data) >= 2:
                    self.arrows_layer.data = self.arrows_layer.data[:-2]
                else:
                    self.arrows_layer.data = []
                self.viewer.status = "Removed last arrow."
            return
        
        # Only left-click from here
        if event.button != 1:
            return

        cursor_pos = np.array(event.position)
        
        if self.start_pos is None:
            self.start_pos = cursor_pos
            self.viewer.status = "Arrow start set. Click again for the end."
        else:
            end_pos = cursor_pos
            vector = end_pos - self.start_pos
            
            # Project vector onto the plane perpendicular to view direction in 3D
            if self.viewer.dims.ndisplay == 3:
                view_dir = np.array(self.viewer.camera.view_direction)
                norm = np.linalg.norm(view_dir)
                if norm > 0:
                    view_dir = view_dir / norm
                    projection = np.dot(vector, view_dir)
                    vector = vector - (projection * view_dir)
            
            end_pos = self.start_pos + vector
            shaft, head = self._calculate_arrow_geometry(self.start_pos, end_pos)
            
            if shaft is not None:
                layer = self._get_or_create_layer()
                layer.add([shaft], shape_type='path', edge_width=2, edge_color='white')
                layer.add([head], shape_type='polygon', face_color='white', edge_color='white', edge_width=0)
            
            self.start_pos = None
            self.viewer.status = "Arrow drawn."

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            self._get_or_create_layer()
            self.viewer.status = "Arrow drawing ON. Left-click to set start."
            if self.on_mouse_click not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self.on_mouse_click)
        else:
            self.viewer.status = "Arrow drawing OFF."
            if self.on_mouse_click in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.remove(self.on_mouse_click)

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

# --- Widget Setup ---
# Add hotkey information and initialize filename
capture_widget.insert(0, widgets.Label(value="Hotkey: P (press in canvas)"))
initial_filename = _get_next_filename()
capture_widget.output_filename.value = initial_filename if initial_filename else "capture1.png"

# Add Arrow drawing tool
arrow_chk = widgets.CheckBox(text="Draw Arrows")
capture_widget.append(arrow_chk)

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
