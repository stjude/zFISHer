import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QPainter, QPen, QBrush, QColor
from qtpy.QtCore import Qt, QRect

from ...core import session
from .. import popups
from ..decorators import require_active_session
from ... import constants

# --- Arrow Drawing (Vectors-based, lightweight, works in 2D and 3D) ---

def _build_vectors(start, end, head_fraction=0.25, head_width_ratio=0.5):
    """Return (N, 2, D) vectors array for one arrow: shaft + two barbs.

    Vectors format: each row is [start_point, direction_vector].
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None

    unit = direction / length
    h_len = np.clip(length * head_fraction, 1.5, length * 0.4)
    h_half_w = h_len * head_width_ratio
    base = end - h_len * unit

    # Perpendicular in YX plane
    ndim = len(direction)
    perp = np.zeros(ndim)
    if ndim >= 3:
        dy, dx = direction[-2], direction[-1]
        yx_len = np.hypot(dy, dx)
        if yx_len > 1e-6:
            perp[-2], perp[-1] = -dx / yx_len, dy / yx_len
        else:
            perp[-2] = 1.0
    else:
        dy, dx = direction[0], direction[1]
        yx_len = np.hypot(dy, dx)
        if yx_len > 1e-6:
            perp[0], perp[1] = -dx / yx_len, dy / yx_len
        else:
            perp[0] = 1.0

    barb_l = base + h_half_w * perp
    barb_r = base - h_half_w * perp

    return np.array([
        [start, direction],          # shaft
        [end, barb_l - end],         # left barb
        [end, barb_r - end],         # right barb
    ])


class ArrowDrawer:
    """Draw arrows on a napari Vectors layer (works in both 2D and 3D)."""

    def __init__(self, viewer: napari.Viewer):
        self.viewer = viewer
        self.start_pos = None
        self.arrows_layer = None
        self._is_active = False
        self._arrow_endpoints = []          # list of (start, end) np arrays
        self._save_callback = self._create_save_callback()

    # ---- persistence ----

    def _create_save_callback(self):
        def _save(event=None):
            out_dir = session.get_data("output_dir")
            if out_dir:
                seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
                seg_dir.mkdir(exist_ok=True, parents=True)
                arrows_path = seg_dir / "Arrows.npy"
                if self._arrow_endpoints:
                    data = np.array([[s, e] for s, e in self._arrow_endpoints])
                else:
                    data = np.empty((0, 2, self.viewer.dims.ndim))
                np.save(arrows_path, data)
                session.set_processed_file(
                    "Arrows", str(arrows_path),
                    layer_type='vectors', metadata={'subtype': 'arrows'}
                )
        return _save

    # ---- layer management ----

    def _get_reference_transform(self):
        """Get scale and translate from the first image layer, or defaults."""
        ndim = self.viewer.dims.ndim
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return np.array(layer.scale), np.array(layer.translate)
        return np.ones(ndim), np.zeros(ndim)

    def _world_to_data(self, world_pos):
        """Convert world coordinates to data coordinates using the reference layer transform."""
        scale, translate = self._get_reference_transform()
        return (np.array(world_pos) - translate) / scale

    def _get_or_create_layer(self):
        if self.arrows_layer and self.arrows_layer.name in self.viewer.layers:
            # Keep scale/translate in sync with image layers
            scale, translate = self._get_reference_transform()
            self.arrows_layer.scale = scale
            self.arrows_layer.translate = translate
            return self.arrows_layer

        for layer in self.viewer.layers:
            if layer.name != "Arrows":
                continue
            if isinstance(layer, napari.layers.Vectors):
                self.arrows_layer = layer
                self._sync_endpoints_from_session()
                scale, translate = self._get_reference_transform()
                self.arrows_layer.scale = scale
                self.arrows_layer.translate = translate
                return layer
            # Old Shapes-based arrows: convert
            if isinstance(layer, napari.layers.Shapes):
                self.viewer.layers.remove(layer)
                self._sync_endpoints_from_session()
                break

        ndim = self.viewer.dims.ndim
        empty = np.empty((0, 2, ndim))
        scale, translate = self._get_reference_transform()
        self.arrows_layer = self.viewer.add_vectors(
            empty, name="Arrows", edge_color='white',
            edge_width=2, opacity=1.0,
            scale=scale, translate=translate,
        )
        if self._arrow_endpoints:
            self._refresh_layer()
            self._save_callback()
        return self.arrows_layer

    def _sync_endpoints_from_session(self):
        if self._arrow_endpoints:
            return
        out_dir = session.get_data("output_dir")
        if not out_dir:
            return
        arrows_path = Path(out_dir) / constants.SEGMENTATION_DIR / "Arrows.npy"
        if arrows_path.exists():
            data = np.load(arrows_path)
            if data.ndim == 3 and data.shape[1] == 2 and data.shape[0] > 0:
                self._arrow_endpoints = [
                    (data[i, 0].copy(), data[i, 1].copy()) for i in range(len(data))
                ]

    # ---- rendering ----

    def _build_all_vectors(self, preview_end=None):
        """Build (N, 2, D) vectors array for all arrows + optional preview."""
        parts = []
        for start, end in self._arrow_endpoints:
            vecs = _build_vectors(start, end)
            if vecs is not None:
                parts.append(vecs)

        if self.start_pos is not None and preview_end is not None:
            # Preview: just a single shaft vector, no arrowhead
            direction = preview_end - self.start_pos
            parts.append(np.array([[self.start_pos, direction]]))

        if parts:
            return np.concatenate(parts, axis=0)

        ndim = self.viewer.dims.ndim
        return np.empty((0, 2, ndim))

    def _refresh_layer(self, preview_end=None):
        """Update the Vectors layer data."""
        if self.arrows_layer is None:
            return
        self.arrows_layer.data = self._build_all_vectors(preview_end)

    # ---- mouse interaction ----

    def on_mouse_drag(self, viewer, event):
        """Generator-based mouse-drag handler for drawing arrows."""
        if not self._is_active:
            return

        # Right-click: delete last arrow
        if event.type == 'mouse_press' and event.button == 2:
            if self._arrow_endpoints:
                self._arrow_endpoints.pop()
                self._refresh_layer()
                self._save_callback()
                self.viewer.status = "Removed last arrow."
            yield
            return

        # Ctrl+Shift + Left-click to draw (combo is unused by napari)
        if not ('Control' in event.modifiers and 'Shift' in event.modifiers) or event.button != 1:
            return

        # --- press ---
        self.start_pos = self._world_to_data(event.position)
        self._refresh_layer(preview_end=self.start_pos)
        self.viewer.status = "Release mouse to finish arrow."
        yield

        # --- move ---
        while event.type == 'mouse_move':
            cursor = self._world_to_data(event.position)
            self._refresh_layer(preview_end=cursor)
            yield

        # --- release ---
        cursor = self._world_to_data(event.position)
        if np.linalg.norm(cursor - self.start_pos) > 1e-3:
            self._arrow_endpoints.append((self.start_pos.copy(), cursor.copy()))
            self.viewer.status = "Arrow drawn."
        else:
            self.viewer.status = "Arrow too small, cancelled."

        self.start_pos = None
        self._refresh_layer()
        self._save_callback()

    # ---- activation ----

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            self._get_or_create_layer()
            self.viewer.status = "Arrow drawing ON. Ctrl+Shift+drag to draw, right-click to remove last."
            if self.on_mouse_drag not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.insert(0, self.on_mouse_drag)
        else:
            if self.start_pos is not None:
                self.start_pos = None
                self._refresh_layer()
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
    
    captures_dir = Path(output_dir) / constants.CAPTURES_DIR
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
        captures_dir = Path(output_dir) / constants.CAPTURES_DIR
        captures_dir.mkdir(parents=True, exist_ok=True)
            
        save_path = captures_dir / output_filename
        
        if save_path.exists():
            next_name = _get_next_filename()
            popups.show_error_popup(
                viewer.window._qt_window,
                "File Exists",
                f"The file '{output_filename}' already exists. The filename has been updated to '{next_name}'. Please try again."
            )
            _capture_widget.output_filename.value = next_name
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
        _capture_widget.output_filename.value = _get_next_filename()
        
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
def _capture_widget(output_filename: str):
    """Magicgui widget to capture the current viewer canvas."""
    viewer = napari.current_viewer()
    _capture_view(viewer, output_filename)

# --- Hotkey setup ---
def capture_with_hotkey(viewer: napari.Viewer):
    """Wrapper to call capture from a hotkey."""
    # Use the filename currently in the widget's textbox
    filename = _capture_widget.output_filename.value
    _capture_view(viewer, filename)

# --- Region Capture (Ctrl+A) ---

class RegionCaptureOverlay(QWidget):
    """Transparent overlay drawn on top of the canvas that lets the user
    click-and-drag a selection rectangle.  On mouse-release the enclosed
    area is grabbed from the canvas and saved as a PNG capture."""

    def __init__(self, canvas_widget, viewer):
        super().__init__(canvas_widget)
        self._canvas = canvas_widget
        self._viewer = viewer
        self._origin = None
        self._current = None
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.hide()

    # -- public API --

    def activate(self):
        """Show the overlay and begin listening for a box selection."""
        self.resize(self._canvas.size())
        self.move(0, 0)
        self.setCursor(Qt.CrossCursor)
        self.show()
        self.raise_()
        self.setFocus()
        self._viewer.status = "Click & drag to select capture region. Escape to cancel."

    # -- Qt event overrides --

    def paintEvent(self, event):
        if self._origin is not None and self._current is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(255, 255, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 40)))
            rect = QRect(self._origin, self._current).normalized()
            painter.drawRect(rect)
            painter.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._origin = event.pos()
            self._current = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self._origin is not None:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._origin is not None:
            self._current = event.pos()
            rect = QRect(self._origin, self._current).normalized()
            self._deactivate()

            if rect.width() < 5 or rect.height() < 5:
                self._viewer.status = "Selection too small — capture cancelled."
                return

            # Grab just the selected rectangle from the canvas underneath
            pixmap = self._canvas.grab(rect)
            _save_region_capture(self._viewer, pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self._deactivate()
            self._viewer.status = "Region capture cancelled."
        else:
            super().keyPressEvent(event)

    # -- internals --

    def _deactivate(self):
        self._origin = None
        self._current = None
        self.hide()
        self.setCursor(Qt.ArrowCursor)


def _save_region_capture(viewer, pixmap):
    """Save a QPixmap from a region capture to the captures directory."""
    global capture_count

    output_dir = session.get_data("output_dir")
    if not output_dir:
        viewer.status = "No active session — cannot save capture."
        return

    captures_dir = Path(output_dir) / constants.CAPTURES_DIR
    captures_dir.mkdir(parents=True, exist_ok=True)

    filename = _get_next_filename()
    if not filename:
        return

    save_path = captures_dir / filename
    pixmap.save(str(save_path))

    capture_count += 1
    next_fn = _get_next_filename()
    if next_fn:
        _capture_widget.output_filename.value = next_fn

    viewer.status = f"Region captured: {save_path.name}"


_region_overlay = None

def region_capture_with_hotkey(viewer: napari.Viewer):
    """Activate the region-capture overlay from a hotkey (Ctrl+A)."""
    global _region_overlay

    if not session.get_data("output_dir"):
        viewer.status = "No active session — start or load a session first."
        return

    canvas_widget = viewer.window.qt_viewer.canvas.native

    if _region_overlay is None:
        _region_overlay = RegionCaptureOverlay(canvas_widget, viewer)

    _region_overlay.activate()


# --- Widget Setup ---
# Add hotkey information and initialize filename
hotkey_container = widgets.Container(layout="horizontal", labels=False)
hotkey_container.append(widgets.Label(value="Shift+P: Capture  |  Ctrl+A: Region Capture"))
_capture_widget.insert(0, hotkey_container)

initial_filename = _get_next_filename()
_capture_widget.output_filename.value = initial_filename if initial_filename else "capture1.png"

# Add Arrow drawing tool
arrow_container = widgets.Container(layout="horizontal", labels=False)
arrow_chk = widgets.CheckBox(text="Draw Arrows")
arrow_container.append(arrow_chk)
_capture_widget.append(arrow_container)

# Add Scale Bar Options
sb_container = widgets.Container(layout="horizontal", labels=False)
sb_label = widgets.Label(value="<b>Scalebar:</b>")
sb_visible = widgets.CheckBox(text="Visible", value=True)
sb_lock = widgets.CheckBox(text="Lock")
sb_pixels = widgets.CheckBox(text="Show Pixels")

sb_container.extend([sb_label, sb_visible, sb_lock, sb_pixels])
_capture_widget.append(sb_container)

@sb_visible.changed.connect
def _on_sb_visible(state: bool):
    """Toggles the visibility of the custom scale bar."""
    viewer = napari.current_viewer()
    if viewer and hasattr(viewer.window, 'custom_scale_bar'):
        if state:
            viewer.window.custom_scale_bar.show()
        else:
            viewer.window.custom_scale_bar.hide()

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

# --- UI Wrapper ---
capture_widget = widgets.Container(labels=False)
header = widgets.Label(value="Capture & Annotate")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Save screenshots and draw annotations.</i>")
capture_widget.extend([header, info, _capture_widget])