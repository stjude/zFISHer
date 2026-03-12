import napari
import numpy as np
import warnings as _warnings
from pathlib import Path
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF
from qtpy.QtCore import Qt, QRect, QPointF, QEvent

from ...core import session
from .. import popups
from ..decorators import require_active_session
from ... import constants

# --- Arrow Overlay (QPainter-based, world-anchored, screen-rendered) ---

class ArrowOverlay(QWidget):
    """Transparent widget that draws arrows on top of the napari canvas.

    Arrow endpoints are stored in *data* coordinates (matching the image
    layers) and projected to screen space via the vispy camera transform
    on every repaint.  This keeps arrows anchored to their 3-D positions
    while rendering them as clean 2-D graphics that are never occluded.
    """

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self._arrow_endpoints = []   # list of (start, end) np arrays in data coords
        self._preview_start = None   # set while an arrow is being drawn
        self._preview_end = None
        self._synced = False         # True once sync_from_session has run

        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # Repaint whenever the camera or displayed slice changes
        viewer.camera.events.connect(self._on_camera_change)
        viewer.dims.events.current_step.connect(self._on_camera_change)

        if parent:
            parent.installEventFilter(self)
            self.resize(parent.size())
            self.move(0, 0)

        self.show()
        self.raise_()

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == QEvent.Resize:
            self.resize(self.parent().size())
            self.move(0, 0)
        return super().eventFilter(obj, event)

    def _on_camera_change(self, event=None):
        # Skip repaint when hidden or when there are no arrows to draw
        if not self.isVisible():
            return
        if not self._arrow_endpoints and self._preview_start is None:
            return
        self.update()

    # ---- coordinate transforms ----

    def _get_reference_transform(self):
        """Scale and translate from the first image layer, or defaults."""
        ndim = self.viewer.dims.ndim
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return np.array(layer.scale), np.array(layer.translate)
        return np.ones(ndim), np.zeros(ndim)

    def _data_to_world(self, data_point):
        scale, translate = self._get_reference_transform()
        return np.asarray(data_point) * scale + translate

    def _get_scene_transform(self):
        """Return the vispy scene-to-canvas Transform, or *None*."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            qt_viewer = self.viewer.window.qt_viewer

        # Strategy 1 – use any layer's vispy visual node (most reliable)
        layer_to_visual = getattr(qt_viewer, 'layer_to_visual', None)
        if layer_to_visual is not None:
            for layer in self.viewer.layers:
                try:
                    visual = layer_to_visual[layer]
                    return visual.node.get_transform(
                        map_from='scene', map_to='canvas'
                    )
                except (KeyError, AttributeError):
                    continue

        # Strategy 2 – get the ViewBox from the canvas or qt_viewer
        for src in (
            getattr(qt_viewer, 'canvas', None),
            qt_viewer,
        ):
            view = getattr(src, 'view', None) or getattr(src, '_view', None)
            if view is not None and hasattr(view, 'scene'):
                return view.scene.get_transform(map_to='canvas')

        return None

    def _device_pixel_ratio(self):
        """Return the device-pixel-ratio for the canvas widget."""
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", FutureWarning)
                native = self.viewer.window.qt_viewer.canvas.native
            return float(getattr(native, 'devicePixelRatio', lambda: 1.0)())
        except Exception:
            return 1.0

    def _build_projection(self):
        """Pre-compute everything needed to project data→screen coords.

        Returns a callable  ``project(data_point) -> (sx, sy) | (None, None)``
        that is cheap to call per-point, or *None* if projection is impossible.
        """
        scale, translate = self._get_reference_transform()
        displayed = list(self.viewer.dims.displayed)
        rev_displayed = list(reversed(displayed))

        # Try vispy scene transform first
        try:
            tr = self._get_scene_transform()
        except Exception:
            tr = None

        if tr is not None:
            dpr = self._device_pixel_ratio()
            buf = np.zeros(4)
            buf[3] = 1.0
            nd = len(rev_displayed)

            def _project_vispy(data_point):
                world = np.asarray(data_point) * scale + translate
                for i, d in enumerate(rev_displayed):
                    buf[i] = world[d]
                screen = tr.map(buf)
                return float(screen[0]) / dpr, float(screen[1]) / dpr

            return _project_vispy

        # Fallback: manual 2-D
        if self.viewer.dims.ndisplay == 2:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", FutureWarning)
                native = self.viewer.window.qt_viewer.canvas.native
            canvas_w = native.width()
            canvas_h = native.height()
            center = self.viewer.camera.center
            zoom = self.viewer.camera.zoom
            ydim, xdim = displayed

            def _project_manual(data_point):
                world = np.asarray(data_point) * scale + translate
                sx = (world[xdim] - center[xdim]) * zoom + canvas_w / 2
                sy = (world[ydim] - center[ydim]) * zoom + canvas_h / 2
                return sx, sy

            return _project_manual

        return None

    # ---- session sync ----

    def sync_from_session(self):
        """Load saved arrow endpoints from the session's .npy file."""
        if self._synced:
            return
        self._synced = True
        out_dir = session.get_data("output_dir")
        if not out_dir:
            return
        arrows_path = Path(out_dir) / constants.SEGMENTATION_DIR / "Arrows.npy"
        if arrows_path.exists():
            data = np.load(arrows_path)
            if data.ndim == 3 and data.shape[1] == 2 and data.shape[0] > 0:
                self._arrow_endpoints = [
                    (data[i, 0].copy(), data[i, 1].copy())
                    for i in range(len(data))
                ]
                self.update()
        _update_arrow_count()

    # ---- rendering ----

    def paintEvent(self, event):
        arrows = list(self._arrow_endpoints)
        has_preview = (
            self._preview_start is not None and self._preview_end is not None
        )
        if has_preview:
            arrows.append((self._preview_start, self._preview_end))

        if not arrows:
            return

        # Build projection once for the whole frame
        project = self._build_projection()
        if project is None:
            return

        # Pre-compute slice filter values (only needed in 2D)
        is_2d = self.viewer.dims.ndisplay == 2
        if is_2d:
            displayed = set(self.viewer.dims.displayed)
            current_step = self.viewer.dims.current_step
            non_displayed = [d for d in range(self.viewer.dims.ndim) if d not in displayed]

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        committed_pen = QPen(QColor(255, 255, 255), 2)
        committed_brush = QBrush(QColor(255, 255, 255))
        preview_pen = QPen(QColor(255, 255, 255, 128), 2, Qt.DashLine)

        n_committed = len(self._arrow_endpoints)

        for i, (start_data, end_data) in enumerate(arrows):
            is_preview = i >= n_committed

            # Slice filter (2D only, skip for preview)
            if not is_preview and is_2d:
                on_slice = True
                for d in non_displayed:
                    arrow_z = (start_data[d] + end_data[d]) * 0.5
                    if abs(arrow_z - current_step[d]) > 0.5:
                        on_slice = False
                        break
                if not on_slice:
                    continue

            sx, sy = project(start_data)
            ex, ey = project(end_data)

            if is_preview:
                painter.setPen(preview_pen)
                painter.setBrush(Qt.NoBrush)
            else:
                painter.setPen(committed_pen)
                painter.setBrush(committed_brush)

            # Shaft
            painter.drawLine(int(sx), int(sy), int(ex), int(ey))

            # Arrowhead
            dx, dy = ex - sx, ey - sy
            length = (dx * dx + dy * dy) ** 0.5
            if length < 2:
                continue
            ux, uy = dx / length, dy / length
            head_len = min(15, length * 0.3)
            head_w = head_len * 0.5
            px, py = -uy, ux
            bx = ex - head_len * ux
            by = ey - head_len * uy
            triangle = QPolygonF([
                QPointF(ex, ey),
                QPointF(bx + head_w * px, by + head_w * py),
                QPointF(bx - head_w * px, by - head_w * py),
            ])
            painter.drawPolygon(triangle)

        painter.end()


class ArrowDrawer:
    """Keyboard-driven arrow drawing that feeds an ArrowOverlay."""

    def __init__(self, viewer: napari.Viewer, overlay: ArrowOverlay):
        self.viewer = viewer
        self.overlay = overlay
        self.start_pos = None
        self._is_active = False
        self._save_callback = self._create_save_callback()

        # Load any previously saved arrows into the overlay
        self.overlay.sync_from_session()

    @property
    def _arrow_endpoints(self):
        return self.overlay._arrow_endpoints

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
                    layer_type='arrows', metadata={'subtype': 'arrows'}
                )
            _update_arrow_count()
        return _save

    # ---- coordinate helpers ----

    def _get_reference_transform(self):
        ndim = self.viewer.dims.ndim
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return np.array(layer.scale), np.array(layer.translate)
        return np.ones(ndim), np.zeros(ndim)

    def _world_to_data(self, world_pos):
        scale, translate = self._get_reference_transform()
        return (np.array(world_pos) - translate) / scale

    # ---- key interaction ----

    def _on_key_a(self, viewer):
        """Press 'a' to set start point, press 'a' again to set end point."""
        if not self._is_active:
            return
        if viewer.dims.ndisplay != 2:
            viewer.status = "Switch to 2D slice view to draw arrows."
            return
        cursor = self._world_to_data(viewer.cursor.position)
        if self.start_pos is None:
            self.start_pos = cursor.copy()
            self.overlay._preview_start = self.start_pos
            self.overlay._preview_end = self.start_pos.copy()
            self.overlay.update()
            viewer.status = "Arrow start set. Press 'A' at end point, 'Escape' to cancel, Ctrl+Z to undo."
        else:
            if np.linalg.norm(cursor - self.start_pos) > 1e-3:
                self._arrow_endpoints.append((self.start_pos.copy(), cursor.copy()))
                viewer.status = "Arrow drawn."
            else:
                viewer.status = "Arrow too small, cancelled."
            self.start_pos = None
            self.overlay._preview_start = None
            self.overlay._preview_end = None
            self.overlay.update()
            self._save_callback()

    def _on_key_escape(self, viewer):
        if not self._is_active:
            return
        if self.start_pos is not None:
            self.start_pos = None
            self.overlay._preview_start = None
            self.overlay._preview_end = None
            self.overlay.update()
            viewer.status = "Arrow cancelled."

    def _on_key_undo(self, viewer):
        if not self._is_active:
            return
        if self._arrow_endpoints:
            self._arrow_endpoints.pop()
            self.start_pos = None
            self.overlay._preview_start = None
            self.overlay._preview_end = None
            self.overlay.update()
            self._save_callback()
            viewer.status = "Removed last arrow."

    def _on_key_delete(self, viewer):
        """Delete the arrow closest to the cursor."""
        if not self._is_active or not self._arrow_endpoints:
            return
        cursor = self._world_to_data(viewer.cursor.position)
        # Find the arrow whose midpoint is nearest to the cursor
        best_idx = None
        best_dist = float('inf')
        for i, (start, end) in enumerate(self._arrow_endpoints):
            midpoint = (start + end) / 2.0
            dist = np.linalg.norm(cursor - midpoint)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx is not None:
            self._arrow_endpoints.pop(best_idx)
            self.overlay.update()
            self._save_callback()
            viewer.status = "Deleted nearest arrow."

    # ---- activation ----

    def set_active(self, active: bool):
        self._is_active = active
        if active:
            self.overlay.sync_from_session()
            self.viewer.status = "Arrow drawing ON. 'A': draw | 'D': delete nearest | Ctrl+Z: undo | Esc: cancel"
            self.viewer.bind_key('a', self._on_key_a, overwrite=True)
            self.viewer.bind_key('d', self._on_key_delete, overwrite=True)
            self.viewer.bind_key('Escape', self._on_key_escape, overwrite=True)
            self.viewer.bind_key('Control-z', self._on_key_undo, overwrite=True)
        else:
            if self.start_pos is not None:
                self.start_pos = None
                self.overlay._preview_start = None
                self.overlay._preview_end = None
                self.overlay.update()
            self.viewer.status = "Arrow drawing OFF."
            self.viewer.bind_key('a', None, overwrite=True)
            self.viewer.bind_key('d', None, overwrite=True)
            self.viewer.bind_key('Escape', None, overwrite=True)
            self.viewer.bind_key('Control-z', None, overwrite=True)

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
initial_filename = _get_next_filename()
_capture_widget.output_filename.value = initial_filename if initial_filename else "capture1.png"

# Arrow checkboxes and tally
arrow_chk = widgets.CheckBox(text="Draw Arrows")
arrow_show_chk = widgets.CheckBox(text="Show Arrows", value=True)
arrow_count_label = widgets.Label(value="Arrows: 0")
arrow_clear_btn = widgets.PushButton(text="Clear All Arrows")

def _update_arrow_count():
    """Refresh the arrow tally label from the overlay."""
    viewer = napari.current_viewer()
    overlay = getattr(viewer.window, 'arrow_overlay', None) if viewer else None
    n = len(overlay._arrow_endpoints) if overlay else 0
    arrow_count_label.value = f"Arrows: {n}"

@arrow_clear_btn.changed.connect
def _on_clear_arrows():
    global arrow_drawer
    viewer = napari.current_viewer()
    if not viewer:
        return
    overlay = getattr(viewer.window, 'arrow_overlay', None)
    if overlay is None:
        return
    overlay._arrow_endpoints.clear()
    overlay._preview_start = None
    overlay._preview_end = None
    overlay._synced = True  # prevent reload from stale file
    overlay.update()
    if arrow_drawer is not None:
        arrow_drawer.start_pos = None
        arrow_drawer._save_callback()
    else:
        # Save empty arrows to disk even without an active drawer
        out_dir = session.get_data("output_dir")
        if out_dir:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            arrows_path = seg_dir / "Arrows.npy"
            np.save(arrows_path, np.empty((0, 2, viewer.dims.ndim)))
    _update_arrow_count()
    viewer.status = "All arrows cleared."

# Scale Bar Options
sb_visible = widgets.CheckBox(text="Visible", value=True)
sb_lock = widgets.CheckBox(text="Lock")
sb_pixels = widgets.CheckBox(text="Show Pixels")
sb_container = widgets.Container(layout="horizontal", labels=False)
sb_container.extend([sb_visible, sb_lock, sb_pixels])

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

# Arrow drawer is created lazily; the overlay is created in viewer.py at startup
# and stored on viewer.window.arrow_overlay.
arrow_drawer = None

@arrow_chk.changed.connect
def _on_arrow_draw_toggled(state: bool):
    global arrow_drawer
    viewer = napari.current_viewer()
    if not viewer:
        arrow_chk.value = False
        return

    overlay = getattr(viewer.window, 'arrow_overlay', None)
    if overlay is None:
        arrow_chk.value = False
        print("Arrow overlay not initialised — cannot draw arrows.")
        return

    if arrow_drawer is None:
        arrow_drawer = ArrowDrawer(viewer, overlay)

    arrow_drawer.set_active(state)

@arrow_show_chk.changed.connect
def _on_arrow_show_toggled(state: bool):
    viewer = napari.current_viewer()
    if viewer and hasattr(viewer.window, 'arrow_overlay'):
        viewer.window.arrow_overlay.setVisible(state)

# --- UI Wrapper ---
from qtpy.QtWidgets import QFrame
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

capture_widget = widgets.Container(labels=False)
header = widgets.Label(value="Capture & Annotate")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Save screenshots and draw annotations.</i>")
info.native.setObjectName("widgetInfo")
_hotkey_label = widgets.Label(value="<i>Shift+P: Capture  |  Ctrl+A: Region Capture</i>")

_capture_header = widgets.Label(value="<b>Capture:</b>")
_arrow_header = widgets.Label(value="<b>Annotations:</b>")
_sb_header = widgets.Label(value="<b>Scale Bar:</b>")

_layout = capture_widget.native.layout()
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
# --- Capture ---
_layout.addWidget(_capture_header.native)
_layout.addWidget(_hotkey_label.native)
_layout.addWidget(_capture_widget.output_filename.native)
_layout.addWidget(_capture_widget.call_button.native)
# --- Annotations ---
_layout.addWidget(_make_divider())
_layout.addWidget(_arrow_header.native)
_layout.addWidget(arrow_chk.native)
_layout.addWidget(arrow_show_chk.native)
_layout.addWidget(arrow_count_label.native)
_arrow_hint = widgets.Label(value="<i>Press D over arrow to delete</i>")
_arrow_hint.native.setObjectName("widgetInfo")
_layout.addWidget(_arrow_hint.native)
_layout.addWidget(arrow_clear_btn.native)
# --- Scale Bar ---
_layout.addWidget(_make_divider())
_layout.addWidget(_sb_header.native)
_layout.addWidget(sb_container.native)