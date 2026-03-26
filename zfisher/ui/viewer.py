import napari
import math
import warnings
from pathlib import Path
from functools import partial

from qtpy.QtWidgets import QApplication, QToolBox, QToolButton, QPushButton, QWidget, QLabel, QVBoxLayout, QDockWidget
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtCore import Qt, QPoint, QTimer, QEvent, QObject
from magicgui import widgets
from ..core import session

# --- RESTORED IMPORTS ---
# Import all the individual widgets from their own scripts
from .widgets.home_widget import HomeWidget
from .widgets.start_session_widget import StartSessionWidget
from .widgets.nuclei_segmentation_widget import NucleiSegmentationWidget
from .widgets.alignment_consensus_widget import AlignmentConsensusWidget
from .widgets.dapi_segmentation_widget import dapi_segmentation_widget
from .widgets.registration_widget import registration_widget
from .widgets.canvas_widget import canvas_widget
from .widgets.nuclei_matching_widget import nuclei_matching_widget
from .widgets.puncta_picking_widget import PunctaPickingWidget
from .widgets.puncta_widget import puncta_widget
from .widgets.colocalization_widget import colocalization_widget
from .widgets.export_visualization_widget import ExportVisualizationWidget
from .widgets.mask_editor_widget import mask_editor_widget, delete_mask_under_mouse
from .widgets.puncta_editor_widget import puncta_editor_widget, delete_point_under_mouse
from .widgets.capture_widget import capture_widget, capture_with_hotkey, region_capture_with_hotkey, ArrowOverlay

# Import the event handlers
from . import events, style

# Module-level flag to suppress custom layer control callbacks during batch loading
_suppress_custom_controls = False


class _PopupSuppressor(QObject):
    """Global event filter that blocks popup windows during layer operations."""

    def eventFilter(self, obj, event):
        return False


# --- Helper Classes ---

from qtpy.QtCore import Qt, QPoint, QTimer, QEvent

# ... (keep existing imports) ...

class DraggableScaleBar(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.dragging = False
        self.drag_start_position = QPoint()
        self.locked = False
        self.show_pixels = False
        self.pen_color = style.SCALE_BAR_PEN_COLOR
        self.font_color = style.SCALE_BAR_FONT_COLOR
        self.font = style.SCALE_BAR_FONT
        self.resize(200, 60)
        
        # New robust positioning logic
        if self.parent():
            self.parent().installEventFilter(self)
        
        self.viewer.camera.events.zoom.connect(self.on_zoom)
        self.viewer.layers.events.inserted.connect(self.on_layer_change)
        self.viewer.layers.events.removed.connect(self.on_layer_change)

        self.pixel_size_um = 1.0
        self.bar_length_um = 10
        self.bar_length_px = 100
        self.text = ""
        self.recalculate()

    def eventFilter(self, watched, event):
        # When the parent widget is resized, move this widget
        if watched == self.parent() and event.type() == QEvent.Resize:
            self.move_to_bottom_right()
        return super().eventFilter(watched, event)

    def move_to_bottom_right(self):
        self.adjustSize()
        parent = self.parent()
        if parent:
            p_w, p_h = parent.width(), parent.height()
            if p_w > 0 and p_h > 0:
                self.move(p_w - self.width() - 20, p_h - self.height() - 20)

    def get_pixel_size(self):
        # This should ideally get the pixel size from the current layer
        # For now, it seems to be hardcoded in dependent calculations
        return 1.0

    def recalculate(self):
        self.pixel_size_um = self.get_pixel_size()
        self.on_zoom()

    def on_layer_change(self, event=None):
        self.recalculate()

    def on_zoom(self, event=None):
        zoom = self.viewer.camera.zoom
        if zoom == 0: return
        target_px = 150
        
        # Get pixel size from the active layer's scale metadata.
        # Fallback to 1.0 when no layer is selected (scale bar is inaccurate
        # but remains visible until a layer provides real pixel spacing).
        active_layer = self.viewer.layers.selection.active
        if active_layer:
             pixel_size_x = active_layer.scale[-1]
        else:
             pixel_size_x = 1.0

        um_per_canvas_px = pixel_size_x / zoom if pixel_size_x > 0 else 1.0 / zoom

        target_um = target_px * um_per_canvas_px
        if target_um <= 0: return

        exponent = math.floor(math.log10(target_um))
        fraction = target_um / (10 ** exponent)

        if fraction < 1.5: nice_fraction = 1
        elif fraction < 3.5: nice_fraction = 2
        elif fraction < 7.5: nice_fraction = 5
        else: nice_fraction = 10
        
        self.bar_length_um = nice_fraction * (10 ** exponent)
        self.bar_length_px = self.bar_length_um / um_per_canvas_px
        self.text = f"{self.bar_length_um:.4g} um"
        if self.show_pixels:
            self.text += f" ({int(self.bar_length_px)} px)"
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.font_color)
        painter.setFont(self.font)
        rect = self.rect()
        text_rect = painter.boundingRect(rect, Qt.AlignHCenter | Qt.AlignTop, self.text)
        total_h = text_rect.height() + 5 + 6
        start_y = (rect.height() - total_h) / 2
        painter.drawText(rect.left(), int(start_y), rect.width(), text_rect.height(), Qt.AlignHCenter, self.text)
        bar_y = start_y + text_rect.height() + 5
        start_x = (rect.width() - self.bar_length_px) / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.pen_color)
        painter.drawRect(int(start_x), int(bar_y), int(self.bar_length_px), 6)
        
    def mousePressEvent(self, event):
        if self.locked: return
        if event.button() == Qt.RightButton:
            self.dragging = True
            self.drag_start_position = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.mapToParent(event.pos() - self.drag_start_position))
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging = False

class WelcomeWidget(QWidget):
    """A solid black splash screen that manages its own visibility."""
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        
        # Opaque black background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, style.WELCOME_WIDGET_BG_COLOR)
        self.setPalette(palette)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignCenter)
        
        # Icon
        icon_path = Path(__file__).parent.parent.parent / "icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path)).scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
            self.layout().addWidget(icon_label)

        # Branding (Mint and White)
        label_html = (
            f"<h1 {style.WELCOME_WIDGET_STYLE['h1']}>zFISHer</h1>"
            f"<p {style.WELCOME_WIDGET_STYLE['p']}>Version 1.0</p>"
        )
        self.label = QLabel(label_html)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.label)

        # Connect internal visibility logic
        self.viewer.layers.events.inserted.connect(self._check_visibility)
        self.viewer.layers.events.removed.connect(self._check_visibility)

        if self.parent():
            self.parent().installEventFilter(self)
            QTimer.singleShot(100, lambda: (self.resize_to_parent(), self._check_visibility()))

    def _check_visibility(self, event=None):
        if len(self.viewer.layers) > 0:
            self.hide()
            self.setEnabled(False) 
        else:
            self.show()
            self.raise_() 
            self.setEnabled(True)

    def paintEvent(self, event):
        if self.isVisible():
            painter = QPainter(self)
            painter.fillRect(self.rect(), style.WELCOME_WIDGET_BG_COLOR)
            
    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == event.Resize:
            self.resize_to_parent()
        return super().eventFilter(obj, event)

    def resize_to_parent(self):
        if not self.parent(): return
        self.resize(self.parent().size())
        self.move(0, 0)

# --- Creation Logic ---


def _patch_vispy_arcball():
    """Fix vispy arcball bug: _arcball receives 3D coords but expects 2D."""
    try:
        import vispy.scene.cameras.arcball as _ab
        _orig = _ab._arcball
        def _safe_arcball(xy, wh):
            return _orig(xy[:2], wh)
        _ab._arcball = _safe_arcball
    except Exception:
        pass

def launch_zfisher():
    _patch_vispy_arcball()

    # FIX: Ensure QApplication is initialized properly
    app = QApplication.instance() or QApplication([])

    # Install global event filter to suppress napari Qt.Popup windows
    # (color swatches, contrast limit popups) during layer operations
    _popup_suppressor = _PopupSuppressor(app)
    app.installEventFilter(_popup_suppressor)

    # --- 1. Amethyst & Mint Theme (RE-ENABLED) ---
    theme_name = style.register_napari_theme()

    # Build icon — generate .ico from .png for Windows taskbar compatibility
    icon_dir = Path(__file__).parent.parent.parent
    ico_path = icon_dir / "icon.ico"
    png_path = icon_dir / "icon.png"
    if not ico_path.exists() and png_path.exists():
        try:
            from PIL import Image
            img = Image.open(png_path).convert('RGBA')
            ico_sizes = [(s, s) for s in [16, 32, 48, 64, 128, 256]]
            img.save(ico_path, format='ICO', sizes=ico_sizes)
        except Exception:
            ico_path = png_path
    # Prefer the full-res PNG for Qt (handles high-DPI natively); .ico for Win32 API
    icon_file = ico_path if ico_path.exists() else png_path
    icon = QIcon()
    if png_path.exists():
        icon.addFile(str(png_path))
    elif icon_file.exists():
        icon.addFile(str(icon_file))
    app.setWindowIcon(icon)

    # --- 2. Create Viewer ---
    viewer = napari.Viewer(title="zFISHer - 3D Colocalization of Sequential Multiplexed FISH in Cell Monolayer", ndisplay=2)

    # Permanently disable napari's notification toast system —
    # zFISHer uses its own status bar and popup system instead.
    try:
        from napari.utils.notifications import notification_manager
        notification_manager.enabled = False
    except Exception:
        pass

    # Apply icon — deferred so it runs after napari finishes its own window setup
    def _apply_icon():
        if icon.isNull():
            return
        qt_win = viewer.window._qt_window
        qt_win.setWindowIcon(icon)
        # Force icon onto the Win32 window handle (bypasses Qt for taskbar/title bar)
        try:
            import ctypes
            ICON_SMALL, ICON_BIG, WM_SETICON = 0, 1, 0x80
            LR_LOADFROMFILE = 0x10
            # Query actual system icon sizes (DPI-aware) instead of hardcoding
            SM_CXICON, SM_CYICON = 11, 12      # big icon
            SM_CXSMICON, SM_CYSMICON = 49, 50  # small icon
            big_w = ctypes.windll.user32.GetSystemMetrics(SM_CXICON) or 48
            big_h = ctypes.windll.user32.GetSystemMetrics(SM_CYICON) or 48
            small_w = ctypes.windll.user32.GetSystemMetrics(SM_CXSMICON) or 16
            small_h = ctypes.windll.user32.GetSystemMetrics(SM_CYSMICON) or 16
            hwnd = int(qt_win.winId())
            ico_str = str(ico_path if ico_path.exists() else icon_file)
            hicon_big = ctypes.windll.user32.LoadImageW(
                None, ico_str, 1, big_w, big_h, LR_LOADFROMFILE
            )
            hicon_small = ctypes.windll.user32.LoadImageW(
                None, ico_str, 1, small_w, small_h, LR_LOADFROMFILE
            )
            if hicon_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
            if hicon_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
        except Exception:
            pass

    # Apply icon immediately on the new window to prevent napari icon flash
    if not icon.isNull():
        viewer.window._qt_window.setWindowIcon(icon)
    # Also apply deferred as a safety net (napari may re-set during init)
    from qtpy.QtCore import QTimer
    QTimer.singleShot(100, _apply_icon)
    
    # Permanently disable napari's native welcome screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = viewer.window.qt_viewer
    qt_viewer._show_welcome_screen = False
    if hasattr(qt_viewer, '_welcome_widget'):
        qt_viewer._welcome_widget.set_welcome_visible(False)
    
    try:
        # Apply the registered theme
        viewer.theme = theme_name
    except Exception as e:
        print(f"Could not apply theme: {e}")

    viewer.scale_bar.visible = False



    # Hide the menu bar and remove all Alt+key shortcuts so it can't be reopened
    menu_bar = viewer.window._qt_window.menuBar()
    menu_bar.setVisible(False)
    for action in menu_bar.actions():
        action.setShortcut("")

    # Remove "console" from the right-click dock-widget context menu.
    # Qt's QMainWindow auto-generates the context menu from each dock
    # widget's toggleViewAction(), so we must hide that action directly.
    try:
        console_dock = viewer.window._qt_viewer.dockConsole
        console_dock.toggleViewAction().setVisible(False)
    except Exception:
        pass
    
    # Prevent layers from entering transform mode (which lets users
    # accidentally drag/rotate/scale layers and corrupt spatial alignment).
    def _block_transform_mode(event=None):
        for layer in viewer.layers:
            if hasattr(layer, 'mode') and layer.mode == 'transform':
                layer.mode = 'pan_zoom'

    def _on_layer_added(event):
        layer = event.value
        if hasattr(layer, 'events') and hasattr(layer.events, 'mode'):
            layer.events.mode.connect(_block_transform_mode)
        # Force volume depiction for image layers
        if hasattr(layer, 'depiction'):
            layer.depiction = 'volume'

    viewer.layers.events.inserted.connect(_on_layer_added)
    # Connect existing layers
    for layer in viewer.layers:
        if hasattr(layer, 'events') and hasattr(layer.events, 'mode'):
            layer.events.mode.connect(_block_transform_mode)

    # Hide unwanted controls from napari's layer controls panel.
    def _strip_to_opacity(page):
        """Hide all descendants of *page* except the opacity row and
        the move-camera mode button."""
        from qtpy.QtWidgets import QWidget, QLabel, QSlider, QRadioButton
        keep = set()

        # Find opacity label
        opacity_label = None
        for lbl in page.findChildren(QLabel):
            if lbl.text().strip().rstrip(":").lower() == "opacity":
                opacity_label = lbl
                keep.add(lbl)
                break

        # Find opacity slider via named attribute or as sibling
        for attr in ("opacitySlider", "opacity_slider"):
            s = getattr(page, attr, None)
            if s is not None:
                keep.add(s)
                break
        else:
            if opacity_label and opacity_label.parent():
                for s in opacity_label.parent().findChildren(QSlider):
                    keep.add(s)
                    break

        # Keep the move-camera mode button (napari mode button with
        # "move" in its tooltip, e.g. "Move point(s)")
        from qtpy.QtWidgets import QAbstractButton
        for btn in page.findChildren(QAbstractButton):
            tip = (btn.toolTip() or '').lower()
            if 'move' in tip:
                keep.add(btn)

        # Keep all ancestor containers of the kept widgets
        for w in list(keep):
            p = w.parent()
            while p is not None and p is not page:
                keep.add(p)
                p = p.parent()

        # Hide everything except the kept widgets + ancestors
        for child in page.findChildren(QWidget):
            if child not in keep:
                child.setVisible(False)
                child.setMaximumHeight(0)

        # Remove empty rows from QFormLayout to eliminate spacing gaps
        from qtpy.QtWidgets import QFormLayout
        layout = page.layout()
        if isinstance(layout, QFormLayout):
            for row in range(layout.rowCount() - 1, -1, -1):
                label_item = layout.itemAt(row, QFormLayout.LabelRole)
                field_item = layout.itemAt(row, QFormLayout.FieldRole)
                span_item = layout.itemAt(row, QFormLayout.SpanningRole)
                label_w = label_item.widget() if label_item else None
                field_w = field_item.widget() if field_item else None
                span_w = span_item.widget() if span_item else None
                # If all widgets in the row are hidden, remove the row
                all_hidden = True
                for w in (label_w, field_w, span_w):
                    if w is not None and w in keep:
                        all_hidden = False
                        break
                if all_hidden and (label_w or field_w or span_w):
                    layout.removeRow(row)

    def _hide_unwanted_controls(event=None):
        try:
            if _suppress_custom_controls:
                return
            from qtpy.QtWidgets import QWidget, QComboBox, QLabel
            controls = viewer.window._qt_viewer.controls

            selected = viewer.layers.selection.active
            is_ids_layer = selected is not None and selected.name.endswith("_IDs")
            is_centroids_layer = selected is not None and selected.name.endswith("_centroids")

            page = controls.currentWidget()
            if not page:
                return

            is_mask_layer = selected is not None and selected.name.endswith("_masks")

            # Hide transform/translate button (all layers)
            from qtpy.QtWidgets import QAbstractButton
            for btn in page.findChildren(QAbstractButton):
                tip = (btn.toolTip() or '').lower()
                if any(kw in tip for kw in ('transform', 'translate', 'move layer')):
                    btn.setVisible(False)

            # Hide unwanted controls on mask layers
            if is_mask_layer:
                from qtpy.QtWidgets import QRadioButton, QFormLayout, QSizePolicy as _QSP
                # Hide fill and polygon mode buttons but retain their space
                for btn in page.findChildren(QRadioButton):
                    tip = (btn.toolTip() or '').lower()
                    if 'fill' in tip or 'polygon' in tip:
                        sp = btn.sizePolicy()
                        sp.setRetainSizeWhenHidden(True)
                        btn.setSizePolicy(sp)
                        btn.setVisible(False)

                # Hide form rows: contiguous, preserve labels, color mode,
                # rendering, and display-selected-label
                # Lock n_edit_dimensions to 3 so paint/erase works across z
                selected.n_edit_dimensions = 3

                _hide_labels = {'contiguous', 'preserve', 'color mode', 'rendering', 'display', 'n edit', 'contour', 'gradient', 'label'}
                layout = page.layout()
                if hasattr(layout, 'rowCount'):
                    for row in range(layout.rowCount()):
                        try:
                            lbl_item = layout.itemAt(row, QFormLayout.LabelRole)
                        except Exception:
                            continue
                        if not lbl_item:
                            continue
                        lbl_w = lbl_item.widget()
                        if not lbl_w:
                            continue
                        txt = lbl_w.text().strip().rstrip(':').lower()
                        if any(k in txt for k in _hide_labels):
                            lbl_w.setVisible(False)
                            lbl_w.setMaximumHeight(0)
                            try:
                                field_item = layout.itemAt(row, QFormLayout.FieldRole)
                                if field_item and field_item.widget():
                                    field_item.widget().setVisible(False)
                                    field_item.widget().setMaximumHeight(0)
                            except Exception:
                                pass

            # Hide unwanted controls on puncta layers
            is_puncta_layer = selected is not None and selected.name.endswith("_puncta")
            if is_puncta_layer:
                from qtpy.QtWidgets import QFormLayout
                _hide_puncta = {'blending', 'projection', 'symbol', 'border color'}
                layout = page.layout()
                if isinstance(layout, QFormLayout):
                    for row in range(layout.rowCount()):
                        lbl_item = layout.itemAt(row, QFormLayout.LabelRole)
                        if not lbl_item:
                            continue
                        lbl_w = lbl_item.widget()
                        if not lbl_w:
                            continue
                        txt = lbl_w.text().strip().rstrip(':').lower()
                        if any(k in txt for k in _hide_puncta):
                            lbl_w.setVisible(False)
                            lbl_w.setMaximumHeight(0)
                            field_item = layout.itemAt(row, QFormLayout.FieldRole)
                            if field_item and field_item.widget():
                                field_item.widget().setVisible(False)
                                field_item.widget().setMaximumHeight(0)

            # Hide depiction dropdown (volume/plane) for image layers
            for combo in page.findChildren(QComboBox):
                items = [combo.itemText(i).lower() for i in range(combo.count())]
                if "volume" in items and "plane" in items:
                    combo.setVisible(False)

            if is_ids_layer or is_centroids_layer:
                _strip_to_opacity(page)
        except Exception:
            pass

    # Custom controls for _IDs layers: text size slider + color picker
    _ids_custom_widgets = []  # track so we can remove on deselection

    _NAPARI_SWATCH_SS = (
        "background-color: {}; border: 1px solid #555; border-radius: 3px;"
        " min-height: 20px;"
    )

    def _add_ids_custom_controls(event=None):
        if _suppress_custom_controls:
            return
        from qtpy.QtWidgets import (
            QWidget, QLabel, QSlider, QHBoxLayout, QColorDialog, QPushButton,
            QFormLayout, QVBoxLayout,
        )
        from qtpy.QtCore import Qt
        controls = viewer.window._qt_viewer.controls

        # Remove previous custom widgets
        for w in _ids_custom_widgets:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        _ids_custom_widgets.clear()

        selected = viewer.layers.selection.active
        if selected is None or not selected.name.endswith("_IDs"):
            return

        layer = selected
        current_page = controls.currentWidget()
        if not current_page:
            return
        page_layout = current_page.layout()
        if not page_layout:
            # _strip_to_opacity may have removed all rows leaving no layout.
            # Create a fresh VBoxLayout on the page.
            page_layout = QVBoxLayout(current_page)
            page_layout.setContentsMargins(4, 4, 4, 4)

        # napari uses QtWrappedLabel (right-aligned, word-wrapped QLabel)
        # for its form row labels. Replicate that for our custom rows.
        def _make_label(text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        # --- Text Size ---
        size_label = _make_label("text size:")

        size_slider = QSlider(Qt.Horizontal)
        size_slider.setFocusPolicy(Qt.NoFocus)
        size_slider.setMinimum(4)
        size_slider.setMaximum(40)
        size_slider.setValue(int(layer.text.size))

        size_value = QLabel(str(int(layer.text.size)))
        size_value.setFixedWidth(26)
        size_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def _on_size_changed(val):
            layer.text.size = val
            size_value.setText(str(val))
            layer.refresh()

        size_slider.valueChanged.connect(_on_size_changed)

        size_field = QWidget()
        size_field.setAttribute(Qt.WA_StyledBackground, False)
        size_field.setStyleSheet("background: transparent;")
        size_field_layout = QHBoxLayout(size_field)
        size_field_layout.setContentsMargins(0, 0, 0, 0)
        size_field_layout.setSpacing(4)
        size_field_layout.addWidget(size_slider)
        size_field_layout.addWidget(size_value)

        # --- Text Color ---
        color_label = _make_label("text color:")

        color_btn = QPushButton()
        color_btn.setFixedHeight(22)
        color_btn.setCursor(Qt.PointingHandCursor)

        try:
            tc = layer.text.color
            if hasattr(tc, 'constant') and tc.constant is not None:
                rgba = tc.constant
            else:
                rgba = [0.25, 0.71, 0.85, 1.0]
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
        except Exception:
            hex_color = '#40b5d8'

        color_btn.setStyleSheet(_NAPARI_SWATCH_SS.format(hex_color))

        def _on_color_clicked():
            from qtpy.QtGui import QColor as _QColor
            current = _QColor(hex_color)
            chosen = QColorDialog.getColor(current, controls, "Choose Text Color")
            if chosen.isValid():
                new_hex = chosen.name()
                color_btn.setStyleSheet(_NAPARI_SWATCH_SS.format(new_hex))
                layer.text.color = new_hex
                layer.refresh()

        color_btn.clicked.connect(_on_color_clicked)

        # Add as proper QFormLayout rows — same as napari does internally
        page_layout.addRow(size_label, size_field)
        page_layout.addRow(color_label, color_btn)
        # Force visibility — _strip_to_opacity may have hidden ancestors
        for w in (size_label, size_field, color_label, color_btn):
            w.setVisible(True)
            w.setMaximumHeight(16777215)  # Reset max height (Qt default)
        _ids_custom_widgets.extend([size_label, size_field, color_label, color_btn])

    # Custom controls for _centroids layers: point size slider + face color picker
    _centroids_custom_widgets = []

    def _add_centroids_custom_controls(event=None):
        if _suppress_custom_controls:
            return
        from qtpy.QtWidgets import (
            QWidget, QLabel, QSlider, QHBoxLayout, QColorDialog, QPushButton,
        )
        from qtpy.QtCore import Qt
        import numpy as np
        controls = viewer.window._qt_viewer.controls

        # Remove previous custom widgets
        for w in _centroids_custom_widgets:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        _centroids_custom_widgets.clear()

        selected = viewer.layers.selection.active
        if selected is None or not selected.name.endswith("_centroids"):
            return

        layer = selected
        current_page = controls.currentWidget()
        if not current_page:
            return
        page_layout = current_page.layout()
        if not page_layout:
            return

        def _make_label(text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        # --- Point Size ---
        size_label = _make_label("point size:")

        current_size = int(np.mean(layer.size)) if len(layer.size) > 0 else 5
        size_slider = QSlider(Qt.Horizontal)
        size_slider.setFocusPolicy(Qt.NoFocus)
        size_slider.setMinimum(1)
        size_slider.setMaximum(50)
        size_slider.setValue(current_size)

        size_value = QLabel(str(current_size))
        size_value.setFixedWidth(26)
        size_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        def _on_size_changed(val):
            layer.size = val
            size_value.setText(str(val))

        size_slider.valueChanged.connect(_on_size_changed)

        size_field = QWidget()
        size_field.setAttribute(Qt.WA_StyledBackground, False)
        size_field.setStyleSheet("background: transparent;")
        size_field_layout = QHBoxLayout(size_field)
        size_field_layout.setContentsMargins(0, 0, 0, 0)
        size_field_layout.setSpacing(4)
        size_field_layout.addWidget(size_slider)
        size_field_layout.addWidget(size_value)

        # --- Face Color ---
        color_label = _make_label("face color:")

        color_btn = QPushButton()
        color_btn.setFixedHeight(22)
        color_btn.setCursor(Qt.PointingHandCursor)

        try:
            fc = layer.face_color
            if len(fc) > 0:
                rgba = fc[0]
            else:
                rgba = [1.0, 0.65, 0.0, 1.0]  # orange default
            hex_color = '#{:02x}{:02x}{:02x}'.format(
                int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
            )
        except Exception:
            hex_color = '#ffa500'

        color_btn.setStyleSheet(_NAPARI_SWATCH_SS.format(hex_color))

        def _on_color_clicked():
            from qtpy.QtGui import QColor as _QColor
            current = _QColor(hex_color)
            chosen = QColorDialog.getColor(current, controls, "Choose Face Color")
            if chosen.isValid():
                new_hex = chosen.name()
                color_btn.setStyleSheet(_NAPARI_SWATCH_SS.format(new_hex))
                layer.face_color = new_hex

        color_btn.clicked.connect(_on_color_clicked)

        # Add as proper QFormLayout rows
        page_layout.addRow(size_label, size_field)
        page_layout.addRow(color_label, color_btn)
        for w in (size_label, size_field, color_label, color_btn):
            w.setVisible(True)
            w.setMaximumHeight(16777215)
        _centroids_custom_widgets.extend([size_label, size_field, color_label, color_btn])

    # Custom controls for _puncta layers: point size, symbol color, text size, text color
    _puncta_custom_widgets = []

    def _add_puncta_custom_controls(event=None):
        if _suppress_custom_controls:
            return
        from qtpy.QtWidgets import (
            QWidget, QLabel, QSlider, QHBoxLayout, QColorDialog, QPushButton,
        )
        from qtpy.QtCore import Qt
        import numpy as np
        controls = viewer.window._qt_viewer.controls

        # Remove previous custom widgets
        for w in _puncta_custom_widgets:
            w.hide()
            w.setParent(None)
            w.deleteLater()
        _puncta_custom_widgets.clear()

        selected = viewer.layers.selection.active
        if selected is None or not selected.name.endswith("_puncta"):
            return

        layer = selected
        current_page = controls.currentWidget()
        if not current_page:
            return
        page_layout = current_page.layout()
        if not page_layout:
            return

        def _make_label(text):
            lbl = QLabel(text)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            return lbl

        def _make_slider_row(label_text, min_val, max_val, current_val, on_change):
            lbl = _make_label(label_text)
            slider = QSlider(Qt.Horizontal)
            slider.setFocusPolicy(Qt.NoFocus)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(current_val)
            val_lbl = QLabel(str(current_val))
            val_lbl.setFixedWidth(26)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            def _changed(v):
                on_change(v)
                val_lbl.setText(str(v))
            slider.valueChanged.connect(_changed)
            field = QWidget()
            field.setAttribute(Qt.WA_StyledBackground, False)
            field.setStyleSheet("background: transparent;")
            fl = QHBoxLayout(field)
            fl.setContentsMargins(0, 0, 0, 0)
            fl.setSpacing(4)
            fl.addWidget(slider)
            fl.addWidget(val_lbl)
            return lbl, field

        def _make_color_btn(get_color, set_color, dialog_title):
            btn = QPushButton()
            btn.setFixedHeight(22)
            btn.setCursor(Qt.PointingHandCursor)
            hex_c = get_color()
            btn.setStyleSheet(_NAPARI_SWATCH_SS.format(hex_c))
            def _clicked():
                from qtpy.QtGui import QColor as _QColor
                chosen = QColorDialog.getColor(_QColor(hex_c), controls, dialog_title)
                if chosen.isValid():
                    new_hex = chosen.name()
                    btn.setStyleSheet(_NAPARI_SWATCH_SS.format(new_hex))
                    set_color(new_hex)
            btn.clicked.connect(_clicked)
            return btn

        # --- Point Size ---
        current_pt_size = int(np.mean(layer.size)) if len(layer.size) > 0 else 3
        pt_lbl, pt_field = _make_slider_row(
            "point size:", 1, 30, current_pt_size,
            lambda v: setattr(layer, 'size', v)
        )

        # --- Symbol Color (face color) ---
        sym_color_lbl = _make_label("symbol color:")
        def _get_face_color():
            try:
                fc = layer.face_color
                if len(fc) > 0:
                    rgba = fc[0]
                    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            except Exception:
                pass
            return '#ffff00'
        sym_color_btn = _make_color_btn(
            _get_face_color,
            lambda c: setattr(layer, 'face_color', c),
            "Choose Symbol Color"
        )

        # --- Text Size ---
        current_text_size = int(layer.text.size) if hasattr(layer.text, 'size') else 8
        txt_size_lbl, txt_size_field = _make_slider_row(
            "text size:", 4, 40, current_text_size,
            lambda v: (setattr(layer.text, 'size', v), layer.refresh())
        )

        # --- Text Color ---
        txt_color_lbl = _make_label("text color:")
        def _get_text_color():
            try:
                tc = layer.text.color
                if hasattr(tc, 'constant') and tc.constant is not None:
                    rgba = tc.constant
                else:
                    rgba = [1.0, 1.0, 1.0, 1.0]
                return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
            except Exception:
                return '#ffffff'
        def _set_text_color(c):
            layer.text.color = c
            layer.refresh()
        txt_color_btn = _make_color_btn(_get_text_color, _set_text_color, "Choose Text Color")

        # --- Draw on Top checkbox ---
        from qtpy.QtWidgets import QCheckBox
        top_chk = QCheckBox()
        is_on_top = (str(layer.blending).lower() == 'translucent_no_depth')
        top_chk.setChecked(is_on_top)

        def _on_top_toggled(checked):
            if checked:
                layer.blending = 'translucent_no_depth'
                try:
                    layer.text.blending = 'translucent_no_depth'
                except Exception:
                    pass
            else:
                layer.blending = 'translucent'
                try:
                    layer.text.blending = 'translucent'
                except Exception:
                    pass
            layer.refresh()

        top_chk.stateChanged.connect(lambda state: _on_top_toggled(bool(state)))
        top_label = _make_label("draw on top:")

        # Add rows
        page_layout.addRow(pt_lbl, pt_field)
        page_layout.addRow(sym_color_lbl, sym_color_btn)
        page_layout.addRow(txt_size_lbl, txt_size_field)
        page_layout.addRow(txt_color_lbl, txt_color_btn)
        page_layout.addRow(top_label, top_chk)
        all_custom = [
            pt_lbl, pt_field, sym_color_lbl, sym_color_btn,
            txt_size_lbl, txt_size_field, txt_color_lbl, txt_color_btn,
            top_label, top_chk,
        ]
        for w in all_custom:
            w.setVisible(True)
            w.setMaximumHeight(16777215)
        _puncta_custom_widgets.extend(all_custom)

    viewer.layers.selection.events.changed.connect(_hide_unwanted_controls)
    viewer.layers.selection.events.changed.connect(_add_ids_custom_controls)
    viewer.layers.selection.events.changed.connect(_add_centroids_custom_controls)
    viewer.layers.selection.events.changed.connect(_add_puncta_custom_controls)

    # Suppress custom controls during layer removal and close any
    # stray popup windows that napari creates during control rebuilds.
    _known_windows = set()

    def _snapshot_windows():
        _known_windows.clear()
        for w in QApplication.topLevelWidgets():
            _known_windows.add(id(w))

    def _suppress_on_removing(event=None):
        global _suppress_custom_controls
        _suppress_custom_controls = True
        try:
            viewer.window._qt_viewer.controls.setVisible(False)
        except Exception:
            pass

    def _unsuppress_on_removed(event=None):
        global _suppress_custom_controls
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, _unsuppress_after_remove)

    def _unsuppress_after_remove():
        global _suppress_custom_controls
        _suppress_custom_controls = False
        try:
            viewer.window._qt_viewer.controls.setVisible(True)
        except Exception:
            pass

    viewer.layers.events.removing.connect(_suppress_on_removing)
    viewer.layers.events.removed.connect(_unsuppress_on_removed)

    # Disable renaming layers (double-click) and right-click context menu
    from qtpy.QtWidgets import QAbstractItemView
    from napari._qt.containers.qt_layer_list import QtLayerList
    for lv in viewer.window._qt_window.findChildren(QtLayerList):
        lv.setEditTriggers(QAbstractItemView.NoEditTriggers)
        delegate = lv.itemDelegate()
        if hasattr(delegate, 'show_context_menu'):
            delegate.show_context_menu = lambda *a, **kw: None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        viewer_canvas_native = viewer.window.qt_viewer.canvas.native

    # Setup Welcome Overlay
    welcome_widget = WelcomeWidget(viewer, parent=viewer_canvas_native)
    viewer.window.custom_welcome_widget = welcome_widget

    scale_bar_widget = DraggableScaleBar(viewer, parent=viewer_canvas_native)
    viewer.window.custom_scale_bar = scale_bar_widget

    arrow_overlay = ArrowOverlay(viewer, parent=viewer_canvas_native)
    viewer.window.arrow_overlay = arrow_overlay
    
    nuclei_segmentation_widget = NucleiSegmentationWidget(viewer)
    alignment_consensus_widget = AlignmentConsensusWidget(viewer)
    puncta_picking_widget = PunctaPickingWidget(viewer)
    export_visualization_widget = ExportVisualizationWidget(viewer)

    widget_map = {
        # Child widgets accessed via parent composite widgets
        "dapi_segmentation": nuclei_segmentation_widget.dapi_widget,
        "mask_editor": nuclei_segmentation_widget.mask_editor_widget,
        "registration": alignment_consensus_widget.registration_widget,
        "canvas": alignment_consensus_widget.canvas_widget,
        "nuclei_matching": alignment_consensus_widget.nuclei_matching_widget,
        "automated_preprocessing": alignment_consensus_widget.automated_widget,
        # Other top-level widgets
        "puncta_detection": puncta_picking_widget.algorithmic_widget,
        "puncta_editor": puncta_picking_widget.manual_widget,
        "colocalization": export_visualization_widget.export_widget,
        "capture": export_visualization_widget.capture_widget,
        "start_session": StartSessionWidget(viewer),
        "nuclei_segmentation": nuclei_segmentation_widget,
        "alignment_consensus": alignment_consensus_widget,
        "puncta_picking": puncta_picking_widget,
        "export_visualization": export_visualization_widget,
    }

    widgets_to_add = [
        (HomeWidget(viewer), "zFISHer Home"),
        (StartSessionWidget(viewer), "1. Session && I/O"),
        (nuclei_segmentation_widget, "2. Nuclei Segmentation"),
        (puncta_picking_widget, "3. Puncta Picking"),
        (alignment_consensus_widget, "4. Alignment && Consensus"),
        (export_visualization_widget, "5. Export && Visualization")
    ]

    # --- 3. Sidebar Toolbox Styling ---
    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    toolbox.setStyleSheet(style.TOOLBOX_STYLESHEET)
    
    for widget, name in widgets_to_add:
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        toolbox.addItem(widget.native, name)

    # Make all widgetInfo labels expand to full width so word-wrap uses all space
    from qtpy.QtWidgets import QSizePolicy
    for lbl in toolbox.findChildren(QLabel):
        if lbl.objectName() == "widgetInfo":
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

    # Reset to pan_zoom (move camera) when switching toolbox tabs
    def _reset_to_pan_zoom(_index=None):
        active = viewer.layers.selection.active
        if active is not None and hasattr(active, 'mode') and active.mode != 'pan_zoom':
            active.mode = 'pan_zoom'
        # Disable hover edit mode when switching widgets
        try:
            from .widgets.mask_editor_widget import deactivate_hover_edit
            deactivate_hover_edit()
        except Exception:
            pass
        # Disable fishing hook when switching widgets
        try:
            from .widgets.puncta_editor_widget import _puncta_editor_widget
            _puncta_editor_widget.fishing_hook.value = False
        except Exception:
            pass

    toolbox.currentChanged.connect(_reset_to_pan_zoom)
    # Also connect nested toolboxes (inside StartSessionWidget, NucleiSegmentationWidget, etc.)
    for child_toolbox in toolbox.findChildren(QToolBox):
        child_toolbox.currentChanged.connect(_reset_to_pan_zoom)

    # Connect puncta editor layer sync (layer list ↔ dropdown)
    from .widgets.puncta_editor_widget import connect_puncta_editor_layer_sync
    connect_puncta_editor_layer_sync(viewer)

    # Auto-open Mask Editor when paint/erase mode is activated on a _masks layer
    _mask_mode_connections = {}

    def _on_mask_mode_changed(event, layer=None):
        if layer is None:
            return
        mode = str(event.mode) if hasattr(event, 'mode') else str(event.value)
        if mode in ('paint', 'erase'):
            # Switch main toolbox to "2. Nuclei Segmentation" (index 2)
            if toolbox.currentIndex() != 2:
                toolbox.setCurrentIndex(2)
            # Switch nested toolbox to "Mask Editor" (index 1)
            nested_tb = nuclei_segmentation_widget.toolbox
            if nested_tb.currentIndex() != 1:
                nested_tb.setCurrentIndex(1)

    def _connect_mask_mode_listeners(event=None):
        """Connect mode listeners to any _masks layers, disconnect removed ones."""
        current_layers = set()
        for layer in viewer.layers:
            if layer.name.endswith("_masks"):
                current_layers.add(id(layer))
                if id(layer) not in _mask_mode_connections:
                    cb = lambda evt, l=layer: _on_mask_mode_changed(evt, layer=l)
                    layer.events.mode.connect(cb)
                    _mask_mode_connections[id(layer)] = (layer, cb)
        # Clean up connections for removed layers
        for lid in list(_mask_mode_connections):
            if lid not in current_layers:
                del _mask_mode_connections[lid]

    viewer.layers.events.inserted.connect(_connect_mask_mode_listeners)
    viewer.layers.events.removed.connect(_connect_mask_mode_listeners)
    _connect_mask_mode_listeners()  # connect to any already-existing layers

    dock_widget = viewer.window.add_dock_widget(toolbox, area="right", name="zFISHer Workflow")

    def _hide_title_bar_buttons(dock):
        """Hide float/hide/close buttons in a dock's napari custom title bar."""
        title_bar = dock.titleBarWidget()
        if title_bar is not None:
            for btn in title_bar.findChildren(QPushButton):
                obj_name = btn.objectName()
                if obj_name in ('QTitleBarHideButton', 'QTitleBarFloatButton', 'QTitleBarCloseButton'):
                    btn.hide()

    def _patch_dock_visibility(dock):
        """Monkey-patch _on_visibility_changed so title bar buttons stay hidden
        even after napari recreates the title bar."""
        if getattr(dock, '_lock_patched', False):
            return
        orig = getattr(dock, '_on_visibility_changed', None)
        if orig is not None:
            def _patched(visible, _orig=orig, _dock=dock):
                _orig(visible)
                _hide_title_bar_buttons(_dock)
            dock._on_visibility_changed = _patched
            # Reconnect the signal to the patched version
            try:
                dock.visibilityChanged.disconnect(orig)
            except (TypeError, RuntimeError):
                pass
            dock.visibilityChanged.connect(_patched)
        dock._lock_patched = True

    def lock_ui():
        # Lock every dock widget: disable float/close but keep titles visible
        qt_window = viewer.window._qt_window
        for child in qt_window.findChildren(QDockWidget):
            child.setFeatures(QDockWidget.NoDockWidgetFeatures)
            _hide_title_bar_buttons(child)
            _patch_dock_visibility(child)
            # In QtViewerButtons: keep only home and 2D/3D, centered with spacing
            # In QtLayerButtons: keep only trash can
            for w in child.findChildren(QWidget):
                class_name = w.__class__.__name__
                if class_name == 'QtViewerButtons':
                    layout = w.layout()
                    if layout:
                        home_btn = nd_btn = None
                        for btn in w.findChildren(QPushButton):
                            mode = btn.property('mode')
                            if mode == 'home':
                                home_btn = btn
                            elif mode == 'ndisplay_button':
                                nd_btn = btn
                            else:
                                btn.hide()
                        # Clear layout and rebuild centered with spacing
                        if home_btn and nd_btn:
                            while layout.count():
                                layout.takeAt(0)
                            layout.addStretch(1)
                            layout.addWidget(home_btn)
                            layout.addSpacing(12)
                            layout.addWidget(nd_btn)
                            layout.addStretch(1)
                elif class_name == 'QtLayerButtons':
                    delete_btn = None
                    for btn in w.findChildren(QPushButton):
                        tooltip = (btn.toolTip() or '').lower()
                        if 'delete' in tooltip:
                            delete_btn = btn
                        elif btn.objectName() != 'toggleVisibilityBtn':
                            btn.hide()
                    # Add a show/hide all layers toggle button (once)
                    if delete_btn and not getattr(w, '_has_toggle', False):
                        from qtpy.QtSvg import QSvgRenderer
                        from qtpy.QtGui import QImage, QPixmap, QPainter
                        from qtpy.QtCore import QSize, Qt as _Qt

                        def _tinted_icon_from_svg(svg_path, color='white', size=28):
                            renderer = QSvgRenderer(svg_path)
                            img = QImage(QSize(size, size), QImage.Format_ARGB32)
                            img.fill(_Qt.transparent)
                            p = QPainter(img)
                            renderer.render(p)
                            p.setCompositionMode(QPainter.CompositionMode_SourceIn)
                            p.fillRect(img.rect(), QColor(color))
                            p.end()
                            return QIcon(QPixmap.fromImage(img))

                        icons_dir = Path(napari.__file__).parent / 'resources' / 'icons'
                        icon_on = _tinted_icon_from_svg(str(icons_dir / 'visibility.svg'))
                        icon_off = _tinted_icon_from_svg(str(icons_dir / 'visibility_off.svg'))

                        toggle_btn = QPushButton(w)
                        toggle_btn.setObjectName('toggleVisibilityBtn')
                        toggle_btn.setToolTip('Show/hide all layers')
                        toggle_btn.setFixedSize(28, 28)
                        toggle_btn.setIconSize(QSize(20, 20))
                        toggle_btn.setIcon(icon_on)
                        toggle_btn.setStyleSheet(
                            "QPushButton { border: none; }"
                            "QPushButton:hover { background: rgba(255,255,255,30); border-radius: 4px; }"
                        )
                        toggle_btn._all_visible = True
                        def _toggle_all_layers(checked=False, btn=toggle_btn, _on=icon_on, _off=icon_off):
                            btn._all_visible = not btn._all_visible
                            btn.setIcon(_on if btn._all_visible else _off)
                            for layer in viewer.layers:
                                layer.visible = btn._all_visible
                        toggle_btn.clicked.connect(_toggle_all_layers)
                        layout = w.layout()
                        if layout:
                            idx = layout.indexOf(delete_btn)
                            layout.insertWidget(idx, toggle_btn)
                        w._has_toggle = True

    # Run immediately so the user never sees the controls flash
    lock_ui()
    # Run again after Qt finishes layout in case napari adds docks lazily
    QTimer.singleShot(0, lock_ui)

    # Event Binding
    events.install_layer_lock(viewer)
    viewer.layers.events.inserted.connect(partial(events.on_layer_inserted, widgets=widget_map))
    viewer.layers.events.removed.connect(partial(events.on_layer_removed, widgets=widget_map))

    viewer.bind_key('Shift-P', capture_with_hotkey, overwrite=True)
    viewer.bind_key('Shift-G', region_capture_with_hotkey, overwrite=True)
    viewer.bind_key('x', delete_point_under_mouse, overwrite=True)
    viewer.bind_key('c', delete_mask_under_mouse, overwrite=True)

    napari.run()