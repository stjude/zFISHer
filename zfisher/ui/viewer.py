import napari
import webbrowser
import math
import warnings
from pathlib import Path
from functools import partial

from qtpy.QtWidgets import QApplication, QToolBox, QToolButton, QWidget, QLabel, QVBoxLayout, QDockWidget
from qtpy.QtGui import QIcon, QPainter, QPalette
from qtpy.QtCore import Qt, QPoint, QTimer, QEvent
from magicgui import widgets
from ..core import session

# --- RESTORED IMPORTS ---
# Import all the individual widgets from their own scripts
from .widgets.start_session_widget import StartSessionWidget
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
from .widgets.capture_widget import capture_widget, capture_with_hotkey

# Import the event handlers
from . import events, style

# --- Helper Classes ---

from qtpy.QtCore import Qt, QPoint, QTimer, QEvent, QEvent

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
        
        # This part has a bug if no layers are present. Let's get scale from layer if possible.
        active_layer = self.viewer.layers.selection.active
        if active_layer:
             pixel_size_x = active_layer.scale[-1]
        else:
             pixel_size_x = 1.0 # fallback

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
        
        # Branding (Mint and White)
        label_html = (
            f"<h1 {style.WELCOME_WIDGET_STYLE['h1']}>zFISHer</h1>"
            f"<p {style.WELCOME_WIDGET_STYLE['p']}>Version 2.0</p>"
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

def create_welcome_widget(viewer):
    container = widgets.Container(labels=False)
    container.append(widgets.Label(value=f"<h1 {style.CREATE_WELCOME_WIDGET_STYLE['h1']}>Welcome to zFISHer</h1>"))
    container.append(widgets.Label(value="<em>Multiplexed Sequential FISH Analysis</em>"))
    container.append(widgets.Label(value="<p>Version 2.0</p>"))
    container.append(widgets.Label(value="<h3>Workflow:</h3>"))
    container.append(widgets.Label(value="1. <b>Load Data</b> (.nd2 files)"))
    container.append(widgets.Label(value="2. <b>Segment Nuclei</b> (DAPI)"))
    container.append(widgets.Label(value="3. <b>Register Rounds</b> (RANSAC)"))
    container.append(widgets.Label(value="4. <b>Generate Canvas</b> (Warp)"))
    container.append(widgets.Label(value="5. <b>Match Nuclei</b>"))
    container.append(widgets.Label(value="6. <b>Detect Puncta</b> (Spots)"))
    container.append(widgets.Label(value="7. <b>Analysis Export</b>"))
    
    btn_row = widgets.Container(layout="horizontal", labels=False)
    help_btn = widgets.PushButton(text="Open README / Help")
    reset_btn = widgets.PushButton(text="Reset")
    btn_row.extend([help_btn, reset_btn])
    container.append(btn_row)
    
    @help_btn.changed.connect
    def open_help():
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        if readme_path.exists():
            webbrowser.open(readme_path.as_uri())
            
    @reset_btn.changed.connect
    def reset_viewer():
        viewer.layers.clear()
        session.clear_session()
        if hasattr(viewer.window, 'custom_scale_bar'):
            viewer.window.custom_scale_bar.hide()
            
    return container

def launch_zfisher():
    # FIX: Ensure QApplication is initialized properly
    app = QApplication.instance() or QApplication([])
    
    # --- 1. Amethyst & Mint Theme (RE-ENABLED) ---
    theme_name = style.register_napari_theme()

    # Build icon — generate .ico from .png for Windows taskbar compatibility
    icon_dir = Path(__file__).parent.parent.parent
    ico_path = icon_dir / "icon.ico"
    png_path = icon_dir / "icon.png"
    if not ico_path.exists() and png_path.exists():
        try:
            from PIL import Image
            img = Image.open(png_path)
            img.save(ico_path, format="ICO", sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
        except Exception:
            ico_path = png_path
    icon_file = ico_path if ico_path.exists() else png_path
    icon = QIcon(str(icon_file)) if icon_file.exists() else QIcon()
    app.setWindowIcon(icon)

    # --- 2. Create Viewer ---
    viewer = napari.Viewer(title="zFISHer - 3D Colocalization", ndisplay=2)

    # Re-apply to the napari QMainWindow directly
    if not icon.isNull():
        viewer.window._qt_window.setWindowIcon(icon)

    # Force icon onto the Win32 window handle (bypasses Qt for taskbar)
    if icon_file.exists() and str(icon_file).lower().endswith('.ico'):
        try:
            import ctypes
            ICON_SMALL, ICON_BIG, WM_SETICON = 0, 1, 0x80
            LR_LOADFROMFILE = 0x10
            hwnd = int(viewer.window._qt_window.winId())
            hicon = ctypes.windll.user32.LoadImageW(None, str(icon_file), 1, 0, 0, LR_LOADFROMFILE)
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
        except Exception:
            pass
    
    # Permanently disable napari's native welcome screen
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
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        viewer_canvas_native = viewer.window.qt_viewer.canvas.native

    # Setup Welcome Overlay
    welcome_widget = WelcomeWidget(viewer, parent=viewer_canvas_native)
    viewer.window.custom_welcome_widget = welcome_widget

    scale_bar_widget = DraggableScaleBar(viewer, parent=viewer_canvas_native)
    viewer.window.custom_scale_bar = scale_bar_widget
    
    alignment_consensus_widget = AlignmentConsensusWidget(viewer)
    puncta_picking_widget = PunctaPickingWidget(viewer)
    export_visualization_widget = ExportVisualizationWidget(viewer)
    
    widget_map = {
        # Child widgets are now accessed via the parent composite widget
        "dapi_segmentation": alignment_consensus_widget.dapi_widget,
        "registration": alignment_consensus_widget.registration_widget,
        "canvas": alignment_consensus_widget.canvas_widget,
        "nuclei_matching": alignment_consensus_widget.nuclei_matching_widget,
        "mask_editor": alignment_consensus_widget.mask_editor_widget,
        "automated_preprocessing": alignment_consensus_widget.automated_widget,
        # Other top-level widgets
        "puncta_detection": puncta_picking_widget.algorithmic_widget,
        "puncta_editor": puncta_picking_widget.manual_widget,
        "colocalization": export_visualization_widget.export_widget,
        "capture": export_visualization_widget.capture_widget,
        "start_session": StartSessionWidget(viewer),
        "alignment_consensus": alignment_consensus_widget, # The parent itself
        "puncta_picking": puncta_picking_widget,
        "export_visualization": export_visualization_widget,
    }

    widgets_to_add = [
        (create_welcome_widget(viewer), "zFISHer Home"),
        (StartSessionWidget(viewer), "1. Session && I/O"),
        (alignment_consensus_widget, "2. Alignment && Consensus"),
        (puncta_picking_widget, "3. Puncta Picking"),
        (export_visualization_widget, "4. Export && Visualization")
    ]

    # --- 3. Sidebar Toolbox Styling ---
    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    toolbox.setStyleSheet(style.TOOLBOX_STYLESHEET)
    
    for widget, name in widgets_to_add:
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        toolbox.addItem(widget.native, name)

    dock_widget = viewer.window.add_dock_widget(toolbox, area="right", name="zFISHer Workflow")

    def lock_ui():
        if dock_widget:
            dock_widget.setFeatures(QDockWidget.NoDockWidgetFeatures)
            empty_title = QWidget()
            empty_title.setFixedHeight(0)
            dock_widget.setTitleBarWidget(empty_title)

    QTimer.singleShot(1000, lock_ui)

    # Event Binding
    viewer.layers.events.inserted.connect(partial(events.on_layer_inserted, widgets=widget_map))
    viewer.layers.events.removed.connect(partial(events.on_layer_removed, widgets=widget_map))

    viewer.bind_key('Shift-P', capture_with_hotkey, overwrite=True)
    viewer.bind_key('x', delete_point_under_mouse, overwrite=True)
    viewer.bind_key('c', delete_mask_under_mouse, overwrite=True)

    napari.run()