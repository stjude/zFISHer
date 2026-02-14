import napari
import webbrowser
import math
import warnings
from pathlib import Path
from functools import partial

from napari.utils.theme import get_theme, register_theme
from qtpy.QtWidgets import QApplication, QToolBox, QToolButton, QWidget, QLabel, QVBoxLayout, QDockWidget
from qtpy.QtGui import QIcon, QPainter, QColor, QFont, QPalette
from qtpy.QtCore import Qt, QPoint, QTimer
from magicgui import widgets
import zfisher.core.session as session

# Import all individual widgets from their own scripts
from .widgets.start_session_widget import StartSessionWidget
from .widgets.dapi_segmentation_widget import dapi_segmentation_widget
from .widgets.registration_widget import registration_widget
from .widgets.canvas_widget import canvas_widget
from .widgets.nuclei_matching_widget import nuclei_matching_widget
from .widgets.puncta_widget import puncta_widget
from .widgets.colocalization_widget import colocalization_widget
from .widgets.distance_widget import distance_widget
from .widgets.mask_editor_widget import mask_editor_widget, delete_mask_under_mouse
from .widgets.puncta_editor_widget import puncta_editor_widget, delete_point_under_mouse
from .widgets.capture_widget import capture_widget, capture_with_hotkey

# Import the event handlers
from . import events

# --- Helper Classes ---

class DraggableScaleBar(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.dragging = False
        self.drag_start_position = QPoint()
        self.locked = False
        self.show_pixels = False
        self.pen_color = QColor("white")
        self.font_color = QColor("white")
        self.font = QFont("Arial", 12)
        self.font.setBold(True)
        self.resize(200, 60)
        QTimer.singleShot(1500, self.move_to_bottom_right)
        self.viewer.camera.events.zoom.connect(self.on_zoom)
        self.viewer.layers.events.inserted.connect(self.on_layer_change)
        self.viewer.layers.events.removed.connect(self.on_layer_change)
        self.pixel_size_um = 1.0
        self.bar_length_um = 10
        self.bar_length_px = 100
        self.text = ""
        self.recalculate()

    def move_to_bottom_right(self):
        self.adjustSize()
        parent = self.parent()
        if parent:
            p_w, p_h = parent.width(), parent.height()
            self.move(p_w - self.width() - 20, p_h - self.height() - 20)

    def get_pixel_size(self):
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
        um_per_canvas_px = 1.0 / zoom
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
        
        # Ensure it is opaque and black
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor('#000000'))
        self.setPalette(palette)
        
        # Prevent the widget from blocking clicks when hidden
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignCenter)
        
        # Mint and White branding for the publication-ready look
        self.label = QLabel(
            "<h1 style='font-size: 50px; color: white; margin-bottom: 0px;'>zFISHer</h1>"
            "<p style='font-size: 24px; color: #b2f2bb; margin-top: 0px;'>Version 2.0</p>"
        )
        self.label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.label)

        # Connect to viewer events to handle auto-hide
        self.viewer.layers.events.inserted.connect(self._check_visibility)
        self.viewer.layers.events.removed.connect(self._check_visibility)

        if self.parent():
            QTimer.singleShot(100, self.resize_to_parent)

    def _check_visibility(self, event=None):
        """Hides when layers exist, shows when empty."""
        if len(self.viewer.layers) > 0:
            self.hide()
            self.setEnabled(False) 
        else:
            self.show()
            self.raise_() 
            self.setEnabled(True)

    def paintEvent(self, event):
        """Force paint the background black to ensure no native leaks."""
        if self.isVisible():
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            
    def resize_to_parent(self):
        if not self.parent(): return
        self.resize(self.parent().size())
        self.move(0, 0)

# --- Creation Logic ---

def create_welcome_widget(viewer):
    """The 'Home' tab content for the sidebar."""
    container = widgets.Container(labels=False)
    # Using Mint color (#b2f2bb) for consistency
    container.append(widgets.Label(value="<h1 style='color: #b2f2bb;'>Welcome to zFISHer</h1>"))
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
    app = QApplication.instance() or QApplication([])
    
    # --- 1. Amethyst & Mint Theme Registration ---
    try:
        custom_theme = get_theme('dark')
        custom_theme.background = '#1a1421' # Deep Amethyst Purple
        custom_theme.canvas = '#000000'     # Black Canvas
        custom_theme.primary = '#b2f2bb'    # Classy Mint
        register_theme('zfisher_theme', custom_theme, 'dark')
    except Exception as e:
        print(f"Theme registration failed: {e}")

    icon_path = Path(__file__).parent.parent.parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # --- 2. Create Viewer ---
    viewer = napari.Viewer(title="zFISHer - 3D Colocalization", ndisplay=2, show_welcome_screen=False)
    #viewer.QtViewer.setAttribute(show_welcome_screen=False) # Allow transparency for custom welcome screen
    # NATIVE FIX: Hide the built-in welcome screen by targeting the widget
    if hasattr(viewer.window.qt_viewer, 'welcome_widget'):
        viewer.window.qt_viewer.welcome_widget.setVisible(False)
    
    try:
        viewer.theme = 'zfisher_theme'
    except Exception as e:
        print(f"Could not apply theme: {e}")

    viewer.scale_bar.visible = False
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        viewer_canvas_native = viewer.window.qt_viewer.canvas.native

    # Setup Custom Welcome Overlay
    welcome_widget = WelcomeWidget(viewer, parent=viewer_canvas_native)
    viewer.window.custom_welcome_widget = welcome_widget

    # Custom Scale Bar
    scale_bar_widget = DraggableScaleBar(viewer, parent=viewer_canvas_native)
    viewer.window.custom_scale_bar = scale_bar_widget
    
    widget_map = {
        "dapi_segmentation": dapi_segmentation_widget,
        "registration": registration_widget,
        "nuclei_matching": nuclei_matching_widget,
        "mask_editor": mask_editor_widget,
        "puncta_detection": puncta_widget,
        "puncta_editor": puncta_editor_widget,
        "colocalization": colocalization_widget,
        "capture": capture_widget,
        "start_session": StartSessionWidget(viewer),
        "canvas": canvas_widget,
        "distance": distance_widget,
    }

    widgets_to_add = [
        (create_welcome_widget(viewer), "zFISHer Home"),
        (StartSessionWidget(viewer), "1. Start Session"),
        (dapi_segmentation_widget, "2. DAPI Mapping"),
        (registration_widget, "3. Registration"),
        (canvas_widget, "4. Global Canvas"),
        (nuclei_matching_widget, "5. Match Nuclei"),
        (mask_editor_widget, "Mask Editor"),
        (puncta_widget, "6. Puncta Detection"),
        (puncta_editor_widget, "Puncta Editor"),
        (distance_widget, "7. Simple Export"),
        (colocalization_widget, "8. Colocalization & Export"),
        (capture_widget, "Capture View")
    ]

    # --- 3. Sidebar Toolbox Styling ---
    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    toolbox.setStyleSheet("""
        QToolBox::tab {
            color: #b2f2bb;       /* Mint Text */
            background: #251f2e;  /* Amethyst Background */
            font-weight: bold;
            border-radius: 4px;
        }
        QToolBox::tab:selected {
            background: #352c42;
            border: 1px solid #b2f2bb;
        }
        QLabel { 
            qproperty-alignment: 'AlignVCenter | AlignLeft'; 
        }
    """)
    
    for widget, name in widgets_to_add:
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        toolbox.addItem(widget.native, name)

    # --- 4. Dock Widget and UI Lock ---
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

    if icon_path.exists() and hasattr(viewer.window, '_qt_window'):
        viewer.window._qt_window.setWindowIcon(QIcon(str(icon_path)))

    napari.run()