import napari
import webbrowser
import math
import warnings
from pathlib import Path
from functools import partial

from qtpy.QtWidgets import QApplication, QToolBox, QToolButton, QWidget
from qtpy.QtGui import QIcon, QPainter, QColor, QFont
from qtpy.QtCore import Qt, QPoint, QTimer
from magicgui import widgets
import zfisher.core.session as session

# Import all the individual widgets from their own scripts
from .widgets.file_selector_widget import file_selector_widget
from .widgets.load_session_widget import load_session_widget
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

class DraggableScaleBar(QWidget):
    """
    A custom scale bar widget that can be dragged anywhere on the viewer canvas.
    """
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.dragging = False
        self.drag_start_position = QPoint()
        self.locked = False
        self.show_pixels = False
        
        # Style settings
        self.pen_color = QColor("white")
        self.font_color = QColor("white")
        self.font = QFont("Arial", 12)
        self.font.setBold(True)
        
        # Initial geometry
        self.resize(200, 60)
        
        # Default to Bottom-Right
        # Use QTimer to ensure parent has correct size after layout
        QTimer.singleShot(1500, self.move_to_bottom_right)
        
        # Connect events
        self.viewer.camera.events.zoom.connect(self.on_zoom)
        self.viewer.layers.events.inserted.connect(self.on_layer_change)
        self.viewer.layers.events.removed.connect(self.on_layer_change)
        
        self.pixel_size_um = 1.0
        self.bar_length_um = 10
        self.bar_length_px = 100
        self.text = ""
        
        self.recalculate()
        self.show()

    def move_to_bottom_right(self):
        self.adjustSize()
        parent = self.parent()
        if parent:
            p_w, p_h = parent.width(), parent.height()
            self.move(p_w - self.width() - 20, p_h - self.height() - 20)

    def get_pixel_size(self):
        # We assume world coordinates are in microns because zFISHer sets layer.scale
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

def create_welcome_widget(viewer):
    """Creates a welcome widget with instructions and a help button."""
    container = widgets.Container(labels=False)
    
    # HTML-subset styling is supported in Qt labels
    container.append(widgets.Label(value="<h1 style='color: #00FFFF;'>zFISHer</h1>"))
    container.append(widgets.Label(value="<em>Multiplexed Sequential FISH Analysis</em>"))
    
    container.append(widgets.Label(value="<h3>Workflow:</h3>"))
    container.append(widgets.Label(value="1. <b>Load Data</b> (.nd2 files)"))
    container.append(widgets.Label(value="2. <b>Segment Nuclei</b> (DAPI)"))
    container.append(widgets.Label(value="3. <b>Register Rounds</b> (RANSAC)"))
    container.append(widgets.Label(value="4. <b>Generate Canvas</b> (Warp)"))
    container.append(widgets.Label(value="5. <b>Match Nuclei</b>"))
    container.append(widgets.Label(value="6. <b>Detect Puncta</b> (Spots)"))
    container.append(widgets.Label(value="7. <b>Analysis Export</b>"))
    
    # Buttons Row
    btn_row = widgets.Container(layout="horizontal", labels=False)
    help_btn = widgets.PushButton(text="Open README / Help")
    reset_btn = widgets.PushButton(text="Reset")
    btn_row.extend([help_btn, reset_btn])
    container.append(btn_row)
    
    @help_btn.changed.connect
    def open_help():
        # Look for README in project root
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        if readme_path.exists():
            webbrowser.open(readme_path.as_uri())
        else:
            print(f"README not found at {readme_path}")
            
    @reset_btn.changed.connect
    def reset_viewer():
        viewer.layers.clear()
        session.clear_session()
        viewer.status = "Viewer cleared."
            
    return container

def launch_zfisher():
    # Set App Icon
    app = QApplication.instance() or QApplication([])
    icon_path = Path(__file__).parent.parent.parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    viewer = napari.Viewer(title="zFISHer - 3D Colocalization", ndisplay=2)

    # The following code is commented out because the napari API has changed,
    # and it was causing a crash. The 'new shapes layer' button is currently visible.
    # # Hide the "new shapes" layer button as requested
    # try:
    #     # Access the controls widget which contains the layer buttons
    #     controls = viewer.window._qt_viewer.controls
    #     layer_buttons = controls.layer_buttons
        
    #     found_button = False
    #     # Use a more specific search to avoid hiding the wrong button
    #     for btn in layer_buttons.findChildren(QToolButton):
    #         tooltip = btn.toolTip().lower()
    #         if 'new shapes layer' in tooltip:
    #             btn.setVisible(False)
    #             found_button = True
    #             break # Exit after finding the button
        
    #     if not found_button:
    #         # This will print to the console if the button wasn't found.
    #         print("DEBUG: The 'new shapes layer' button could not be found to be hidden.")
            
    # except Exception as e:
    #     # Make errors more visible in the console
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #     print(f"ERROR: An exception occurred while trying to hide the shapes layer button: {e}")
    #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("INFO: The 'new shapes layer' button is currently visible due to a napari API change.")

    # Disable native scale bar and use custom draggable one
    viewer.scale_bar.visible = False
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        viewer_canvas_native = viewer.window.qt_viewer.canvas.native

    scale_bar_widget = DraggableScaleBar(viewer, parent=viewer_canvas_native)
    viewer.window.custom_scale_bar = scale_bar_widget
    
    # This dictionary holds all widget objects, keyed by a simple name.
    # It's passed to the event handlers so they can update the correct widget.
    widget_map = {
        "dapi_segmentation": dapi_segmentation_widget,
        "registration": registration_widget,
        "nuclei_matching": nuclei_matching_widget,
        "mask_editor": mask_editor_widget,
        "puncta_detection": puncta_widget,
        "puncta_editor": puncta_editor_widget,
        "colocalization": colocalization_widget,
        "capture": capture_widget,
        # These are not modified by events, but included for completeness
        "file_selector": file_selector_widget,
        "load_session": load_session_widget,
        "canvas": canvas_widget,
        "distance": distance_widget,
    }

    # The order of widgets in the UI
    widgets_to_add = [
        (create_welcome_widget(viewer), "Home"),
        (load_session_widget, "Resume Session"),
        (file_selector_widget, "1. File Selection"),
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

    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    toolbox.setStyleSheet("QLabel { qproperty-alignment: 'AlignVCenter | AlignLeft'; }")
    
    for widget, name in widgets_to_add:
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        
        toolbox.addItem(widget.native, name)

    viewer.window.add_dock_widget(toolbox, area="right", name="zFISHer Workflow")

    # Connect layer events to handlers, passing the widget_map to each.
    viewer.layers.events.inserted.connect(partial(events.on_layer_inserted, widgets=widget_map))
    viewer.layers.events.removed.connect(partial(events.on_layer_removed, widgets=widget_map))

    # --- Register Hotkeys ---
    viewer.bind_key('p', capture_with_hotkey, overwrite=True)
    viewer.bind_key('x', delete_point_under_mouse, overwrite=True)
    viewer.bind_key('c', delete_mask_under_mouse, overwrite=True)

    napari.run()