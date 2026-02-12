import napari
import webbrowser
from pathlib import Path
from functools import partial

from qtpy.QtWidgets import QApplication, QToolBox, QToolButton
from qtpy.QtGui import QIcon
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
from .widgets.mask_editor_widget import mask_editor_widget
from .widgets.puncta_editor_widget import puncta_editor_widget
from .widgets.capture_widget import capture_widget, capture_with_hotkey

# Import the event handlers
from . import events

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

    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    
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

    napari.run()