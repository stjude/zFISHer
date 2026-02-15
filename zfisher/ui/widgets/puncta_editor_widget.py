import napari
from magicgui import magicgui, widgets

import zfisher.core.session as session
from .. import popups
from ..decorators import require_active_session

@magicgui(
    call_button="Delete Selected Points",
    points_layer={"label": "Layer to Edit"}
)
@require_active_session("Please start or load a session before editing puncta.")
def puncta_editor_widget(points_layer: "napari.layers.Points"):
    """Editor for manually adding/removing puncta."""
    viewer = napari.current_viewer()

    if points_layer:
        if points_layer.mode == 'select':
            if len(points_layer.selected_data) > 0:
                points_layer.remove_selected()
                viewer.status = f"Deleted selected points in {points_layer.name}."
            else:
                viewer.status = "No points selected."
        else:
            viewer.status = "Switch to Select Mode to delete points."

# Tools for Puncta Editor
pe_lbl = widgets.Label(value="<b>Editing Tools:</b>")
pe_container = widgets.Container(layout="horizontal", labels=False)
pe_add_chk = widgets.CheckBox(text="Add Mode")
pe_select_chk = widgets.CheckBox(text="Select Mode")
pe_pan_btn = widgets.PushButton(text="Pan/Zoom")

pe_container.extend([pe_add_chk, pe_select_chk, pe_pan_btn])
puncta_editor_widget.insert(0, pe_lbl)
puncta_editor_widget.insert(1, pe_container)

hotkey_lbl = widgets.Label(value="Hotkey: 'X' (Hover to Delete)")
puncta_editor_widget.insert(2, hotkey_lbl)

def delete_point_under_mouse(viewer):
    """Deletes the point currently under the mouse cursor in the active editor layer."""
    layer = puncta_editor_widget.points_layer.value
    
    if not layer or not isinstance(layer, napari.layers.Points):
        return
        
    # Only allow deletion if the layer is visible
    if not layer.visible:
        return

    # Get value at cursor position
    val = layer.get_value(
        viewer.cursor.position, 
        view_direction=viewer.camera.view_direction, 
        dims_displayed=list(viewer.dims.displayed), 
        world=True
    )
    
    if val is not None:
        # Select and remove
        layer.selected_data = {val}
        layer.remove_selected()
        viewer.status = f"Deleted point {val} from {layer.name}"

@require_active_session("Please start or load a session before editing puncta.")
def _on_pe_add(value: bool):
    if not session.get_data("output_dir"): # Check again in case session was closed
        pe_add_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = puncta_editor_widget.points_layer.value
    if layer:
        if value:
            pe_select_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'add'
            viewer.status = "Add Points Mode."
        elif layer.mode == 'add':
            layer.mode = 'pan_zoom'
            viewer.status = "Pan/Zoom Mode."

@require_active_session("Please start or load a session before editing puncta.")
def _on_pe_select(value: bool):
    if not session.get_data("output_dir"): # Check again in case session was closed
        pe_select_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = puncta_editor_widget.points_layer.value
    if layer:
        if value:
            pe_add_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'select'
            viewer.status = "Select Mode. Click/Drag to select, then click 'Delete Selected'."
        elif layer.mode == 'select':
            layer.mode = 'pan_zoom'
            viewer.status = "Pan/Zoom Mode."

@require_active_session("Please start or load a session before editing puncta.")
def _on_pe_pan():
    viewer = napari.current_viewer()
    layer = puncta_editor_widget.points_layer.value
    if layer:
        pe_add_chk.value = False
        pe_select_chk.value = False
        layer.mode = 'pan_zoom'
        viewer.status = "Pan/Zoom Mode."

# Connect signals after defining functions
pe_add_chk.changed.connect(_on_pe_add)
pe_select_chk.changed.connect(_on_pe_select)
pe_pan_btn.clicked.connect(_on_pe_pan)
