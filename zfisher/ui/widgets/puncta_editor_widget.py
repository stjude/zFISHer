import napari
from magicgui import magicgui, widgets
import numpy as np

from ...core import session
from ..decorators import require_active_session

@magicgui(
    call_button="Delete Selected Points",
    points_layer={"label": "Layer to Edit"},
    point_size={"label": "Display Size", "min": 0, "max": 20, "value": 3},
    show_all_z={"label": "Project All Z (Out of Slice)"}
)
@require_active_session()
def puncta_editor_widget(
    points_layer: "napari.layers.Points",
    point_size: int = 3,
    show_all_z: bool = True
):
    """Enhanced editor for high-density puncta curation."""
    if points_layer:
        # Batch update visibility settings
        points_layer.size = point_size
        points_layer.out_of_slice = show_all_z
        points_layer.projection_mode = 'all' if show_all_z else 'none'
        
        if points_layer.mode == 'select' and len(points_layer.selected_data) > 0:
            points_layer.remove_selected()

# --- Sync UI Sliders to Layer ---
@puncta_editor_widget.point_size.changed.connect
def _on_size_change(value):
    if puncta_editor_widget.points_layer.value:
        puncta_editor_widget.points_layer.value.size = value

@puncta_editor_widget.show_all_z.changed.connect
def _on_projection_change(value):
    layer = puncta_editor_widget.points_layer.value
    if layer:
        layer.out_of_slice = value
        layer.projection_mode = 'all' if value else 'none'

# --- Toolbar Construction ---
pe_lbl = widgets.Label(value="<b>Editing Hub:</b>")
pe_container = widgets.Container(layout="horizontal", labels=False)
pe_add_chk = widgets.CheckBox(text="Add (A)")
pe_select_chk = widgets.CheckBox(text="Select (S)")
pe_pan_btn = widgets.PushButton(text="Pan/Zoom (Z)")

pe_container.extend([pe_add_chk, pe_select_chk, pe_pan_btn])
puncta_editor_widget.insert(0, pe_lbl)
puncta_editor_widget.insert(1, pe_container)

# --- Mouse & Hotkey Logic ---
def delete_point_under_mouse(viewer):
    """Rapid deletion for cleaning up noisy detection."""
    layer = puncta_editor_widget.points_layer.value
    if not layer or not layer.visible:
        return

    # Find point index under cursor
    val = layer.get_value(
        viewer.cursor.position, 
        view_direction=viewer.camera.view_direction, 
        dims_displayed=list(viewer.dims.displayed), 
        world=True
    )
    
    if val is not None:
        layer.selected_data = {val}
        layer.remove_selected()
        viewer.status = f"Deleted spot {val}"

# Register Hotkeys in the Viewer
def register_editor_hotkeys(viewer):
    @viewer.bind_key('x', overwrite=True)
    def _delete_hotkey(v):
        delete_point_under_mouse(v)
    
    @viewer.bind_key('a', overwrite=True)
    def _add_mode(v):
        pe_add_chk.value = True
    
    @viewer.bind_key('s', overwrite=True)
    def _select_mode(v):
        pe_select_chk.value = True

# Mode Toggles (Mirrored from your current logic)
@puncta_editor_widget.points_layer.changed.connect
def _on_layer_change(new_layer):
    if new_layer:
        # Calculate mean size but ensure it's at least 1
        avg_size = int(np.mean(new_layer.size))
        puncta_editor_widget.point_size.value = max(1, avg_size) 
        
        # Keep the getattr fix for out_of_slice
        oos_val = getattr(new_layer, 'out_of_slice_dist', getattr(new_layer, 'out_of_slice', True))
        puncta_editor_widget.show_all_z.value = bool(oos_val)