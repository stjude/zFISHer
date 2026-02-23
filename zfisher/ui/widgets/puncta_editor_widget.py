import napari
from magicgui import magicgui, widgets
import numpy as np
from scipy.spatial import cKDTree

from ...core import session
from ..decorators import require_active_session

@magicgui(
    call_button="Delete Selected Points",
    points_layer={"label": "Layer to Edit"},
    point_size={"label": "Display Size", "min": 0, "max": 20, "value": 3},
    show_all_z={"label": "Project All Z (Out of Slice)"},
    fishing_hook={"label": "Enable Fishing Hook", "value": True},
    volume_optimization={"label": "Volume Optimization", "value": True},
    opt_radius={"label": "Opt. Radius (um)", "value": "0.1"}
)
@require_active_session()
def puncta_editor_widget(
    points_layer: "napari.layers.Points",
    point_size: int = 3,
    show_all_z: bool = True,
    fishing_hook: bool = True,           # <--- ADD THIS
    volume_optimization: bool = True,    # <--- ADD THIS
    opt_radius: str = "0.1"              # <--- ADD THIS
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
        
        # --- ATTACH FISHING HOOK CALLBACK ---
        if fishing_hook_callback not in new_layer.mouse_drag_callbacks:
            new_layer.mouse_drag_callbacks.append(fishing_hook_callback)
        
        # Keep the getattr fix for out_of_slice
        oos_val = getattr(new_layer, 'out_of_slice_dist', getattr(new_layer, 'out_of_slice', True))
        puncta_editor_widget.show_all_z.value = bool(oos_val)

def calculate_fishing_hook(img_layer, event, viewer, use_optimization=True, radius_um=0.1):
    """Pure math: Ray-casts and returns the optimized 3D coordinate."""
    # 1. Ray-Casting
    near_point, far_point = img_layer.get_ray_intersections(
        event.position,
        view_direction=viewer.camera.view_direction,
        dims_displayed=list(viewer.dims.displayed), 
        world=True
    )
    
    if near_point is None:
        return None

    # 2. Intensity Sampling (500 steps for 71-slice stacks)
    steps = np.linspace(near_point, far_point, num=500)
    intensities = np.nan_to_num([img_layer.get_value(s, world=True) or 0 for s in steps])
    target_coord = steps[np.argmax(intensities)]
    
    # 3. Volume Optimization (Voxel-accurate refinement)
    if use_optimization:
        voxels = session.get_data("voxels", (1.0, 1.0, 1.0)) 
        z_rad, y_rad, x_rad = [int(max(1, radius_um / v)) for v in voxels]

        z, y, x = np.round(target_coord).astype(int)
        z_s, y_s, x_s = (slice(max(0, z-z_rad), z+z_rad+1), 
                         slice(max(0, y-y_rad), y+y_rad+1), 
                         slice(max(0, x-x_rad), x+x_rad+1))

        local_vol = img_layer.data[z_s, y_s, x_s]
        if local_vol.size > 0:
            local_peak = np.unravel_index(np.argmax(local_vol), local_vol.shape)
            target_coord = np.array([z_s.start + local_peak[0], 
                                     y_s.start + local_peak[1], 
                                     x_s.start + local_peak[2]])
    return target_coord


def fishing_hook_callback(layer, event):
    """Mouse callback to handle Shift+Click puncta placement."""
    if 'Shift' not in event.modifiers or not puncta_editor_widget.fishing_hook.value:
        return

    viewer = napari.current_viewer()
    img_layer = next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and l.visible), None)
    
    if not img_layer or not layer:
        return

    target_coord = calculate_fishing_hook(
        img_layer, event, viewer, 
        use_optimization=puncta_editor_widget.volume_optimization.value,
        radius_um=float(puncta_editor_widget.opt_radius.value)
    )

    if target_coord is not None:
        # Collision Guard: 1.5px radius
        if len(layer.data) > 0 and cKDTree(layer.data).query(target_coord, k=1)[0] < 1.5:
            return

        # Safety: Block set_data events during the add to prevent IndexError
        with layer.events.set_data.blocker():
            layer.add(target_coord)
        layer.refresh()
        viewer.status = f"Hooked puncta at {np.round(target_coord, 1)}"