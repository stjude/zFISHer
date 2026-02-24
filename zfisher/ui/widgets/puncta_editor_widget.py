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
    fishing_hook: bool = True,
    volume_optimization: bool = True,
    opt_radius: str = "0.1"
):
    """Enhanced editor for high-density puncta curation."""
    if points_layer:
        points_layer.size = point_size
        points_layer.out_of_slice = show_all_z
        points_layer.projection_mode = 'all' if show_all_z else 'none'
        if points_layer.mode == 'select' and len(points_layer.selected_data) > 0:
            points_layer.remove_selected()

# --- UI Connections ---
@puncta_editor_widget.point_size.changed.connect
def _on_size_change(value):
    if puncta_editor_widget.points_layer.value: puncta_editor_widget.points_layer.value.size = value

@puncta_editor_widget.show_all_z.changed.connect
def _on_projection_change(value):
    layer = puncta_editor_widget.points_layer.value
    if layer:
        layer.out_of_slice = value
        layer.projection_mode = 'all' if value else 'none'

# --- Toolbar Construction ---
pe_lbl = widgets.Label(value="<b>Editing Hub:</b>")
pe_container = widgets.Container(layout="horizontal", labels=False)
pe_add_chk, pe_select_chk, pe_pan_btn = widgets.CheckBox(text="Add (A)"), widgets.CheckBox(text="Select (S)"), widgets.PushButton(text="Pan/Zoom (Z)")
pe_container.extend([pe_add_chk, pe_select_chk, pe_pan_btn])
puncta_editor_widget.insert(0, pe_lbl)
puncta_editor_widget.insert(1, pe_container)

# --- Mouse & Hotkey Logic ---
def delete_point_under_mouse(viewer):
    layer = puncta_editor_widget.points_layer.value
    if not layer or not layer.visible: return
    val = layer.get_value(viewer.cursor.position, view_direction=viewer.camera.view_direction, dims_displayed=list(viewer.dims.displayed), world=True)
    if val is not None:
        layer.selected_data = {val}; layer.remove_selected()
        viewer.status = f"Deleted spot {val}"

def register_editor_hotkeys(viewer):
    @viewer.bind_key('x', overwrite=True)
    def _delete_hotkey(v): delete_point_under_mouse(v)
    @viewer.bind_key('a', overwrite=True)
    def _add_mode(v): pe_add_chk.value = True
    @viewer.bind_key('s', overwrite=True)
    def _select_mode(v): pe_select_chk.value = True

@puncta_editor_widget.points_layer.changed.connect
def _on_layer_change(new_layer):
    if new_layer:
        puncta_editor_widget.point_size.value = max(1, int(np.mean(new_layer.size))) 
        if fishing_hook_callback not in new_layer.mouse_drag_callbacks:
            new_layer.mouse_drag_callbacks.append(fishing_hook_callback)
        oos_val = getattr(new_layer, 'out_of_slice_dist', getattr(new_layer, 'out_of_slice', True))
        puncta_editor_widget.show_all_z.value = bool(oos_val)

# --- 1. Algorithmic Math (Operates strictly on Pixel Arrays) ---
# --- 1. Algorithmic Math (Operates strictly on Pixel Arrays) ---
def calculate_fishing_hook(img_layer, data_pos, viewer, use_optimization=True, radius_um=0.1):
    """Calculates peak intensity using a data coordinate, not a world coordinate."""
    # 1. Get the camera direction (World Space)
    view_direction = np.array(viewer.camera.view_direction)
    
    # 2. NEW: Convert the direction vector to Data Space (Pixels)
    data_view_dir = view_direction / img_layer.scale
    
    # 3. NEW: Normalize the vector so our 500-pixel steps are mathematically accurate
    data_view_dir = data_view_dir / np.linalg.norm(data_view_dir)
    
    # 4. Apply the corrected pixel vector
    start_pt = data_pos - data_view_dir * 500
    end_pt = data_pos + data_view_dir * 500
    
    steps = np.linspace(start_pt, end_pt, num=1000) # Increased steps for better sampling
    
    intensities = []
    valid_coords = []

    for s in steps:
        coords = np.round(s).astype(int)
        if all(0 <= coords[i] < img_layer.data.shape[i] for i in range(len(coords))):
            intensities.append(img_layer.data[tuple(coords)])
            valid_coords.append(s)

    if not intensities:
        return None

    # Find the coordinate corresponding to the max intensity
    target_coord_data = valid_coords[np.argmax(np.nan_to_num(intensities))]

    if use_optimization:
        voxels = img_layer.scale 
        z_rad, y_rad, x_rad = [int(max(1, radius_um / v)) for v in voxels]
        z, y, x = np.round(target_coord_data).astype(int)
        
        z_s, y_s, x_s = (slice(max(0, z-z_rad), z+z_rad+1), 
                         slice(max(0, y-y_rad), y+y_rad+1), 
                         slice(max(0, x-x_rad), x+x_rad+1))

        local_vol = img_layer.data[z_s, y_s, x_s]
        if local_vol.size > 0:
            local_peak = np.unravel_index(np.argmax(local_vol), local_vol.shape)
            target_coord_data = np.array([z_s.start + local_peak[0], 
                                         y_s.start + local_peak[1], 
                                         x_s.start + local_peak[2]])
            
    return target_coord_data

# --- 2. The Yield Callback (Emulates puncta.py segmentation) ---
def fishing_hook_callback(layer, event):
    if 'Shift' not in event.modifiers or not puncta_editor_widget.fishing_hook.value:
        return

    viewer = napari.current_viewer()
    img_layer = next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and l.visible), None)
    if not img_layer or not layer: return
    
    # This is the critical change: ensure transforms are aligned so we can use data coordinates directly.
    if not np.allclose(layer.scale, img_layer.scale) or not np.allclose(layer.translate, img_layer.translate):
        layer.scale = img_layer.scale
        layer.translate = img_layer.translate

    # Convert mouse position from the Points layer's world/canvas space to the IMAGE data space.
    # This ensures that even if transforms were misaligned, we get the right starting point.
    cursor_pos_data = img_layer.world_to_data(viewer.cursor.position)

    target_coord_data = calculate_fishing_hook(
        img_layer, 
        cursor_pos_data,
        viewer, # Pass viewer for camera info
        use_optimization=puncta_editor_widget.volume_optimization.value,
        radius_um=float(puncta_editor_widget.opt_radius.value)
    )

    # GENERATOR: Let napari drop its native point first, then we process it
    yield 
    while event.type == 'mouse_move':
        yield
        
    if target_coord_data is not None:
        new_data = layer.data.copy()
        
        # --- NEW: Safe Collision Guard ---
        # If in 'add' mode, napari just added a point at the end of new_data.
        # We must exclude it from our search to avoid colliding with ourselves.
        search_data = new_data[:-1] if layer.mode == 'add' and len(new_data) > 0 else new_data
        
        if len(search_data) > 0:
            dist, _ = cKDTree(search_data).query(target_coord_data, k=1)
            if dist < 1.5:  # Collision detected (1.5 pixel radius)
                # Rollback napari's native broken point
                if layer.mode == 'add' and len(new_data) > 0:
                    layer.data = new_data[:-1] 
                viewer.status = "Point already exists here."
                return
        # ---------------------------------

        if layer.mode == 'add' and len(new_data) > 0:
            new_data[-1] = target_coord_data # Overwrite napari's bad point
        else:
            new_data = np.vstack((new_data, target_coord_data))
            
        layer.data = new_data
        layer.refresh()
        viewer.status = f"Algorithmic Snap: Pixel {np.round(target_coord_data, 1)}"