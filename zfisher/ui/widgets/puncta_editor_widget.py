import napari
from magicgui import magicgui, widgets
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pathlib import Path

from ...core import session
from ..decorators import require_active_session
from ... import constants

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
def _puncta_editor_widget(
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

@magicgui(call_button="Save Puncta Changes")
@require_active_session()
def _save_widget(layer: "napari.layers.Points"):
    if not layer:
        print("No points layer selected to save.")
        return

    viewer = napari.current_viewer()
    out_dir = session.get_data("output_dir")
    if not out_dir:
        print("No output directory set. Cannot save.")
        return
    
    try:
        # Combine coordinates and features for saving
        coords_df = pd.DataFrame(layer.data, columns=['Z', 'Y', 'X'])
        full_df_to_save = pd.concat([layer.features.reset_index(drop=True), coords_df.reset_index(drop=True)], axis=1)

        # Define path and save
        reports_dir = Path(out_dir) / constants.REPORTS_DIR
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / f"{layer.name}.csv"
        
        full_df_to_save.to_csv(csv_path, index=False)
        
        # Update session file to point to this new CSV
        session.set_processed_file(layer.name, str(csv_path), layer_type='points', metadata={'subtype': 'puncta_csv'})
        
        viewer.status = f"Saved {len(full_df_to_save)} puncta for '{layer.name}' to {csv_path.name}"
        print(f"Saved puncta changes for layer '{layer.name}'")

    except Exception as e:
        print(f"Error saving puncta layer '{layer.name}': {e}")
        viewer.status = f"Error saving '{layer.name}': {e}"

# --- UI Connections ---
@_puncta_editor_widget.point_size.changed.connect
def _on_size_change(value):
    if _puncta_editor_widget.points_layer.value: _puncta_editor_widget.points_layer.value.size = value

@_puncta_editor_widget.show_all_z.changed.connect
def _on_projection_change(value):
    layer = _puncta_editor_widget.points_layer.value
    if layer:
        layer.out_of_slice = value
        layer.projection_mode = 'all' if value else 'none'

# --- Toolbar Construction ---
pe_lbl = widgets.Label(value="<b>Editing Hub:</b>")
pe_container = widgets.Container(layout="horizontal", labels=False)
pe_add_chk, pe_select_chk, pe_pan_btn = widgets.CheckBox(text="Add (A)"), widgets.CheckBox(text="Select (S)"), widgets.PushButton(text="Pan/Zoom (Z)")
pe_container.extend([pe_add_chk, pe_select_chk, pe_pan_btn])
_puncta_editor_widget.insert(0, pe_lbl)
_puncta_editor_widget.insert(1, pe_container)

# --- Mouse & Hotkey Logic ---
def delete_point_under_mouse(viewer):
    layer = _puncta_editor_widget.points_layer.value
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

@_puncta_editor_widget.points_layer.changed.connect
def _on_layer_change(new_layer):
    if new_layer:
        # Prevent crash if layer has no points, causing mean to be NaN
        mean_val = np.mean(new_layer.size)
        if np.isnan(mean_val):
            size = 3 # Default size for empty layers
        else:
            size = int(mean_val)

        _puncta_editor_widget.point_size.value = max(1, size)
        
        if fishing_hook_callback not in new_layer.mouse_drag_callbacks:
            new_layer.mouse_drag_callbacks.append(fishing_hook_callback)
        oos_val = getattr(new_layer, 'out_of_slice_dist', getattr(new_layer, 'out_of_slice', True))
        _puncta_editor_widget.show_all_z.value = bool(oos_val)

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
    if 'Shift' not in event.modifiers or not _puncta_editor_widget.fishing_hook.value:
        return

    viewer = napari.current_viewer()
    img_layer = next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and l.visible), None)
    if not img_layer or not layer: return
    
    if not np.allclose(layer.scale, img_layer.scale) or not np.allclose(layer.translate, img_layer.translate):
        layer.scale = img_layer.scale
        layer.translate = img_layer.translate

    cursor_pos_data = img_layer.world_to_data(viewer.cursor.position)

    target_coord_data = calculate_fishing_hook(
        img_layer, 
        cursor_pos_data,
        viewer,
        use_optimization=_puncta_editor_widget.volume_optimization.value,
        radius_um=float(_puncta_editor_widget.opt_radius.value)
    )

    yield 
    while event.type == 'mouse_move':
        yield
        
    if target_coord_data is not None:
        new_data = layer.data.copy()
        
        search_data = new_data[:-1] if layer.mode == 'add' and len(new_data) > 0 else new_data
        
        if len(search_data) > 0:
            dist, _ = cKDTree(search_data).query(target_coord_data, k=1)
            if dist < 1.5:
                if layer.mode == 'add' and len(new_data) > 0:
                    layer.data = new_data[:-1] 
                viewer.status = "Point already exists here."
                return
        
        # --- ID Management ---
        current_features = layer.features
        if 'puncta_id' not in current_features:
            current_features['puncta_id'] = pd.Series(dtype='int')

        # Determine next ID, ignoring NaNs
        if not current_features['puncta_id'].empty:
            max_id = current_features['puncta_id'].dropna().max()
            if pd.notna(max_id):
                next_id = int(max_id) + 1
            else: # No valid IDs exist
                next_id = 0
        else:
            next_id = 0

        new_properties = {
            'puncta_id': next_id,
            'Nucleus_ID': 0,
            'Intensity': np.nan,
            'SNR': np.nan,
        }

        # Ensure all required columns exist
        for col, default_val in new_properties.items():
            if col not in current_features:
                current_features[col] = default_val

        if layer.mode == 'add' and len(new_data) > 0:
            new_data[-1] = target_coord_data
            
            # Update the properties for the just-added point
            new_point_index = current_features.index[-1]
            for col, value in new_properties.items():
                current_features.loc[new_point_index, col] = value

        else: 
            new_data = np.vstack((new_data, target_coord_data))
            new_row = pd.DataFrame([new_properties])
            current_features = pd.concat([current_features, new_row], ignore_index=True)
            
        layer.data = new_data
        layer.features = current_features
        layer.refresh()
        viewer.status = f"Algorithmic Snap: ID {next_id}, Pixel {np.round(target_coord_data, 1)}"

# --- UI Wrapper ---
puncta_editor_widget = widgets.Container(labels=False)
header = widgets.Label(value="Puncta Editor")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Advanced editing of puncta.</i>")
puncta_editor_widget.extend([header, info, _puncta_editor_widget, _save_widget])

# Link the layer dropdowns
_save_widget.layer.bind(_puncta_editor_widget.points_layer)