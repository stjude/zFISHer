import napari
from collections import deque
from magicgui import magicgui, widgets
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..decorators import require_active_session
from ... import constants


class _PunctaUndoStack:
    """Stores full snapshots of (data, features) — points data is small."""
    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)

    def push(self, layer):
        self._stack.append((layer.data.copy(), layer.features.copy()))

    def undo(self, layer):
        if not self._stack:
            return False
        data, features = self._stack.pop()
        layer.data = data
        layer.features = features
        return True

    def clear(self):
        self._stack.clear()

    def __len__(self):
        return len(self._stack)


_puncta_undo = _PunctaUndoStack()

@magicgui(
    call_button="Delete Selected Points",
    points_layer={"label": "Layer to Edit"},
    fishing_hook={"label": "Enable Fishing Hook", "value": True},
    volume_optimization={"label": "Volume Optimization", "value": True},
    opt_radius={"label": "Opt. Radius (um)", "value": "0.1"}
)
@require_active_session()
def _puncta_editor_widget(
    points_layer: "napari.layers.Points",
    fishing_hook: bool = True,
    volume_optimization: bool = True,
    opt_radius: str = "0.1"
):
    """Enhanced editor for high-density puncta curation."""
    if points_layer:
        if points_layer.mode == 'select' and len(points_layer.selected_data) > 0:
            _puncta_undo.push(points_layer)
            points_layer.remove_selected()


# --- Mouse & Hotkey Logic ---
def delete_point_under_mouse(viewer):
    layer = _puncta_editor_widget.points_layer.value
    if not layer or not layer.visible: return
    try:
        val = layer.get_value(viewer.cursor.position, view_direction=viewer.camera.view_direction, dims_displayed=list(viewer.dims.displayed), world=True)
    except Exception:
        return
    if val is not None:
        _puncta_undo.push(layer)
        layer.selected_data = {val}; layer.remove_selected()
        viewer.status = f"Deleted spot {val}"

def register_editor_hotkeys(viewer):
    @viewer.bind_key('x', overwrite=True)
    def _delete_hotkey(v): delete_point_under_mouse(v)

_previous_editor_layer = [None]  # mutable container to allow closure mutation

@_puncta_editor_widget.points_layer.changed.connect
def _on_layer_change(new_layer):
    # Remove fishing hook from the *previous* layer so callbacks don't accumulate
    old_layer = _previous_editor_layer[0]
    if old_layer is not None and old_layer is not new_layer:
        try:
            if fishing_hook_callback in old_layer.mouse_drag_callbacks:
                old_layer.mouse_drag_callbacks.remove(fishing_hook_callback)
        except (ValueError, RuntimeError):
            pass
    _previous_editor_layer[0] = new_layer

    if new_layer:
        if fishing_hook_callback not in new_layer.mouse_drag_callbacks:
            new_layer.mouse_drag_callbacks.append(fishing_hook_callback)

# --- 1. Algorithmic Math (Operates strictly on Pixel Arrays) ---
# --- 1. Algorithmic Math (Operates strictly on Pixel Arrays) ---
def calculate_fishing_hook(img_layer, data_pos, viewer, use_optimization=True, radius_um=0.1):
    """Calculates peak intensity using a data coordinate, not a world coordinate."""
    # 1. Get the camera direction (World Space)
    try:
        view_direction = np.array(viewer.camera.view_direction)
        if view_direction is None or np.any(np.isnan(view_direction)):
            return None
    except Exception:
        return None
    
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
        _puncta_undo.push(layer)
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

# --- Undo Button ---
pe_undo_btn = widgets.PushButton(text="Undo")

def _on_puncta_undo():
    viewer = napari.current_viewer()
    layer = _puncta_editor_widget.points_layer.value
    if not layer:
        viewer.status = "No points layer selected."
        return
    if _puncta_undo.undo(layer):
        viewer.status = f"Undo ({len(_puncta_undo)} remaining)."
    else:
        viewer.status = "Nothing to undo."

pe_undo_btn.clicked.connect(_on_puncta_undo)

# --- UI Wrapper ---
from qtpy.QtWidgets import QFrame
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

class _PunctaEditorWidgetContainer(widgets.Container):
    """Wrapper that delegates reset_choices to the inner magicgui widget."""
    def reset_choices(self):
        _puncta_editor_widget.reset_choices()

puncta_editor_widget = _PunctaEditorWidgetContainer(labels=False)
puncta_editor_widget._puncta_editor_widget = _puncta_editor_widget
header = widgets.Label(value="Puncta Editor")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Advanced editing of puncta.</i>")
info.native.setObjectName("widgetInfo")

# Insert section headers/dividers into the magicgui form's internal layout
# Widget order: points_layer(0), fishing_hook(1), volume_opt(2), opt_radius(3), call_button(4)
_inner = _puncta_editor_widget.native.layout()
_fishing_hook_header = widgets.Label(value="<b>Fishing Hook:</b>")
from qtpy.QtWidgets import QLabel
_fishing_hook_desc = QLabel(
    "<i>Shift+Click to place a punctum. The algorithm casts a ray through "
    "the volume along your viewing angle, finds the brightest voxel, then "
    "refines to the local intensity peak. Ideal for accurate placement in 3D.</i>"
)
_fishing_hook_desc.setWordWrap(True)
_erase_header = widgets.Label(value="<b>Erase:</b>")
# Insert before fishing_hook (index 1) — pushes everything after it down
_inner.insertWidget(1, _make_divider())
_inner.insertWidget(2, _fishing_hook_header.native)
_inner.insertWidget(3, _fishing_hook_desc)
# call_button is now at index 7 (was 4, +3 from divider/header/desc inserts)
_inner.insertWidget(7, _make_divider())
_inner.insertWidget(8, _erase_header.native)

# Outer layout: add the whole form as one block (preserves tight magicgui spacing)
_layout = puncta_editor_widget.native.layout()
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_puncta_editor_widget.native)
_layout.addWidget(pe_undo_btn.native)