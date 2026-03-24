import logging
import napari
from collections import deque
from magicgui import magicgui, widgets
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..decorators import require_active_session
from ... import constants

logger = logging.getLogger(__name__)


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

def _get_puncta_layers(gui):
    """Return only puncta Points layers (exclude _IDs and _centroids)."""
    viewer = napari.current_viewer()
    if not viewer:
        return []
    return [
        l for l in viewer.layers
        if isinstance(l, napari.layers.Points)
        and not l.name.endswith("_IDs")
        and not l.name.endswith("_centroids")
    ]

@magicgui(
    call_button="Delete Selected Points",
    points_layer={"label": "Layer to Edit", "choices": _get_puncta_layers, "tooltip": "The puncta points layer to edit."},
    fishing_hook={"label": "Enable Fishing Hook", "value": False, "tooltip": "Click to place puncta. Automatically snaps to the nearest intensity peak."},
    volume_optimization={"label": "Volume Optimization", "value": True, "tooltip": "Refine each placed punctum position to the local intensity maximum."},
    opt_radius={"label": "Opt. Radius (um)", "value": "0.1", "tooltip": "Search radius in microns for volume optimization refinement."}
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
            logger.info("PUNCTA EDIT: Deleted %d selected points on layer '%s'", len(points_layer.selected_data), points_layer.name)
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
        logger.info("PUNCTA EDIT: Deleted spot %d under cursor on layer '%s'", val, layer.name)
        _puncta_undo.push(layer)
        layer.selected_data = {val}; layer.remove_selected()
        viewer.status = f"Deleted spot {val}"

def register_editor_hotkeys(viewer):
    @viewer.bind_key('x', overwrite=True)
    def _delete_hotkey(v): delete_point_under_mouse(v)

_previous_editor_layer = [None]  # mutable container to allow closure mutation

_prev_data_connection = [None]  # (layer, callback) for data event
_prev_point_count = [0]
_skip_data_sync = False  # set True when our own code modifies data

def _on_points_data_changed(event=None):
    """Auto-assign unique puncta_id when points are added via napari's native Add mode."""
    global _skip_data_sync
    if _skip_data_sync:
        return
    layer = _puncta_editor_widget.points_layer.value
    if not layer:
        return
    n = len(layer.data)
    prev = _prev_point_count[0]
    _prev_point_count[0] = n

    if n <= prev:
        return  # points were removed or unchanged, not added

    # Points were added — assign unique IDs to all new points
    n_new = n - prev
    features = layer.features
    if 'puncta_id' not in features.columns:
        features['puncta_id'] = pd.Series(dtype='float')

    max_id = features['puncta_id'].dropna().max()
    next_id = int(max_id) + 1 if pd.notna(max_id) else 0

    for i in range(prev, n):
        idx = features.index[i] if i < len(features) else i
        if i < len(features) and pd.notna(features.loc[idx].get('puncta_id')) and features.loc[idx]['puncta_id'] != features.iloc[prev - 1]['puncta_id'] if prev > 0 else False:
            continue  # already has a unique ID (set by fishing hook)
        features.loc[idx, 'puncta_id'] = next_id
        # Fill other required columns with defaults
        for col, default in [('Nucleus_ID', 0), ('Intensity', np.nan), ('SNR', np.nan)]:
            if col not in features.columns:
                features[col] = default
            elif pd.isna(features.loc[idx].get(col, np.nan)):
                features.loc[idx, col] = default
        next_id += 1

    _skip_data_sync = True
    try:
        layer.features = features
        layer.text = {
            'string': '{puncta_id:.0f}', 'size': layer.text.size,
            'color': 'white', 'translation': np.array([0, 5, 5]),
        }
        layer.refresh()
    finally:
        _skip_data_sync = False

@_puncta_editor_widget.points_layer.changed.connect
def _on_layer_change(new_layer):
    # Clear undo stack when switching layers to prevent cross-layer undo
    _puncta_undo.clear()

    # Disconnect previous data event
    if _prev_data_connection[0] is not None:
        old_layer, old_cb = _prev_data_connection[0]
        try:
            old_layer.events.data.disconnect(old_cb)
        except (RuntimeError, TypeError):
            pass
        _prev_data_connection[0] = None

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
        # Connect data change listener for auto-ID assignment
        _prev_point_count[0] = len(new_layer.data)
        new_layer.events.data.connect(_on_points_data_changed)
        _prev_data_connection[0] = (new_layer, _on_points_data_changed)

        # Sync: select the layer in the viewer layer list and make it visible
        viewer = napari.current_viewer()
        if viewer and new_layer in viewer.layers:
            if not _syncing_puncta_selection:
                new_layer.visible = True
                viewer.layers.selection.active = new_layer


# --- Bidirectional sync: layer list ↔ dropdown ---
_syncing_puncta_selection = False

def _on_viewer_layer_selection_for_puncta(event=None):
    """When a puncta layer is selected in the layer list, update the dropdown."""
    global _syncing_puncta_selection
    if _syncing_puncta_selection:
        return
    from .. import viewer as _viewer_mod
    if getattr(_viewer_mod, '_suppress_custom_controls', False):
        return
    try:
        viewer = napari.current_viewer()
        if not viewer:
            return
        active = viewer.layers.selection.active
        if active is None:
            return
        if not active.name.endswith("_puncta"):
            return
        if not isinstance(active, napari.layers.Points):
            return
        # Check if it's a valid choice in the dropdown
        current = _puncta_editor_widget.points_layer.value
        if current is active:
            return
        _syncing_puncta_selection = True
        try:
            _puncta_editor_widget.reset_choices()
            _puncta_editor_widget.points_layer.value = active
        except (ValueError, AttributeError):
            pass
        finally:
            _syncing_puncta_selection = False
    except Exception:
        pass

def connect_puncta_editor_layer_sync(viewer):
    """Call once from viewer.py to wire up the selection sync."""
    viewer.layers.selection.events.changed.connect(_on_viewer_layer_selection_for_puncta)


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

# --- F key state tracking for fishing hook ---
_f_key_held = False
_f_key_bound = False

def _ensure_f_key_bound():
    """Lazily bind the F key to the viewer for fishing hook activation."""
    global _f_key_bound
    if _f_key_bound:
        return
    viewer = napari.current_viewer()
    if not viewer:
        return
    @viewer.bind_key('f', overwrite=True)
    def _on_f_press(viewer):
        global _f_key_held
        _f_key_held = True
        yield
        _f_key_held = False
    _f_key_bound = True

# --- 2. The Yield Callback (Emulates puncta.py segmentation) ---
def fishing_hook_callback(layer, event):
    _ensure_f_key_bound()
    if not _f_key_held or not _puncta_editor_widget.fishing_hook.value:
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
        global _skip_data_sync
        _puncta_undo.push(layer)
        _skip_data_sync = True
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
        _prev_point_count[0] = len(new_data)  # sync count so data listener doesn't re-process
        _skip_data_sync = False
        layer.refresh()
        logger.info("PUNCTA EDIT: Added point ID %d at %s on layer '%s'", next_id, np.round(target_coord_data, 1), layer.name)
        viewer.status = f"Algorithmic Snap: ID {next_id}, Pixel {np.round(target_coord_data, 1)}"

# --- Undo Button ---
pe_undo_btn = widgets.PushButton(text="Undo", tooltip="Undo the last puncta edit (add or remove).")

def _on_puncta_undo():
    viewer = napari.current_viewer()
    layer = _puncta_editor_widget.points_layer.value
    if not layer:
        viewer.status = "No points layer selected."
        return
    if _puncta_undo.undo(layer):
        logger.info("PUNCTA EDIT: Undo on layer '%s' (%d remaining)", layer.name, len(_puncta_undo))
        viewer.status = f"Undo ({len(_puncta_undo)} remaining)."
    else:
        viewer.status = "Nothing to undo."

pe_undo_btn.clicked.connect(_on_puncta_undo)

# --- UI Wrapper ---
from qtpy.QtWidgets import QFrame
from ..style import COLORS

from qtpy.QtWidgets import QLabel as _QLabel, QWidget as _QWidget

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

def _make_section_header(text):
    label = _QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

def _make_section_desc(text):
    desc = _QLabel(text)
    desc.setWordWrap(True)
    desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
    return desc

def _make_spacer():
    s = _QWidget()
    s.setFixedHeight(20)
    return s

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

# --- "Target Layer" section: points_layer(0) ---
_pe_target_header = _make_section_header("Target Layer")
_pe_target_desc = _make_section_desc("Select the puncta layer to edit.")
_inner.insertWidget(0, _pe_target_header)
_inner.insertWidget(1, _pe_target_desc)
# points_layer(2)

# --- "Fishing Hook" section: fishing_hook(3), volume_opt(4), opt_radius(5) ---
_fishing_hook_header = _make_section_header("Fishing Hook")
_fishing_hook_desc = _make_section_desc(
    "Hold F + Click to place a punctum. The algorithm casts a ray through "
    "the volume along your viewing angle, finds the brightest voxel, then "
    "refines to the local intensity peak. Ideal for accurate placement in 3D."
)
_inner.insertWidget(3, _make_spacer())
_inner.insertWidget(4, _make_divider())
_inner.insertWidget(5, _fishing_hook_header)
_inner.insertWidget(6, _fishing_hook_desc)
# fishing_hook(7), volume_opt(8), opt_radius(9), call_button(10)

# Set fishing hook icon on the checkbox
from pathlib import Path as _Path
from qtpy.QtGui import QIcon as _QIcon
_hook_icon_path = _Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / "fishing_hook.svg"
if _hook_icon_path.exists():
    _fishing_hook_cb = _puncta_editor_widget.fishing_hook.native
    _fishing_hook_cb.setIcon(_QIcon(str(_hook_icon_path)))
    from qtpy.QtCore import QSize as _QSize
    _fishing_hook_cb.setIconSize(_QSize(18, 18))

# --- "Erase" section: call_button stays at the end ---
# Remove call_button from its position and re-add after the erase section
_erase_header = _make_section_header("Erase")
_erase_desc = _make_section_desc("Select points in the viewer and click Delete to remove them.")
_inner.insertWidget(10, _make_spacer())
_inner.insertWidget(11, _make_divider())
_inner.insertWidget(12, _erase_header)
_inner.insertWidget(13, _erase_desc)
# call_button is now at 14 — that's the fishing hook's call button, not erase
# We don't want it here, move it after erase content

# Tighten the inner magicgui form layout
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

# Outer layout: add the whole form as one block
_layout = puncta_editor_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_puncta_editor_widget.native)

# Clear All Points button
_clear_all_btn = widgets.PushButton(text="Clear All Points", tooltip="Remove all points from the current layer.")

def _on_clear_all(_=False):
    layer = _puncta_editor_widget.points_layer.value
    viewer = napari.current_viewer()
    if not layer or not viewer:
        return
    from ..popups import show_yes_no_popup
    if not show_yes_no_popup(
        viewer.window._qt_window,
        "Clear All Points",
        f"Remove all {len(layer.data)} points from '{layer.name}'?\n\nThis cannot be undone.",
    ):
        return
    _puncta_undo.push(layer)
    global _skip_data_sync
    _skip_data_sync = True
    try:
        layer.data = np.empty((0, layer.ndim))
        layer.features = pd.DataFrame(columns=layer.features.columns)
        _prev_point_count[0] = 0
        layer.refresh()
    finally:
        _skip_data_sync = False
    logger.info("PUNCTA EDIT: Cleared all points on layer '%s'", layer.name)
    viewer.status = f"Cleared all points from {layer.name}."

_clear_all_btn.changed.connect(_on_clear_all)
_layout.addWidget(_clear_all_btn.native)

# Undo in its own section
_layout.addWidget(_make_spacer())
_layout.addWidget(_make_divider())
_layout.addWidget(pe_undo_btn.native)
_layout.addStretch(1)