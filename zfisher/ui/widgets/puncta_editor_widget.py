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
    """Stores undo entries — either data snapshots or deleted layer snapshots."""
    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)

    def push(self, layer):
        """Push a data snapshot for the given layer."""
        self._stack.append(('data', layer.name, layer.data.copy(), layer.features.copy()))

    def push_layer_delete(self, layer):
        """Push a full layer snapshot so deletion can be undone."""
        self._stack.append(('layer', layer.name, layer.data.copy(), layer.features.copy(), {
            'scale': tuple(layer.scale),
            'translate': tuple(layer.translate),
            'size': layer.size if hasattr(layer, 'size') else 10,
            'face_color': layer.face_color if hasattr(layer, 'face_color') else 'yellow',
            'text': {'string': '{puncta_id:.0f}', 'size': layer.text.size, 'color': 'white',
                     'translation': list(layer.text.translation)} if layer.text.visible else None,
            'visible': layer.visible,
            'blending': str(layer.blending),
        }))

    def undo(self, current_layer=None):
        """Undo the last action. Returns ('data', layer) or ('layer', new_layer) or None."""
        if not self._stack:
            return None
        entry = self._stack.pop()
        if entry[0] == 'data':
            _, name, data, features = entry
            target = None
            if current_layer and current_layer.name == name:
                target = current_layer
            else:
                viewer = napari.current_viewer()
                if viewer and name in viewer.layers:
                    target = viewer.layers[name]
            if target:
                # Workaround for napari text rendering during undo:
                # When layer.data is resized, napari calls text.apply() on the
                # feature table before we can set features. The resized table has
                # None values, and the format string '{puncta_id:.0f}' crashes on None.
                # Fix: temporarily set format to '' (constant), swap data+features,
                # then restore the format string once features are valid.
                was_visible = target.visible
                target.visible = False
                target._Points__indices_view = np.empty(0, int)
                # Save and replace text encoding with safe constant
                saved_text_format = None
                try:
                    saved_text_format = target.text.string.format
                    target.text.string.format = ''
                except AttributeError:
                    pass
                target.data = data
                target.features = features
                # Restore the format string now that features are valid
                if saved_text_format is not None:
                    target.text.string.format = saved_text_format
                    target.text.string._apply(target.features)
                target.visible = was_visible
                return ('data', target)
            return None
        elif entry[0] == 'layer':
            _, name, data, features, props = entry
            viewer = napari.current_viewer()
            if not viewer:
                return None
            text_props = props.get('text')
            layer = viewer.add_points(
                data, name=name,
                features=features,
                scale=props['scale'],
                translate=props['translate'],
                size=props['size'],
                face_color=props['face_color'],
                text=text_props,
                visible=props['visible'],
                blending=props['blending'],
            )
            return ('layer', layer)
        return None

    def clear(self):
        self._stack.clear()

    def __len__(self):
        return len(self._stack)


_puncta_undo = _PunctaUndoStack()


def _compute_quality_at_point(points_layer, z, y, x, radius=2):
    """Find the source image layer for a points layer and compute (intensity, SNR)
    at the given (z, y, x) voxel. Returns (np.nan, np.nan) if the image can't be
    resolved or the point is out of bounds.
    """
    viewer = napari.current_viewer()
    if viewer is None:
        return np.nan, np.nan
    img_name = points_layer.name
    if img_name.endswith(constants.PUNCTA_SUFFIX):
        img_name = img_name[: -len(constants.PUNCTA_SUFFIX)]
    if img_name not in viewer.layers:
        return np.nan, np.nan
    img = viewer.layers[img_name].data
    if img.ndim != 3:
        return np.nan, np.nan
    z, y, x = int(round(z)), int(round(y)), int(round(x))
    if not (0 <= z < img.shape[0] and 0 <= y < img.shape[1] and 0 <= x < img.shape[2]):
        return np.nan, np.nan
    y0, y1 = max(0, y - radius), min(img.shape[1], y + radius + 1)
    x0, x1 = max(0, x - radius), min(img.shape[2], x + radius + 1)
    crop = img[z, y0:y1, x0:x1]
    peak = float(img[z, y, x])
    bg = float(np.median(crop)) if crop.size > 0 else 1.0
    snr = peak / bg if bg > 0 else 0.0
    return peak, snr


def reset_puncta_editor_state():
    """Clear all module-level state. Called on session reset."""
    global _skip_data_sync, _f_key_held, _f_key_bound
    _puncta_undo.clear()
    _previous_editor_layer[0] = None
    if _prev_data_connection[0] is not None:
        try:
            old_layer, old_cb = _prev_data_connection[0]
            old_layer.events.data.disconnect(old_cb)
        except Exception:
            pass
    _prev_data_connection[0] = None
    _prev_snapshot[0] = None
    _skip_data_sync = False
    _f_key_held = False
    _f_key_bound = False


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
    call_button="Delete Selected Puncta",
    points_layer={"label": "Layer to Edit", "choices": _get_puncta_layers, "tooltip": "The puncta layer to edit."},
    fishing_hook={"label": "Enable Fishing Hook", "value": False, "tooltip": "Enable to place puncta with automatic snapping to intensity peaks. Hold F + Click to place. When disabled, use standard add mode."},
    volume_optimization={"label": "Volume Optimization", "value": True, "tooltip": "Refine punctum position to the local intensity maximum within the specified search radius."},
    opt_radius={"label": "Opt. Radius (um)", "value": "0.1", "tooltip": "Search radius (in microns) to find the nearest intensity peak when volume optimization is enabled."}
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
_prev_snapshot = [None]  # (data, features) snapshot before native add
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
        # Points removed or unchanged — save snapshot for next undo
        _prev_snapshot[0] = (layer.data.copy(), layer.features.copy())
        return  # points were removed or unchanged, not added

    # Push pre-change snapshot to undo stack (if available)
    if _prev_snapshot[0] is not None:
        _puncta_undo._stack.append(('data', layer.name, _prev_snapshot[0][0], _prev_snapshot[0][1]))
        _prev_snapshot[0] = None

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
        # Compute intensity + SNR at the point from the source image layer
        z, y, x = layer.data[i]
        peak, snr = _compute_quality_at_point(layer, z, y, x)
        for col, default in [('Nucleus_ID', 0), ('Intensity', peak), ('SNR', snr), ('Source', 'manual')]:
            if col not in features.columns:
                features[col] = default
            else:
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
        _prev_snapshot[0] = (new_layer.data.copy(), new_layer.features.copy())
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
    """Place a punctum by casting a ray through the volume along the camera direction,
    finding the brightest voxel along the ray, then optionally refining to the local
    intensity peak within radius_um. Ideal for accurate 3D placement from any view angle."""
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
    channel_name = layer.name.replace(constants.PUNCTA_SUFFIX, "")
    img_layer = next(
        (l for l in viewer.layers if isinstance(l, napari.layers.Image) and l.name == channel_name),
        next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and l.visible), None)
    )
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
                    layer._Points__indices_view = np.empty(0, int)
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

        z, y, x = target_coord_data
        _peak, _snr = _compute_quality_at_point(layer, z, y, x)
        new_properties = {
            'puncta_id': next_id,
            'Nucleus_ID': 0,
            'Intensity': _peak,
            'SNR': _snr,
            'Source': 'manual',
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
            
        layer._Points__indices_view = np.empty(0, int)
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
    if not viewer:
        return
    result = _puncta_undo.undo(layer)
    if result is None:
        viewer.status = "Nothing to undo."
        return
    kind, restored_layer = result
    if kind == 'data':
        logger.info("PUNCTA EDIT: Undo on layer '%s' (%d remaining)", restored_layer.name, len(_puncta_undo))
        viewer.status = f"Undo ({len(_puncta_undo)} remaining)."
    elif kind == 'layer':
        logger.info("PUNCTA EDIT: Undo layer delete — restored '%s' (%d remaining)", restored_layer.name, len(_puncta_undo))
        _puncta_editor_widget.reset_choices()
        _puncta_editor_widget.points_layer.value = restored_layer
        viewer.status = f"Restored layer {restored_layer.name} ({len(_puncta_undo)} remaining)."

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
        # Filter out _IDs and _centroids layers from the dropdown —
        # magicgui ignores the choices callback for napari layer types
        # and auto-populates from the viewer, so we filter post-hoc.
        w = _puncta_editor_widget.points_layer
        choices = list(w.choices)
        filtered = [c for c in choices if not c.name.endswith("_IDs") and not c.name.endswith("_centroids")]
        if len(filtered) != len(choices):
            w.choices = filtered

puncta_editor_widget = _PunctaEditorWidgetContainer(labels=False)
puncta_editor_widget._puncta_editor_widget = _puncta_editor_widget
header = widgets.Label(value="Puncta Editor")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Manually add, delete, and edit puncta positions.</i>")
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

# --- "Tools" section (between Target Layer and Fishing Hook) ---
from pathlib import Path as _Path
from qtpy.QtGui import QIcon as _QIcon
from qtpy.QtCore import QSize as _QSize
from qtpy.QtWidgets import QPushButton as _QPushButton, QHBoxLayout as _QHBoxLayout, QCheckBox as _QCheckBox

_tools_header = _make_section_header("Tools")
_tools_desc = _make_section_desc("Select a tool. Only one mode is active at a time.")

_napari_icon_dir = _Path(napari.__file__).parent / "resources" / "icons"
_hook_icon_path = _Path(__file__).resolve().parent.parent.parent / "resources" / "icons" / "fishing_hook.svg"

def _white_icon(svg_path):
    """Load an SVG and recolor it white by injecting a fill attribute."""
    from qtpy.QtSvg import QSvgRenderer
    from qtpy.QtGui import QPixmap, QPainter
    from qtpy.QtCore import QByteArray, Qt

    svg_text = _Path(svg_path).read_text(encoding='utf-8')
    # Only inject fill="white" if the SVG doesn't already specify fill on the root tag
    if 'fill="white"' not in svg_text and 'fill="none"' not in svg_text:
        svg_text = svg_text.replace('<svg ', '<svg fill="white" ', 1)
    elif 'fill="none"' in svg_text:
        # stroke-based SVG — ensure stroke is white
        if 'stroke="white"' not in svg_text:
            svg_text = svg_text.replace('<svg ', '<svg stroke="white" ', 1)
    renderer = QSvgRenderer(QByteArray(svg_text.encode('utf-8')))
    pixmap = QPixmap(28, 28)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return _QIcon(pixmap)

def _make_tool_btn(icon_name, tooltip, checkable=True):
    """Create a full-width gray button matching mask editor style."""
    btn = _QPushButton()
    icon_path = _napari_icon_dir / f"{icon_name}.svg"
    if icon_path.exists():
        btn.setIcon(_white_icon(icon_path))
    elif icon_name == "fishing_hook" and _hook_icon_path.exists():
        btn.setIcon(_white_icon(_hook_icon_path))
    btn.setIconSize(_QSize(20, 20))
    btn.setToolTip(tooltip)
    btn.setCheckable(checkable)
    return btn

_btn_delete = _make_tool_btn("delete_shape", "Delete Selected Points", checkable=False)
_btn_add = _make_tool_btn("add", "Add Point")
_btn_select = _make_tool_btn("select", "Select Points")
_btn_move = _make_tool_btn("pan_arrows", "Move Camera (pan/zoom)")
_btn_hook = _make_tool_btn("fishing_hook", "Fishing Hook (F+Click)")

# --- Tool sync logic ---
# Mode buttons (move/add/select) are mutually exclusive.
# Fishing hook is INDEPENDENT — it's a toggle that works alongside any mode.
_syncing_tools = False

def _set_mode(mode):
    """Set the napari layer mode and sync mode buttons. Does NOT touch fishing hook."""
    global _syncing_tools
    if _syncing_tools:
        return
    _syncing_tools = True
    try:
        layer = _puncta_editor_widget.points_layer.value
        _btn_move.setChecked(mode == 'pan_zoom')
        _btn_add.setChecked(mode == 'add')
        _btn_select.setChecked(mode == 'select')
        if layer and mode in ('pan_zoom', 'add', 'select'):
            layer.mode = mode
    finally:
        _syncing_tools = False

def _toggle_fishing_hook(enabled):
    """Toggle fishing hook independently. Syncs checkbox and duplicate button."""
    global _syncing_tools
    if _syncing_tools:
        return
    _syncing_tools = True
    try:
        _puncta_editor_widget.fishing_hook.value = enabled
        if hasattr(_toggle_fishing_hook, '_dup_btn'):
            _toggle_fishing_hook._dup_btn.setChecked(enabled)
    finally:
        _syncing_tools = False

def _on_tool_delete(_=False):
    layer = _puncta_editor_widget.points_layer.value
    if not layer:
        return
    if len(layer.selected_data) == 0:
        return
    _puncta_undo.push(layer)
    logger.info("PUNCTA EDIT: Deleted %d selected points on layer '%s'", len(layer.selected_data), layer.name)
    layer.remove_selected()

_btn_delete.clicked.connect(_on_tool_delete)
_btn_move.clicked.connect(lambda: _set_mode('pan_zoom'))
_btn_add.clicked.connect(lambda: _set_mode('add'))
_btn_select.clicked.connect(lambda: _set_mode('select'))
_btn_hook.clicked.connect(lambda checked: _toggle_fishing_hook(checked))

# Sync from layer mode changes (e.g. layer controls)
def _sync_tools_from_layer_mode(event=None):
    global _syncing_tools
    if _syncing_tools:
        return
    layer = _puncta_editor_widget.points_layer.value
    if not layer:
        return
    mode = layer.mode
    # Don't override if fishing hook is active and mode is 'add'
    if _puncta_editor_widget.fishing_hook.value and mode == 'add':
        return
    _syncing_tools = True
    try:
        _btn_move.setChecked(mode == 'pan_zoom')
        _btn_add.setChecked(mode == 'add')
        _btn_select.setChecked(mode == 'select')
        _btn_hook.setChecked(False)
        if hasattr(_toggle_fishing_hook, '_dup_btn'):
            _toggle_fishing_hook._dup_btn.setChecked(False)
    finally:
        _syncing_tools = False

# Connect mode sync when layer changes
_prev_mode_layer = [None]

def _connect_mode_sync(new_layer):
    old = _prev_mode_layer[0]
    if old is not None:
        try:
            old.events.mode.disconnect(_sync_tools_from_layer_mode)
        except (RuntimeError, TypeError):
            pass
    if new_layer:
        new_layer.events.mode.connect(_sync_tools_from_layer_mode)
    _prev_mode_layer[0] = new_layer

_puncta_editor_widget.points_layer.changed.connect(_connect_mode_sync)

# Sync fishing hook checkbox → duplicate button (independent of mode)
def _sync_hook_checkbox(val):
    _toggle_fishing_hook(val)

_puncta_editor_widget.fishing_hook.changed.connect(_sync_hook_checkbox)

# Insert into inner layout
_idx = 3  # after points_layer(2)
_inner.insertWidget(_idx, _make_spacer()); _idx += 1
_inner.insertWidget(_idx, _make_divider()); _idx += 1
_inner.insertWidget(_idx, _tools_header); _idx += 1
_inner.insertWidget(_idx, _tools_desc); _idx += 1
# Tool buttons as a row
_tools_row = _QWidget()
_tools_row_layout = _QHBoxLayout(_tools_row)
_tools_row_layout.setContentsMargins(0, 4, 0, 4)
_tools_row_layout.setSpacing(4)
for _b in [_btn_delete, _btn_add, _btn_select, _btn_move]:
    _tools_row_layout.addWidget(_b)
_inner.insertWidget(_idx, _tools_row); _idx += 1
# Now fishing_hook is at _idx, volume_opt at _idx+1, etc.

# --- "Fishing Hook" section ---
_fishing_hook_header = _make_section_header("Fishing Hook")
_fishing_hook_desc = _make_section_desc(
    "Hold F + Click to place a punctum. The algorithm casts a ray through "
    "the volume along your viewing angle, finds the brightest voxel, then "
    "refines to the local intensity peak. Ideal for accurate placement in 3D."
)
_fh_start = _idx  # fishing_hook widget is here
_inner.insertWidget(_fh_start, _make_spacer()); _fh_start += 1
_inner.insertWidget(_fh_start, _make_divider()); _fh_start += 1
_inner.insertWidget(_fh_start, _fishing_hook_header); _fh_start += 1
_inner.insertWidget(_fh_start, _fishing_hook_desc); _fh_start += 1

# Duplicate fishing hook button in this section
_btn_hook_dup = _make_tool_btn("fishing_hook", "Fishing Hook (F+Click)")
if _hook_icon_path.exists():
    _btn_hook_dup.setIcon(_white_icon(_hook_icon_path))
_btn_hook_dup.clicked.connect(lambda checked: _toggle_fishing_hook(checked))
_toggle_fishing_hook._dup_btn = _btn_hook_dup
_inner.insertWidget(_fh_start, _btn_hook_dup); _fh_start += 1

# Status label for fishing hook state
_hook_status_label = _QLabel("<i style='color: white;'>Fishing Hook disabled</i>")
_hook_status_label.setStyleSheet("background: transparent; margin: 4px 2px;")
_inner.insertWidget(_fh_start, _hook_status_label); _fh_start += 1

def _update_hook_status(enabled):
    if enabled:
        _hook_status_label.setText("<i style='color: #4a8;'>Fishing Hook enabled</i>")
    else:
        _hook_status_label.setText("<i style='color: white;'>Fishing Hook disabled</i>")

_puncta_editor_widget.fishing_hook.changed.connect(_update_hook_status)

# Hide the checkbox — the button + status label replace it
_puncta_editor_widget.fishing_hook.native.setVisible(False)
# fishing_hook(hidden), volume_opt, opt_radius, call_button follow

# --- "Erase" section ---
# Find where call_button ended up and insert erase section before it
_erase_idx = _inner.count() - 1  # call_button is last in the magicgui form
_erase_header = _make_section_header("Erase")
_erase_desc = _make_section_desc("Select points in the viewer and click Delete to remove them.")
_inner.insertWidget(_erase_idx, _make_spacer()); _erase_idx += 1
_inner.insertWidget(_erase_idx, _make_divider()); _erase_idx += 1
_inner.insertWidget(_erase_idx, _erase_header); _erase_idx += 1
_inner.insertWidget(_erase_idx, _erase_desc); _erase_idx += 1

# Hide the magicgui call_button (Delete Selected Points) —
# the Tools section X button handles deletion instead.
_call_btn = _inner.itemAt(_inner.count() - 1).widget()
if _call_btn:
    _call_btn.setVisible(False)

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
_clear_all_btn = widgets.PushButton(text="Clear All Points in Layer", tooltip="Remove all points from the current layer.")

def _on_clear_all(_=False):
    layer = _puncta_editor_widget.points_layer.value
    viewer = napari.current_viewer()
    if not layer or not viewer:
        return
    if len(layer.data) == 0:
        viewer.status = "No points to clear."
        return
    _puncta_undo.push(layer)
    global _skip_data_sync
    _skip_data_sync = True
    try:
        layer._Points__indices_view = np.empty(0, int)
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

# Delete Layer button
_delete_layer_btn = widgets.PushButton(text="Delete Layer", tooltip="Remove the current puncta layer from the viewer.")

def _on_delete_layer(_=False):
    layer = _puncta_editor_widget.points_layer.value
    viewer = napari.current_viewer()
    if not layer or not viewer:
        return
    from ..popups import show_yes_no_popup
    if not show_yes_no_popup(
        viewer.window._qt_window,
        "Delete Layer",
        f"Delete '{layer.name}' and all its points?\n\nYou can undo this action.",
    ):
        return
    _puncta_undo.push_layer_delete(layer)
    layer_name = layer.name
    viewer.layers.remove(layer)
    logger.info("PUNCTA EDIT: Deleted layer '%s'", layer_name)
    viewer.status = f"Deleted layer {layer_name}."
    _puncta_editor_widget.reset_choices()

_delete_layer_btn.changed.connect(_on_delete_layer)
_layout.addWidget(_delete_layer_btn.native)

# Undo in its own section
_layout.addWidget(_make_spacer())
_layout.addWidget(_make_divider())
_layout.addWidget(pe_undo_btn.native)
_layout.addStretch(1)