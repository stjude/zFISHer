import napari
import numpy as np
from collections import deque
from magicgui import magicgui, widgets
import tifffile
from pathlib import Path
from qtpy.QtCore import QTimer

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session
from ...core import segmentation
from ... import constants


class _MaskUndoStack:
    """Stores diffs (changed indices + old values) to keep memory usage low."""
    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)
        self._pre_edit = None

    def begin(self, data):
        """Snapshot the current data before an edit."""
        self._pre_edit = data.copy()

    def end(self, data):
        """Compute diff against the snapshot and store it."""
        if self._pre_edit is None:
            return
        diff_mask = self._pre_edit != data
        if np.any(diff_mask):
            indices = np.where(diff_mask)
            self._stack.append((indices, self._pre_edit[indices]))
        self._pre_edit = None

    def undo(self, data):
        """Apply the last stored diff in reverse. Returns True if undo was performed."""
        if not self._stack:
            return False
        indices, old_values = self._stack.pop()
        data[indices] = old_values
        return True

    def clear(self):
        self._stack.clear()
        self._pre_edit = None

    def __len__(self):
        return len(self._stack)


_mask_undo = _MaskUndoStack()

class MaskHighlighter:
    """Highlights the hovered nucleus by setting its color to red via the Labels layer color dict."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.last_layer = None
        self.last_id = None
        self._original_color = None
        self.active = False

    def enable(self):
        if not self.active:
            self.viewer.mouse_move_callbacks.append(self.on_mouse_move)
            self.active = True

    def disable(self):
        if self.active:
            if self.on_mouse_move in self.viewer.mouse_move_callbacks:
                self.viewer.mouse_move_callbacks.remove(self.on_mouse_move)
            self.reset_highlight()
            self.active = False

    def reset_highlight(self):
        """Restores the original color of the last highlighted label."""
        if self.last_layer is not None and self.last_id is not None:
            if self._original_color is not None:
                self.last_layer.color = {self.last_id: self._original_color}
            self._original_color = None
        self.last_layer = None
        self.last_id = None

    def perform_highlight(self, layer, label_id):
        """Sets the hovered label to opaque red."""
        # Restore previous highlight first
        if self.last_id is not None and self.last_id != label_id and self._original_color is not None:
            layer.color = {self.last_id: self._original_color}

        # Store original color before overriding
        self._original_color = layer.get_color(label_id).copy()
        layer.color = {label_id: np.array([1.0, 0.0, 0.0, 1.0])}

        self.last_layer = layer
        self.last_id = label_id
        self.viewer.status = f"Hovering Nucleus ID: {label_id} (Press 'C' to delete)"

    def on_mouse_move(self, viewer, event):
        if not self.active:
            return

        layer = _mask_editor_widget.mask_layer.value
        if not isinstance(layer, napari.layers.Labels):
            if self.last_id is not None:
                self.reset_highlight()
            return

        id_under_cursor = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=list(event.dims_displayed),
            world=True
        )

        if id_under_cursor == self.last_id:
            return  # still over same nucleus, nothing to do

        if id_under_cursor is None or id_under_cursor == 0:
            if self.last_id is not None:
                self.reset_highlight()
        else:
            self.perform_highlight(layer, id_under_cursor)

# Global instance
_highlighter = None

@magicgui(
    call_button="Merge IDs",
    mask_layer={"label": "Layer to Edit"},
    source_id={"label": "Source ID"},
    target_id={"label": "Target ID"}
)
@require_active_session("Please start or load a session before editing masks.")
def _mask_editor_widget(
    mask_layer: "napari.layers.Labels",
    source_id: int = 0,
    target_id: int = 0
):
    """Merges two labels in the selected mask layer."""
    viewer = napari.current_viewer()

    if mask_layer is None:
        viewer.status = "No mask layer selected."
        return

    if source_id == target_id:
        viewer.status = "Source and Target IDs must be different."
        return
        
    count = np.sum(mask_layer.data == source_id)
    
    if count == 0:
        viewer.status = f"ID {source_id} not found."
        return
        
    _mask_undo.begin(mask_layer.data)
    mask_layer.data = segmentation.merge_labels(mask_layer.data, source_id, target_id)
    _mask_undo.end(mask_layer.data)

    viewer.status = f"Merged ID {source_id} into {target_id} ({count} pixels)."

def _delete_label_inplace(layer, label_id):
    """Delete a label in-place and refresh without triggering a full data reassignment."""
    _mask_undo.begin(layer.data)
    layer.data[layer.data == label_id] = 0
    _mask_undo.end(layer.data)
    layer.refresh()
    # Remove the deleted ID from the IDs layer directly (avoids recomputing all centroids)
    _remove_id_from_points_layer(layer, label_id)
    # Debounce save to disk
    _schedule_save(layer)

def _remove_id_from_points_layer(mask_layer, deleted_id):
    """Remove a single ID from the IDs points layer without recomputing centroids."""
    viewer = napari.current_viewer()
    if not viewer:
        return
    ids_name = f"{mask_layer.name}_IDs"
    if ids_name not in viewer.layers:
        return
    pts_layer = viewer.layers[ids_name]
    labels = pts_layer.properties.get('label', np.empty(0))
    if len(labels) == 0:
        return
    keep = labels != deleted_id
    pts_layer.data = pts_layer.data[keep]
    pts_layer.properties = {'label': labels[keep]}

_save_timer = QTimer()
_save_timer.setSingleShot(True)
_save_pending_layer = None

def _schedule_save(layer):
    """Debounced save — writes mask to disk 500ms after the last edit."""
    global _save_pending_layer
    _save_pending_layer = layer
    _save_timer.start(500)

def _do_save():
    global _save_pending_layer
    layer = _save_pending_layer
    if layer is None:
        return
    out_dir = session.get_data("output_dir")
    if out_dir and layer.name:
        seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
        seg_dir.mkdir(exist_ok=True, parents=True)
        mask_path = seg_dir / f"{layer.name}.tif"
        tifffile.imwrite(mask_path, layer.data)
        session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
    _save_pending_layer = None

_save_timer.timeout.connect(_do_save)

def delete_mask_under_mouse(viewer):
    """Deletes the mask label currently under the mouse cursor."""
    global _highlighter

    # 1. If hover-edit is active, use its state for accuracy.
    if _highlighter and _highlighter.active and _highlighter.last_id is not None:
        layer = _highlighter.last_layer
        id_to_delete = _highlighter.last_id

        if layer and id_to_delete:
            _highlighter.reset_highlight()
            _delete_label_inplace(layer, id_to_delete)
            viewer.status = f"Deleted Nucleus ID {id_to_delete}"
            return

    # 2. Fallback: Calculate value under cursor manually
    layer = viewer.layers.selection.active
    if isinstance(layer, napari.layers.Labels):
        val = layer.get_value(
            viewer.cursor.position,
            view_direction=viewer.camera.view_direction,
            dims_displayed=list(viewer.dims.displayed),
            world=True
        )
        if val is not None and val > 0:
            _delete_label_inplace(layer, val)
            viewer.status = f"Deleted Nucleus ID {val}"

_divider_style = "<hr style='border: 1px solid #555; margin: 4px 0;'>"

# --- Merge Section ---
merge_label = widgets.Label(value="<b>Merge Nuclei</b>")
delete_btn = widgets.PushButton(text="Delete Source ID")

# --- Paint Section ---
paint_label = widgets.Label(value="<b>Paint New Mask</b>")
paint_chk = widgets.CheckBox(text="Paint (New ID)")
extrude_btn = widgets.PushButton(text="Extrude ID (Fill Z)")

# --- Erase Section ---
erase_label = widgets.Label(value="<b>Erase</b>")
erase_chk = widgets.CheckBox(text="Erase")
erase_radius_slider = widgets.Slider(label="Radius", min=1, max=50, value=5)
erase_depth_slider = widgets.Slider(label="Depth (Z)", min=1, max=20, value=1)
erase_slider_container = widgets.Container(layout="vertical", labels=True)
erase_slider_container.extend([erase_radius_slider, erase_depth_slider])
hover_chk = widgets.CheckBox(text="Hover Edit Mode (Red + 'C' to Del)")

# --- Utilities ---
undo_btn = widgets.PushButton(text="Undo")
refresh_ids_btn = widgets.PushButton(text="Show/Refresh IDs")

# Layout with dividers
_mask_editor_widget.append(widgets.Label(value=_divider_style))
_mask_editor_widget.append(merge_label)
_mask_editor_widget.append(delete_btn)

_mask_editor_widget.append(widgets.Label(value=_divider_style))
_mask_editor_widget.append(paint_label)
_mask_editor_widget.append(paint_chk)
_mask_editor_widget.append(extrude_btn)

_mask_editor_widget.append(widgets.Label(value=_divider_style))
_mask_editor_widget.append(erase_label)
_mask_editor_widget.append(erase_chk)
_mask_editor_widget.append(erase_slider_container)
_mask_editor_widget.append(hover_chk)

_mask_editor_widget.append(widgets.Label(value=_divider_style))
_mask_editor_widget.append(undo_btn)
_mask_editor_widget.append(refresh_ids_btn)

@require_active_session("Please start or load a session before editing masks.")
def _on_paint(value: bool):
    if not session.get_data("output_dir"):
        paint_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            erase_chk.value = False
            hover_chk.value = False  # hover refresh conflicts with paint interaction
            viewer.layers.selection.active = layer
            layer.mode = 'paint'
            layer.n_edit_dimensions = 2
            new_id = int(layer.data.max()) + 1
            layer.selected_label = new_id
            viewer.status = f"Painting Mode. New ID: {new_id}"
        elif layer.mode == 'paint':
            layer.mode = 'pan_zoom'
            viewer.status = "Painting Mode Off."

def _apply_cylinder_erase(layer, world_pos):
    """Erases a cylinder of voxels (circle in YX * depth in Z) at the given world position."""
    coords = layer.world_to_data(world_pos)
    if len(coords) < 3:
        return
    z, y, x = int(round(coords[0])), int(round(coords[1])), int(round(coords[2]))
    radius = erase_radius_slider.value
    depth = erase_depth_slider.value
    dz, dy, dx = layer.data.shape

    z_start = max(0, z - depth // 2)
    z_end = min(dz, z_start + depth)
    y_lo = max(0, y - radius)
    y_hi = min(dy, y + radius + 1)
    x_lo = max(0, x - radius)
    x_hi = min(dx, x + radius + 1)

    # Build a circular structuring element and clip to data bounds
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    circle = (yy ** 2 + xx ** 2) <= radius ** 2
    cy_lo = y_lo - (y - radius)
    cy_hi = circle.shape[0] - ((y + radius + 1) - y_hi)
    cx_lo = x_lo - (x - radius)
    cx_hi = circle.shape[1] - ((x + radius + 1) - x_hi)
    clipped = circle[cy_lo:cy_hi, cx_lo:cx_hi]

    for zi in range(z_start, z_end):
        slc = layer.data[zi, y_lo:y_hi, x_lo:x_hi]
        slc[clipped] = 0
    layer.refresh()

def _cylinder_erase_drag(viewer, event):
    """Mouse drag callback that performs cylinder erase on click and drag."""
    layer = _mask_editor_widget.mask_layer.value
    if not isinstance(layer, napari.layers.Labels):
        return
    # Block event from reaching napari's pan/zoom handler
    event.handled = True
    _mask_undo.begin(layer.data)
    _apply_cylinder_erase(layer, event.position)
    yield
    while event.type == 'mouse_move':
        _apply_cylinder_erase(layer, event.position)
        yield
    _mask_undo.end(layer.data)
    _schedule_save(layer)

@require_active_session("Please start or load a session before editing masks.")
def _on_erase(value: bool):
    if not session.get_data("output_dir"):
        erase_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            paint_chk.value = False
            hover_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'pan_zoom'
            viewer.mouse_drag_callbacks.append(_cylinder_erase_drag)
            viewer.status = f"Erase Mode. Radius={erase_radius_slider.value}, Depth={erase_depth_slider.value}"
        else:
            if _cylinder_erase_drag in viewer.mouse_drag_callbacks:
                viewer.mouse_drag_callbacks.remove(_cylinder_erase_drag)
            viewer.status = "Erase Mode Off."

@require_active_session("Please start or load a session before editing masks.")
def _on_extrude():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer: return
    
    label_id = layer.selected_label
    if label_id == 0:
        viewer.status = "Select a label to extrude (cannot extrude 0)."
        return
        
    if layer.ndim != 3:
        viewer.status = "Extrusion only works on 3D layers."
        return

    z_idx = int(viewer.dims.current_step[0])
    current_slice = layer.data[z_idx]
    mask = (current_slice == label_id)
    
    if not np.any(mask):
        viewer.status = f"Label {label_id} not found on current slice {z_idx}."
        return
        
    _mask_undo.begin(layer.data)
    layer.data = segmentation.extrude_label(layer.data, z_idx, label_id)
    _mask_undo.end(layer.data)
    viewer.status = f"Extruded ID {label_id} through all Z slices."

@require_active_session("Please start or load a session before editing masks.")
def _on_delete():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    src = _mask_editor_widget.source_id.value
    if layer and src > 0:
        if np.sum(layer.data == src) > 0:
            _mask_undo.begin(layer.data)
            layer.data = segmentation.delete_label(layer.data, src)
            _mask_undo.end(layer.data)
            viewer.status = f"Deleted ID {src}."
        else:
            viewer.status = f"ID {src} not found."

@hover_chk.changed.connect
def _on_hover_mode(value: bool):
    viewer = napari.current_viewer()
    global _highlighter
    if _highlighter is None:
        _highlighter = MaskHighlighter(viewer)
    
    if value:
        _highlighter.enable()
        viewer.status = "Hover Edit Mode ON. Nuclei turn red. Press 'C' to delete."
    else:
        _highlighter.disable()
        viewer.status = "Hover Edit Mode OFF."

@require_active_session("Please start or load a session before refreshing IDs.")
def _on_refresh_ids():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    viewer_helpers.add_or_update_label_ids(viewer, layer)

def _on_mask_undo():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        viewer.status = "No mask layer selected."
        return
    if _mask_undo.undo(layer.data):
        layer.refresh()
        viewer.status = f"Undo ({len(_mask_undo)} remaining)."
    else:
        viewer.status = "Nothing to undo."

# Connect signals after defining functions
paint_chk.changed.connect(_on_paint)
erase_chk.changed.connect(_on_erase)
extrude_btn.clicked.connect(_on_extrude)
delete_btn.clicked.connect(_on_delete)
undo_btn.clicked.connect(_on_mask_undo)
# Note: hover_chk is already connected via @hover_chk.changed.connect decorator on _on_hover_mode
refresh_ids_btn.clicked.connect(_on_refresh_ids)

# --- Auto-saving for selected mask layer ---

# Store a reference to the layer and the callback to allow disconnection
_mask_editor_widget._current_layer = None
_mask_editor_widget._current_callback = None

def _create_save_callback(layer):
    """Factory to create a save callback for a specific layer."""
    _refresh_timer = QTimer()
    _refresh_timer.setSingleShot(True)

    def _refresh_ids():
        viewer = napari.current_viewer()
        if viewer:
            ids_name = f"{layer.name}_IDs"
            if ids_name in viewer.layers:
                viewer_helpers.add_or_update_label_ids(viewer, layer)

    _refresh_timer.timeout.connect(_refresh_ids)

    def _save_mask_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir and layer and layer.name:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_path = seg_dir / f"{layer.name}.tif"
            tifffile.imwrite(mask_path, layer.data)
            session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
        # Debounced refresh: update existing ID layer 800 ms after the last edit
        _refresh_timer.start(800)

    return _save_mask_data

@_mask_editor_widget.mask_layer.changed.connect
def _on_mask_layer_changed(new_layer: "napari.layers.Labels"):
    """Disconnects the old listener and connects a new one to the selected layer."""
    old_layer = _mask_editor_widget._current_layer
    old_callback = _mask_editor_widget._current_callback

    if old_layer and old_callback and old_callback in old_layer.events.data.callbacks:
        old_layer.events.data.disconnect(old_callback)

    if new_layer:
        new_callback = _create_save_callback(new_layer)
        new_layer.events.data.connect(new_callback)
        _mask_editor_widget._current_layer = new_layer
        _mask_editor_widget._current_callback = new_callback
    else:
        _mask_editor_widget._current_layer = None
        _mask_editor_widget._current_callback = None

# --- UI Wrapper ---
mask_editor_widget = widgets.Container(labels=False)
header = widgets.Label(value="Mask Editor")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Manual editing of segmentation masks.</i>")
mask_editor_widget.extend([header, info, _mask_editor_widget])
