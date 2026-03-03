import napari
import numpy as np
from magicgui import magicgui, widgets
import tifffile
from pathlib import Path

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session
from ...core import segmentation
from ... import constants

class MaskHighlighter:
    """Highlights the hovered nucleus by overriding its color directly in the Labels layer."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.last_layer = None
        self.last_id = None
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
        """Removes the red color override from the last highlighted label."""
        if self.last_layer is not None and self.last_id is not None:
            try:
                cmap = self.last_layer.colormap
                cmap.color_dict.pop(int(self.last_id), None)
                if hasattr(cmap, '_clear_cache'):
                    cmap._clear_cache()
                self.last_layer.events.colormap()
                self.last_layer.refresh(extent=False)
            except Exception:
                pass
        self.last_layer = None
        self.last_id = None

    def perform_highlight(self, layer, label_id):
        """Sets the hovered label to red in the Labels layer colormap."""
        try:
            cmap = layer.colormap
            if self.last_id is not None and self.last_id != label_id:
                cmap.color_dict.pop(int(self.last_id), None)
            cmap.color_dict[int(label_id)] = np.array([1.0, 0.0, 0.0, 1.0])
            if hasattr(cmap, '_clear_cache'):
                cmap._clear_cache()
            layer.events.colormap()
            layer.refresh(extent=False)
        except Exception:
            pass

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
        
    mask_layer.data = segmentation.merge_labels(mask_layer.data, source_id, target_id)
    
    viewer.status = f"Merged ID {source_id} into {target_id} ({count} pixels)."

def delete_mask_under_mouse(viewer):
    """Deletes the mask label currently under the mouse cursor."""
    global _highlighter

    # 1. If hover-edit is active, use its state for accuracy.
    if _highlighter and _highlighter.active and _highlighter.last_id is not None:
        layer = _highlighter.last_layer
        
        id_to_delete = _highlighter.last_id

        if layer and id_to_delete:
            # First, remove the temporary highlight layer.
            _highlighter.reset_highlight()

            # Now, delete the label from the original data array.
            layer.data = segmentation.delete_label(layer.data, id_to_delete)
            
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
            layer.data = segmentation.delete_label(layer.data, val)
            viewer.status = f"Deleted Nucleus ID {val}"

# Add Tools to Mask Editor
editor_label = widgets.Label(value="<b>Editing Tools:</b>")
btn_container = widgets.Container(layout="horizontal", labels=False)
paint_chk = widgets.CheckBox(text="Paint (New ID)")
erase_chk = widgets.CheckBox(text="Erase")
pick_btn = widgets.PushButton(text="Pick ID")
extrude_btn = widgets.PushButton(text="Extrude ID (Fill Z)")
delete_btn = widgets.PushButton(text="Delete Source ID")
hover_chk = widgets.CheckBox(text="Hover Edit Mode (Red + 'C' to Del)")
refresh_ids_btn = widgets.PushButton(text="Show/Refresh IDs")

btn_container.extend([paint_chk, erase_chk, pick_btn])

_mask_editor_widget.append(editor_label)
_mask_editor_widget.append(btn_container)
_mask_editor_widget.append(hover_chk)
_mask_editor_widget.append(extrude_btn)
_mask_editor_widget.append(delete_btn)
_mask_editor_widget.append(refresh_ids_btn)

@require_active_session("Please start or load a session before editing masks.")
def _on_paint(value: bool):
    if not session.get_data("output_dir"): # Check again in case session was closed
        paint_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            erase_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'paint'
            layer.n_edit_dimensions = 2
            new_id = int(layer.data.max()) + 1
            layer.selected_label = new_id
            viewer.status = f"Painting Mode. New ID: {new_id}"
        elif layer.mode == 'paint':
            layer.mode = 'pan_zoom'
            viewer.status = "Painting Mode Off."

@require_active_session("Please start or load a session before editing masks.")
def _on_erase(value: bool):
    if not session.get_data("output_dir"): # Check again in case session was closed
        erase_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            paint_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'erase'
            viewer.status = "Erase Mode."
        elif layer.mode == 'erase':
            layer.mode = 'pan_zoom'
            viewer.status = "Erase Mode Off."

@require_active_session("Please start or load a session before editing masks.")
def _on_pick():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        viewer.layers.selection.active = layer
        layer.mode = 'pick'
        paint_chk.value = False
        erase_chk.value = False
        viewer.status = "Pick Mode. Click a label to select its ID."

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
        
    layer.data = segmentation.extrude_label(layer.data, z_idx, label_id)
    viewer.status = f"Extruded ID {label_id} through all Z slices."

@require_active_session("Please start or load a session before editing masks.")
def _on_delete():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    src = _mask_editor_widget.source_id.value
    if layer and src > 0:
        if np.sum(layer.data == src) > 0:
            layer.data = segmentation.delete_label(layer.data, src)
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

# Connect signals after defining functions
paint_chk.changed.connect(_on_paint)
erase_chk.changed.connect(_on_erase)
pick_btn.clicked.connect(_on_pick)
extrude_btn.clicked.connect(_on_extrude)
delete_btn.clicked.connect(_on_delete)
hover_chk.changed.connect(_on_hover_mode)
refresh_ids_btn.clicked.connect(_on_refresh_ids)

# --- Auto-saving for selected mask layer ---

# Store a reference to the layer and the callback to allow disconnection
_mask_editor_widget._current_layer = None
_mask_editor_widget._current_callback = None

def _create_save_callback(layer):
    """Factory to create a save callback for a specific layer."""
    def _save_mask_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir and layer and layer.name:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_path = seg_dir / f"{layer.name}.tif"
            tifffile.imwrite(mask_path, layer.data)
            session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
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
