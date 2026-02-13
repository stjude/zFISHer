import napari
import numpy as np
from magicgui import magicgui, widgets
import tifffile
from pathlib import Path

import zfisher.core.session as session
from .. import popups
from zfisher.core.segmentation import get_mask_centroids

class MaskHighlighter:
    """Helper class to highlight labels in red under the mouse cursor."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.last_layer = None
        self.last_id = None
        self.saved_color = None
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
        if self.last_layer and self.last_id is not None:
            try:
                # Restore original color
                if self.saved_color is None:
                    if self.last_id in self.last_layer.color:
                        del self.last_layer.color[self.last_id]
                else:
                    self.last_layer.color[self.last_id] = self.saved_color
                self.last_layer.refresh()
            except Exception:
                pass
        self.last_layer = None
        self.last_id = None
        self.saved_color = None

    def on_mouse_move(self, viewer, event):
        if not self.active: return
        
        layer = mask_editor_widget.mask_layer.value
        if not isinstance(layer, napari.layers.Labels):
            self.reset_highlight()
            return

        val = layer.get_value(
            event.position,
            view_direction=event.view_direction,
            dims_displayed=list(event.dims_displayed),
            world=True
        )
        
        if val is None or val == 0:
            self.reset_highlight()
            return
            
        if val != self.last_id or layer != self.last_layer:
            self.reset_highlight()
            self.last_layer = layer
            self.last_id = val
            self.saved_color = layer.color.get(val)
            
            # Set to bright red
            layer.color[val] = 'red'
            layer.refresh()
            viewer.status = f"Hovering Nucleus ID: {val} (Press 'C' to delete)"

# Global instance
_highlighter = None

@magicgui(
    call_button="Merge IDs",
    mask_layer={"label": "Layer to Edit"},
    source_id={"label": "Source ID"},
    target_id={"label": "Target ID"}
)
def mask_editor_widget(
    mask_layer: "napari.layers.Labels",
    source_id: int = 0,
    target_id: int = 0
):
    """Merges two labels in the selected mask layer."""
    viewer = napari.current_viewer()

    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        return

    if mask_layer is None:
        viewer.status = "No mask layer selected."
        return

    if source_id == target_id:
        viewer.status = "Source and Target IDs must be different."
        return
        
    data = mask_layer.data
    count = np.sum(data == source_id)
    
    if count == 0:
        viewer.status = f"ID {source_id} not found."
        return
        
    new_data = data.copy()
    new_data[new_data == source_id] = target_id
    mask_layer.data = new_data
    
    viewer.status = f"Merged ID {source_id} into {target_id} ({count} pixels)."

def delete_mask_under_mouse(viewer):
    """Deletes the mask label currently under the mouse cursor."""
    global _highlighter
    
    # 1. Try to use the highlighter's cached ID for speed/accuracy if active
    if _highlighter and _highlighter.active and _highlighter.last_id:
        layer = _highlighter.last_layer
        val = _highlighter.last_id
        if layer and val:
            # Delete the label
            layer.data[layer.data == val] = 0
            layer.refresh()
            viewer.status = f"Deleted Nucleus ID {val}"
            # Reset highlighter since ID is gone
            _highlighter.reset_highlight()
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
            layer.data[layer.data == val] = 0
            layer.refresh()
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

mask_editor_widget.append(editor_label)
mask_editor_widget.append(btn_container)
mask_editor_widget.append(hover_chk)
mask_editor_widget.append(extrude_btn)
mask_editor_widget.append(delete_btn)
mask_editor_widget.append(refresh_ids_btn)

@paint_chk.changed.connect
def _on_paint(value: bool):
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        paint_chk.value = False
        return

    layer = mask_editor_widget.mask_layer.value
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

@erase_chk.changed.connect
def _on_erase(value: bool):
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        erase_chk.value = False
        return

    layer = mask_editor_widget.mask_layer.value
    if layer:
        if value:
            paint_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'erase'
            viewer.status = "Erase Mode."
        elif layer.mode == 'erase':
            layer.mode = 'pan_zoom'
            viewer.status = "Erase Mode Off."

@pick_btn.clicked.connect
def _on_pick():
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        return

    layer = mask_editor_widget.mask_layer.value
    if layer:
        viewer.layers.selection.active = layer
        layer.mode = 'pick'
        paint_chk.value = False
        erase_chk.value = False
        viewer.status = "Pick Mode. Click a label to select its ID."

@extrude_btn.clicked.connect
def _on_extrude():
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        return

    layer = mask_editor_widget.mask_layer.value
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
        
    layer.data[:, mask] = label_id
    layer.refresh()
    viewer.status = f"Extruded ID {label_id} through all Z slices."

@delete_btn.clicked.connect
def _on_delete():
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before editing masks."
        )
        return

    layer = mask_editor_widget.mask_layer.value
    src = mask_editor_widget.source_id.value
    if layer and src > 0:
        data = layer.data
        if np.sum(data == src) > 0:
            new_data = data.copy()
            new_data[new_data == src] = 0
            layer.data = new_data
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

@refresh_ids_btn.clicked.connect
def _on_refresh_ids():
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before refreshing IDs."
        )
        return

    layer = mask_editor_widget.mask_layer.value
    if not layer: return
    
    pts_data = get_mask_centroids(layer.data)
    
    name = f"{layer.name}_IDs"
    coords = np.array([p['coord'] for p in pts_data]) if pts_data else np.empty((0, 3))
    labels = np.array([p['label'] for p in pts_data]) if pts_data else np.empty(0)
    
    if name in viewer.layers:
        viewer.layers.remove(name)
        
    if len(coords) > 0:
        viewer.add_points(
            coords,
            name=name,
            size=0,
            scale=layer.scale,
            properties={'label': labels},
            text={'string': '{label}', 'size': 10, 'color': 'cyan', 'translation': np.array([0, -5, 0])},
            blending='translucent_no_depth'
        )
    viewer.status = f"Refreshed IDs for {layer.name}"

# --- Auto-saving for selected mask layer ---

# Store a reference to the layer and the callback to allow disconnection
mask_editor_widget._current_layer = None
mask_editor_widget._current_callback = None

def _create_save_callback(layer):
    """Factory to create a save callback for a specific layer."""
    def _save_mask_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir and layer and layer.name: # Ensure layer name is valid
            seg_dir = Path(out_dir) / "segmentation"
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_path = seg_dir / f"{layer.name}.tif"
            tifffile.imwrite(mask_path, layer.data)
            session.set_processed_file(layer.name, str(mask_path))
            session.save_session()
    return _save_mask_data

@mask_editor_widget.mask_layer.changed.connect
def _on_mask_layer_changed(new_layer: "napari.layers.Labels"):
    """Disconnects the old listener and connects a new one to the selected layer."""
    old_layer = mask_editor_widget._current_layer
    old_callback = mask_editor_widget._current_callback

    if old_layer and old_callback and old_callback in old_layer.events.data.callbacks:
        old_layer.events.data.disconnect(old_callback)

    if new_layer:
        new_callback = _create_save_callback(new_layer)
        new_layer.events.data.connect(new_callback)
        mask_editor_widget._current_layer = new_layer
        mask_editor_widget._current_callback = new_callback
    else:
        mask_editor_widget._current_layer = None
        mask_editor_widget._current_callback = None
