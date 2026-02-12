import napari
import numpy as np
from magicgui import magicgui, widgets

import zfisher.core.session as session
from .. import popups
from zfisher.core.segmentation import get_mask_centroids

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

# Add Tools to Mask Editor
editor_label = widgets.Label(value="<b>Editing Tools:</b>")
btn_container = widgets.Container(layout="horizontal", labels=False)
paint_chk = widgets.CheckBox(text="Paint (New ID)")
erase_chk = widgets.CheckBox(text="Erase")
pick_btn = widgets.PushButton(text="Pick ID")
extrude_btn = widgets.PushButton(text="Extrude ID (Fill Z)")
delete_btn = widgets.PushButton(text="Delete Source ID")
refresh_ids_btn = widgets.PushButton(text="Show/Refresh IDs")

btn_container.extend([paint_chk, erase_chk, pick_btn])

mask_editor_widget.append(editor_label)
mask_editor_widget.append(btn_container)
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
