import napari
import numpy as np
from pathlib import Path
from qtpy.QtCore import QTimer
import zfisher.core.session as session

def attach_puncta_listener(layer, name):
    """Attaches listeners to a points layer for auto-saving and color syncing."""
    def sync_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir:
            seg_dir = Path(out_dir) / "segmentation"
            seg_dir.mkdir(exist_ok=True, parents=True)
            puncta_path = seg_dir / f"{name}.npy"
            np.save(puncta_path, layer.data)
            session.set_processed_file(name, str(puncta_path))
            session.save_session()
            
    def sync_color(event=None):
        # Auto-update all points when layer color properties change
        layer.face_color = layer.current_face_color

    # Disconnect any previous listeners to be safe
    for callback in list(layer.events.data.callbacks):
        layer.events.data.disconnect(callback)
    for callback in list(layer.events.current_face_color.callbacks):
        layer.events.current_face_color.disconnect(callback)

    layer.events.data.connect(sync_data)
    layer.events.current_face_color.connect(sync_color)

def on_layer_inserted(event, widgets):
    """
    Handles auto-selecting layers in widgets when a new layer is added.
    """
    layer = event.value
    
    def update_widgets():
        # First, refresh all dropdowns to ensure the new layer is in the list
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                w.reset_choices()

        # Now, try to intelligently set the value based on layer type and name
        if isinstance(layer, napari.layers.Image):
            if "DAPI" in layer.name.upper():
                if "R1" in layer.name.upper():
                    widgets['dapi_segmentation'].r1_layer.value = layer
                elif "R2" in layer.name.upper():
                    widgets['dapi_segmentation'].r2_layer.value = layer
        
        elif isinstance(layer, napari.layers.Points):
            if "centroids" in layer.name.lower():
                if "R1" in layer.name.upper():
                    widgets['registration'].r1_points.value = layer
                elif "R2" in layer.name.upper():
                    widgets['registration'].r2_points.value = layer
            
            # Auto-select the new points layer in editors/analysis widgets
            widgets['puncta_editor'].points_layer.value = layer
            widgets['colocalization'].source_layer.value = layer
            
            # If it's a puncta layer, ensure its listener is attached.
            if "puncta" in layer.name.lower():
                 attach_puncta_listener(layer, layer.name)
        
        elif isinstance(layer, napari.layers.Labels):
            name = layer.name.upper()
            # Heuristic for matching aligned/warped DAPI masks
            if "DAPI" in name and ("ALIGNED" in name or "WARPED" in name):
                if "R1" in name:
                    widgets['nuclei_matching'].r1_mask_layer.value = layer
                elif "R2" in name:
                    widgets['nuclei_matching'].r2_mask_layer.value = layer
            
            # General purpose editors
            widgets['mask_editor'].mask_layer.value = layer
            # Auto-select a mask for puncta detection, but not other puncta layers
            if "puncta" not in name.lower():
                widgets['puncta_detection'].nuclei_layer.value = layer
    
    # Use a QTimer to delay execution slightly, ensuring the layer is fully added
    QTimer.singleShot(100, update_widgets)

def on_layer_removed(event, widgets):
    """
    Refreshes all widget dropdowns when a layer is removed.
    """
    def update_choices():
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                w.reset_choices()
    
    QTimer.singleShot(10, update_choices)
