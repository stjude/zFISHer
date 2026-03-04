import logging
import napari
import numpy as np
from pathlib import Path
from qtpy.QtCore import QTimer
from ..core import session
from .. import constants

logger = logging.getLogger(__name__)

# Track layers that already have listeners to prevent duplicate attachment
_attached_listeners = set()

def attach_puncta_listener(layer, name):
    """Attaches listeners to a points layer for auto-saving and color syncing."""
    if id(layer) in _attached_listeners:
        return
    _attached_listeners.add(id(layer))
    def sync_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            puncta_path = seg_dir / f"{name}.npy"
            np.save(puncta_path, layer.data)
            session.set_processed_file(name, str(puncta_path), layer_type='points', metadata={'subtype': 'puncta'})
            
    def sync_color(event=None):
        # Auto-update all points when layer color properties change
        layer.face_color = layer.current_face_color



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
                    widgets['dapi_segmentation']._dapi_segmentation_widget.r1_layer.value = layer
                    widgets['automated_preprocessing']._automated_preprocessing_magic_widget.r1_dapi_layer.value = layer
                elif "R2" in layer.name.upper():
                    widgets['dapi_segmentation']._dapi_segmentation_widget.r2_layer.value = layer
                    widgets['automated_preprocessing']._automated_preprocessing_magic_widget.r2_dapi_layer.value = layer
        
        elif isinstance(layer, napari.layers.Points):
            if "centroids" in layer.name.lower():
                if "R1" in layer.name.upper():
                    widgets['registration']._registration_widget.r1_points.value = layer
                elif "R2" in layer.name.upper():
                    widgets['registration']._registration_widget.r2_points.value = layer
            
            # Auto-select the new points layer in editors/analysis widgets
            widgets['puncta_editor']._puncta_editor_widget.points_layer.value = layer
            widgets['colocalization']._rule_builder.source_layer.value = layer
            
            # If it's a puncta layer, ensure its listener is attached.
            if "puncta" in layer.name.lower():
                 attach_puncta_listener(layer, layer.name)
        
        elif isinstance(layer, napari.layers.Labels):
            name = layer.name.upper()
            # Heuristic for matching aligned/warped DAPI masks
            if "DAPI" in name and ("ALIGNED" in name or "WARPED" in name):
                if "R1" in name:
                    widgets['nuclei_matching']._nuclei_matching_widget.r1_mask_layer.value = layer
                elif "R2" in name:
                    widgets['nuclei_matching']._nuclei_matching_widget.r2_mask_layer.value = layer
            
            # General purpose editors
            widgets['mask_editor']._mask_editor_widget.mask_layer.value = layer
            # Auto-select a mask for puncta detection, but not other puncta layers
            if "puncta" not in name.lower():
                widgets['puncta_detection']._puncta_widget.nuclei_layer.value = layer
    
    # Use a QTimer to delay execution slightly, ensuring the layer is fully added
    QTimer.singleShot(100, update_widgets)

def on_layer_removed(event, widgets):
    """
    Refreshes all widget dropdowns when a layer is removed.
    For Labels layers, also removes the associated _IDs points layer
    and cleans up the session registry.
    """
    layer = event.value
    viewer = napari.current_viewer()

    # Remove the orphan _IDs display layer (if it exists) and clean up session data
    if isinstance(layer, napari.layers.Labels):
        ids_name = f"{layer.name}_IDs"
        if viewer and ids_name in viewer.layers:
            viewer.layers.remove(viewer.layers[ids_name])
        session.remove_processed_file(layer.name)
        session.remove_processed_file(ids_name)

    # Clean up stale session entry and listener tracking for removed points layers
    elif isinstance(layer, napari.layers.Points):
        session.remove_processed_file(layer.name)
        _attached_listeners.discard(id(layer))

    def update_choices():
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                w.reset_choices()

    QTimer.singleShot(10, update_choices)
