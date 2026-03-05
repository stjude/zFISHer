import logging
import napari
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtCore import QTimer
from ..core import session
from .. import constants

logger = logging.getLogger(__name__)

# Track layers that already have listeners to prevent duplicate attachment.
# Maps layer id -> {'layer': layer_ref, 'sync_data': callback, 'sync_color': callback}
_attached_listeners = {}

def attach_puncta_listener(layer, name):
    """Attaches listeners to a points layer for auto-saving and color syncing."""
    if id(layer) in _attached_listeners:
        return

    def sync_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir:
            reports_dir = Path(out_dir) / constants.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)
            csv_path = reports_dir / f"{name}.csv"
            try:
                coords_df = pd.DataFrame(layer.data, columns=['Z', 'Y', 'X'])
                features = layer.features.reset_index(drop=True) if hasattr(layer, 'features') and not layer.features.empty else pd.DataFrame()
                full_df = pd.concat([features, coords_df], axis=1)
                full_df.to_csv(csv_path, index=False)
            except Exception:
                # Fallback: save coordinates only
                np.savetxt(csv_path, layer.data, delimiter=',', header='Z,Y,X', comments='')
            session.set_processed_file(name, str(csv_path), layer_type='points', metadata={'subtype': 'puncta_csv'})

    _sync_color_guard = {'active': False}

    def sync_color(event=None):
        # Guard against re-entrant calls: setting face_color can re-emit
        # current_face_color events, causing a feedback loop during 2D/3D
        # mode transitions.
        if _sync_color_guard['active']:
            return
        _sync_color_guard['active'] = True
        try:
            layer.face_color = layer.current_face_color
        finally:
            _sync_color_guard['active'] = False

    layer.events.data.connect(sync_data)
    layer.events.current_face_color.connect(sync_color)

    _attached_listeners[id(layer)] = {
        'layer': layer,
        'sync_data': sync_data,
        'sync_color': sync_color,
    }


def detach_puncta_listener(layer):
    """Disconnects listeners previously attached to a points layer."""
    info = _attached_listeners.pop(id(layer), None)
    if info is None:
        return
    try:
        layer.events.data.disconnect(info['sync_data'])
    except (TypeError, RuntimeError):
        pass
    try:
        layer.events.current_face_color.disconnect(info['sync_color'])
    except (TypeError, RuntimeError):
        pass

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
        detach_puncta_listener(layer)

    def update_choices():
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                w.reset_choices()

    QTimer.singleShot(10, update_choices)
