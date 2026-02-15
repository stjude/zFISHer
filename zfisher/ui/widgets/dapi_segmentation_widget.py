import napari
import numpy as np
from magicgui import magicgui

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.segmentation import segment_nuclei_classical
from ... import constants

@magicgui(
    call_button="Run DAPI Mapping",
    r1_layer={"label": "Round 1 (DAPI)"},
    r2_layer={"label": "Round 2 (DAPI)"},
    auto_call=False,
)
@require_active_session("Please start or load a session before running segmentation.")
@error_handler("DAPI Segmentation Failed")
def dapi_segmentation_widget(
    r1_layer: "napari.layers.Image",
    r2_layer: "napari.layers.Image"
):
    """Runs segmentation on selected DAPI channels."""
    viewer = napari.current_viewer()

    layers_to_process = [l for l in [r1_layer, r2_layer] if l is not None]
    
    if not layers_to_process:
        viewer.status = "No channels selected."
        return

    viewer.status = f"Segmenting {len(layers_to_process)} layer(s)..."
    
    with popups.ProgressDialog(viewer.window._qt_window, title="Segmenting Nuclei...") as dialog:
        num_layers = len(layers_to_process)
        for i, layer in enumerate(layers_to_process):
            dialog.update_progress(0, f"Starting segmentation for {layer.name}...")
            
            # Create a callback that scales progress to the layer's portion of the bar
            def on_progress(value, text):
                base_progress = (i / num_layers) * 100
                scaled_value = base_progress + (value / num_layers)
                dialog.update_progress(int(scaled_value), f"{layer.name}: {text}")

            # 1. Call core segmentation function
            masks, centroids = segment_nuclei_classical(layer.data, progress_callback=on_progress)
            
            # 2. Pass results to UI helper
            viewer_helpers.add_segmentation_results_to_viewer(viewer, layer, masks, centroids)
        
        dialog.update_progress(100, "Complete.")
        viewer.status = "Segmentation complete."
