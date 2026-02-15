import napari
import numpy as np
import tifffile
from pathlib import Path
from magicgui import magicgui

from ...core import session
from .. import popups
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

            masks, centroids = segment_nuclei_classical(layer.data, progress_callback=on_progress)
            
            out_dir = session.get_data("output_dir")
            if out_dir:
                seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
                
            if masks is not None:
                viewer.add_labels(masks, name=f"{layer.name}{constants.MASKS_SUFFIX}", opacity=0.3, visible=False, scale=layer.scale)
                if out_dir:
                    mask_path = seg_dir / f"{layer.name}{constants.MASKS_SUFFIX}.tif"
                    tifffile.imwrite(mask_path, masks)
                    session.set_processed_file(f"{layer.name}{constants.MASKS_SUFFIX}", str(mask_path), layer_type='labels', metadata={'subtype': 'mask'})
                
            if centroids is not None:
                ids = np.arange(len(centroids)) + 1
                viewer.add_points(
                    centroids,
                    name=f"{layer.name}{constants.CENTROIDS_SUFFIX}",
                    size=5,
                    face_color='orange',
                    scale=layer.scale,
                    properties={'id': ids},
                    text={'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, -5, 0])},
                    blending='translucent_no_depth'
                )
                if out_dir:
                    cent_path = seg_dir / f"{layer.name}{constants.CENTROIDS_SUFFIX}.npy"
                    np.save(cent_path, centroids)
                    session.set_processed_file(f"{layer.name}{constants.CENTROIDS_SUFFIX}", str(cent_path), layer_type='points', metadata={'subtype': 'centroids'})
        
        dialog.update_progress(100, "Complete.")
        viewer.status = "Segmentation complete."
