import napari
import numpy as np
import tifffile
import gc 
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session
from .. import popups
from ..decorators import require_active_session, error_handler
from ...core.segmentation import (
    segment_nuclei_classical,
    match_nuclei_labels,
    merge_labeled_masks,
)
from ...core.registration import align_centroids_ransac, generate_global_canvas
from ... import constants

@magicgui(
    call_button="Run Automated Preprocessing",
    viewer={"visible": False, "label": " "},
    r1_dapi_layer={"label": "Round 1 DAPI Layer"},
    r2_dapi_layer={"label": "Round 2 DAPI Layer"},
    match_nuclei={"label": "Create Consensus Nuclei Mask"},
    match_threshold={"label": "Nuclei Match Distance (px)", "min": 0, "max": 100, "step": 1},
    hide_raw={"label": "Hide Raw Layers After?"}
)
@require_active_session("Please start or load a session first.")
@error_handler("Automated Preprocessing Failed")
def _automated_preprocessing_magic_widget(
    viewer: "napari.Viewer",
    r1_dapi_layer: "napari.layers.Image",
    r2_dapi_layer: "napari.layers.Image",
    match_nuclei: bool = True,
    match_threshold: float = 20.0,
    hide_raw: bool = True
):
    """
    Automated pipeline mirrored from manual widgets with strict type handling 
    to prevent 'Non-integer label_image' and 'scale' errors.
    """
    if not r1_dapi_layer or not r2_dapi_layer:
        viewer.status = "Please select both DAPI layers."
        return

    with popups.ProgressDialog(None, "Automated Preprocessing...") as dialog:
        output_dir = session.get_data("output_dir")
        
        # === STEP 1: SEGMENTATION ===
        dialog.update_progress(5, "Segmenting R1 DAPI...")
        r1_masks, r1_centroids = segment_nuclei_classical(r1_dapi_layer.data, progress_callback=lambda p, m: dialog.update_progress(5 + int(p*0.1), f"R1: {m}"))
        
        dialog.update_progress(20, "Segmenting R2 DAPI...")
        r2_masks, r2_centroids = segment_nuclei_classical(r2_dapi_layer.data, progress_callback=lambda p, m: dialog.update_progress(20 + int(p*0.1), f"R2: {m}"))

        # === STEP 2: REGISTRATION ===
        dialog.update_progress(35, "Calculating Global Shift...")
        shift, rmsd = align_centroids_ransac(r1_centroids, r2_centroids, progress_callback=lambda p, m: dialog.update_progress(35 + int(p*0.1), f"Registering: {m}"))
        session.update_data("shift", shift.tolist())
        
        # === STEP 3: GLOBAL CANVAS (MIRRORED LOGIC) ===
        dialog.update_progress(50, "Preparing Global Canvas...")
        
        r1_layers_data, r2_layers_data = [], []
        for l in viewer.layers:
            # FIX: Only collect Image or Labels layers. Ignore Points/Centroids.
            if isinstance(l, (napari.layers.Image, napari.layers.Labels)):
                if any(x in l.name for x in ["Aligned", "Warped", "Consensus"]): 
                    continue
                
                is_label = isinstance(l, napari.layers.Labels)
                
                # FIX: Strict type casting for SimpleITK 'Ambiguous' error
                if is_label:
                    layer_data = l.data.astype(np.uint32)
                else:
                    layer_data = l.data.astype(np.float32) 
                
                layer_info = {
                    'name': l.name, 
                    'data': layer_data, 
                    'colormap': getattr(l.colormap, 'name', 'gray') if hasattr(l, 'colormap') else 'gray', 
                    'scale': l.scale, # Now safe because we only have Image/Labels
                    'is_label': is_label
                }
                
                if "R1" in l.name: r1_layers_data.append(layer_info)
                elif "R2" in l.name: r2_layers_data.append(layer_info)

        canvas_out = Path(output_dir) / constants.ALIGNED_DIR
        canvas_out.mkdir(exist_ok=True, parents=True)
        
        # Call core with callback mirrored from canvas_widget.py
        results = generate_global_canvas(
            r1_layers_data, r2_layers_data, shift, canvas_out, 
            apply_warp=True, 
            progress_callback=lambda p, m: dialog.update_progress(50 + int(p * 0.35), m)
        )

        aligned_r1_dapi, aligned_r2_dapi = None, None
        for layer_info in results:
            layer_type = layer_info['type']
            meta = layer_info['meta']
            new_l = None

            if layer_type == 'labels':
                new_l = viewer.add_labels(layer_info['data'].astype(np.uint32), name=layer_info['name'], scale=meta['scale'], opacity=0.6)
            elif layer_type == 'image':
                new_l = viewer.add_image(
                    layer_info['data'], name=layer_info['name'], 
                    colormap=meta.get('colormap', 'gray'), 
                    scale=meta['scale'], 
                    blending=meta.get('blending', 'additive'),
                    opacity=meta.get('opacity', 1.0))
            elif layer_type == 'vectors':
                new_l = viewer.add_vectors(
                    layer_info['data'],
                    name=layer_info['name'],
                    scale=meta['scale'],
                    edge_width=0.2,
                    length=2.5,
                    edge_color='cyan'
                )

            # Identify layers for Step 4
            if new_l and layer_info['name'] == f"{constants.ALIGNED_PREFIX} R1 - DAPI":
                aligned_r1_dapi = new_l
            elif new_l and layer_info['name'] == f"{constants.WARPED_PREFIX} R2 - DAPI":
                aligned_r2_dapi = new_l

        # Clean memory to prevent 'Killed' error
        r1_layers_data.clear(); r2_layers_data.clear(); gc.collect()

        # === STEP 4: NUCLEI MATCHING (MIRRORED LOGIC) ===
        if match_nuclei and aligned_r1_dapi and aligned_r2_dapi:
            dialog.update_progress(90, "Creating Consensus Mask...")
            from ...core import segmentation
            merged_mask, pts1 = segmentation.process_consensus_nuclei(
                aligned_r1_dapi.data, aligned_r2_dapi.data, output_dir, 
                threshold=match_threshold, method="Union", 
                progress_callback=lambda p, m: dialog.update_progress(90 + int(p*0.1), m)
            )
            viewer.add_labels(merged_mask, name=constants.CONSENSUS_MASKS_NAME, scale=aligned_r1_dapi.scale, opacity=0.5)

        if hide_raw:
            for layer in viewer.layers:
                layer.visible = any(x in layer.name for x in ["Aligned", "Warped", "Consensus"])

        dialog.update_progress(100, "Complete."); viewer.status = "Automated Preprocessing Complete."

# UI Wrapper
automated_preprocessing_widget = Container(labels=False)
header = Label(value="Automated Preprocessing")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Segmentation, Registration, and Warping (Mirrored).</i>")
automated_preprocessing_widget.extend([header, info, _automated_preprocessing_magic_widget])