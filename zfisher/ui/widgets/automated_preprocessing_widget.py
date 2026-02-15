import napari
import numpy as np
import tifffile
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session
from .. import popups
from ..decorators import require_active_session, error_handler
from ...core.registration import align_centroids_ransac
from ...core.segmentation import (
    segment_nuclei_classical,
    match_nuclei_labels,
    merge_labeled_masks,
)
from ...core.pipeline import generate_global_canvas
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
    Runs the full preprocessing pipeline:
    1. Segments DAPI nuclei to find centroids.
    2. Registers rounds using centroids (RANSAC).
    3. Generates a global canvas by warping all R2 layers.
    4. (Optional) Matches nuclei from aligned DAPI masks to create a consensus mask.
    """
    if not r1_dapi_layer or not r2_dapi_layer:
        viewer.status = "Please select both DAPI layers."
        return

    with popups.ProgressDialog(viewer.window._qt_window, "Automated Preprocessing...") as dialog:
        output_dir = session.get_data("output_dir")
        # === STEP 1: DAPI SEGMENTATION ===
        dialog.update_progress(5, "Segmenting R1 DAPI...")
        r1_masks, r1_centroids = segment_nuclei_classical(r1_dapi_layer.data, progress_callback=lambda p, m: dialog.update_progress(5 + int(p*0.15), f"Segmenting R1: {m}"))
        if r1_centroids is None or len(r1_centroids) == 0:
            raise ValueError("R1 DAPI segmentation failed or found no nuclei.")
        
        viewer.add_points(r1_centroids, name="R1_centroids", scale=r1_dapi_layer.scale, face_color='cyan', size=10)

        dialog.update_progress(20, "Segmenting R2 DAPI...")
        r2_masks, r2_centroids = segment_nuclei_classical(r2_dapi_layer.data, progress_callback=lambda p, m: dialog.update_progress(20 + int(p*0.15), f"Segmenting R2: {m}"))
        if r2_centroids is None or len(r2_centroids) == 0:
            raise ValueError("R2 DAPI segmentation failed or found no nuclei.")
        
        viewer.add_points(r2_centroids, name="R2_centroids", scale=r2_dapi_layer.scale, face_color='magenta', size=10)

        # === STEP 2: REGISTRATION ===
        dialog.update_progress(35, "Registering rounds...")
        shift = align_centroids_ransac(r1_centroids, r2_centroids, progress_callback=lambda p, m: dialog.update_progress(35 + int(p*0.15), f"Registering: {m}"))
        session.update_data("shift", shift.tolist())
        viewer.status = f"Calculated Shift: {np.round(shift, 2)}"

        # === STEP 3: GENERATE GLOBAL CANVAS ===
        dialog.update_progress(50, "Generating global canvas...")
        
        r1_layers_data, r2_layers_data = [], []
        for l in viewer.layers:
            if isinstance(l, (napari.layers.Image, napari.layers.Labels)) and ("R1" in l.name or "R2" in l.name):
                if "Aligned" in l.name or "Warped" in l.name or "Consensus" in l.name: continue
                layer_info = {'name': l.name, 'data': l.data, 'scale': l.scale, 'colormap': getattr(l.colormap, 'name', None), 'is_label': isinstance(l, napari.layers.Labels)}
                if "R1" in l.name: r1_layers_data.append(layer_info)
                elif "R2" in l.name: r2_layers_data.append(layer_info)
        
        canvas_output_dir = Path(output_dir) / "aligned"
        canvas_output_dir.mkdir(exist_ok=True, parents=True)
        
        aligned_r1_dapi_mask, aligned_r2_dapi_mask = None, None
        pipeline_generator = generate_global_canvas(r1_layers_data, r2_layers_data, shift, canvas_output_dir, apply_warp=True)
        num_layers = len(r1_layers_data) + len(r2_layers_data)

        for i, (progress, message, result) in enumerate(pipeline_generator):
            dialog.update_progress(50 + int((i / num_layers) * 30), message)
            if not result: continue

            for layer_data in [res for res in [result.get('r1'), result.get('r2')] if res]:
                if layer_data.get('is_label'):
                    new_layer = viewer.add_labels(layer_data['data'].astype(np.uint32), name=layer_data['name'], scale=layer_data['scale'], opacity=0.6)
                else:
                    new_layer = viewer.add_image(layer_data['data'], name=layer_data['name'], colormap=layer_data.get('colormap', 'gray'), scale=layer_data['scale'], blending='additive')
                
                if match_nuclei:
                    if layer_data['name'] == r1_dapi_layer.name + "_Aligned": aligned_r1_dapi_mask = new_layer
                    elif layer_data['name'] == r2_dapi_layer.name + "_Warped": aligned_r2_dapi_mask = new_layer

        # === STEP 4: NUCLEI MATCHING ===
        if match_nuclei:
            dialog.update_progress(85, "Matching nuclei...")
            if aligned_r1_dapi_mask is None or aligned_r2_dapi_mask is None:
                raise ValueError("Could not find aligned DAPI masks for matching.")

            new_mask2, _, _ = match_nuclei_labels(aligned_r1_dapi_mask.data, aligned_r2_dapi_mask.data, threshold=match_threshold)
            merged_mask = merge_labeled_masks(aligned_r1_dapi_mask.data, new_mask2)
            
            layer_name = constants.CONSENSUS_MASKS_NAME
            viewer.add_labels(merged_mask, name=layer_name, scale=aligned_r1_dapi_mask.scale, opacity=0.5)
            
            seg_dir = Path(output_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_save_path = seg_dir / f"{layer_name}.tif"
            tifffile.imwrite(mask_save_path, merged_mask)
            session.set_processed_file(layer_name, str(mask_save_path), layer_type='labels', metadata={'subtype': 'consensus_mask'})

        dialog.update_progress(95, "Cleaning up...")
        if hide_raw:
            for layer in viewer.layers:
                is_result = "Aligned" in layer.name or "Warped" in layer.name or "Consensus" in layer.name
                layer.visible = is_result

        dialog.update_progress(100, "Complete.")
        viewer.status = "Automated preprocessing complete."

# Create the final widget container
automated_preprocessing_widget = Container(labels=False)
header = Label(value="Automated Preprocessing")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Runs segmentation, registration, warping, and nuclei matching in one step.</i>")
automated_preprocessing_widget.extend([header, info, _automated_preprocessing_magic_widget])