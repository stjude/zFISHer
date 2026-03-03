import napari
import numpy as np
import tifffile
import gc
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, segmentation, registration
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.registration import generate_global_canvas
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
    if not r1_dapi_layer or not r2_dapi_layer:
        viewer.status = "Please select both DAPI layers."
        return

    with popups.ProgressDialog(viewer.window._qt_window, "Automated Preprocessing...") as dialog:
        output_dir = Path(session.get_data("output_dir"))
        voxels = tuple(r1_dapi_layer.scale)

        # === STEP 1: DAPI SEGMENTATION ===
        seg_results = segmentation.process_session_dapi(
            r1_data=r1_dapi_layer.data,
            r2_data=r2_dapi_layer.data,
            output_dir=output_dir,
            progress_callback=lambda p, t: dialog.update_progress(5 + int(p * 0.2), t)
        )

        viewer_helpers.add_segmentation_results_to_viewer(viewer, r1_dapi_layer, seg_results['R1'][0], seg_results['R1'][1])
        viewer_helpers.add_segmentation_results_to_viewer(viewer, r2_dapi_layer, seg_results['R2'][0], seg_results['R2'][1])

        # === STEP 2: REGISTRATION ===
        shift, _ = registration.calculate_session_registration(
            seg_results['R1'][1], seg_results['R2'][1],
            voxels=voxels,
            progress_callback=lambda p, t: dialog.update_progress(25 + int(p * 0.1), t)
        )
        if shift is None:
            popups.show_error_popup(
                None, "Registration Failed",
                "Could not calculate a valid shift. Check that both rounds have sufficient DAPI signal."
            )
            return

        # === STEP 3: GLOBAL CANVAS ===
        r1_layers_data, r2_layers_data = [], []
        for l in viewer.layers:
            if not isinstance(l, (napari.layers.Image, napari.layers.Labels)):
                continue
            if any(x in l.name for x in ["Aligned", "Warped", "Consensus", "_highlight"]):
                continue
            is_label = isinstance(l, napari.layers.Labels)
            layer_info = {
                'name': l.name,
                'data': l.data.astype(np.uint32) if is_label else l.data.astype(np.float32),
                'scale': tuple(l.scale),
                'is_label': is_label,
            }
            if "R1" in l.name:
                r1_layers_data.append(layer_info)
            elif "R2" in l.name:
                r2_layers_data.append(layer_info)

        aligned_dir = output_dir / constants.ALIGNED_DIR
        aligned_dir.mkdir(exist_ok=True, parents=True)

        results = generate_global_canvas(
            r1_layers_data, r2_layers_data, shift, aligned_dir,
            apply_warp=True,
            progress_callback=lambda p, t: dialog.update_progress(35 + int(p * 0.35), t)
        )
        session.update_data("canvas_scale", voxels)
        r1_layers_data.clear(); r2_layers_data.clear(); gc.collect()

        for layer_info in results:
            layer_type = layer_info['type']
            meta = layer_info['meta']
            if layer_type == 'labels':
                lyr = viewer.add_labels(
                    layer_info['data'].astype(np.uint32), name=layer_info['name'],
                    scale=meta['scale'], opacity=0.6
                )
                lyr.rendering = 'iso_categorical'
            elif layer_type == 'image':
                viewer.add_image(
                    layer_info['data'], name=layer_info['name'],
                    colormap=meta.get('colormap', 'gray'), scale=meta['scale'],
                    blending=meta.get('blending', 'additive'), opacity=meta.get('opacity', 1.0)
                )
            elif layer_type == 'vectors':
                viewer.add_vectors(
                    layer_info['data'], name=layer_info['name'],
                    scale=meta['scale'], edge_width=0.2, length=2.5, edge_color='cyan'
                )

        # === STEP 4: CONSENSUS NUCLEI ===
        if match_nuclei:
            r1_mask_path = aligned_dir / f"Aligned_R1_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
            r2_mask_path = aligned_dir / f"Warped_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
            if not r2_mask_path.exists():
                r2_mask_path = aligned_dir / f"Aligned_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"

            if not (r1_mask_path.exists() and r2_mask_path.exists()):
                popups.show_error_popup(
                    None, "Consensus Failed",
                    "Could not find aligned DAPI masks. Canvas generation may have failed."
                )
            else:
                merged_mask, pts1 = segmentation.process_consensus_nuclei(
                    mask1=tifffile.imread(r1_mask_path),
                    mask2=tifffile.imread(r2_mask_path),
                    output_dir=output_dir,
                    threshold=match_threshold,
                    method="Intersection",
                    progress_callback=lambda p, t: dialog.update_progress(70 + int(p * 0.25), t)
                )

                # Use the aligned R1 mask layer as the scale/translate reference
                ref_layer = next(
                    (l for l in viewer.layers
                     if isinstance(l, napari.layers.Labels)
                     and "Aligned" in l.name and "R1" in l.name
                     and constants.DAPI_CHANNEL_NAME in l.name),
                    None
                )
                if ref_layer:
                    viewer_helpers.add_consensus_nuclei_to_viewer(viewer, ref_layer, merged_mask, pts1)
                else:
                    lyr = viewer.add_labels(
                        merged_mask, name=constants.CONSENSUS_MASKS_NAME,
                        scale=voxels, opacity=0.5
                    )
                    lyr.rendering = 'iso_categorical'

        if hide_raw:
            for layer in viewer.layers:
                layer.visible = any(x in layer.name for x in ["Aligned", "Warped", "Consensus"])

        dialog.update_progress(100, "Complete.")
        viewer.status = "Automated Preprocessing Complete."

# UI Wrapper
automated_preprocessing_widget = Container(labels=False)
header = Label(value="Automated Preprocessing")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Segmentation, Registration, Warping, and Consensus Nuclei.</i>")
automated_preprocessing_widget.extend([header, info, _automated_preprocessing_magic_widget])
