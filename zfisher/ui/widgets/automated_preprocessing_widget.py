import napari
import numpy as np
import tifffile
import gc
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, segmentation, registration, puncta
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.registration import generate_global_canvas
from ... import constants


def _get_qt_parent(viewer):
    """Get the Qt parent window for dialogs without using deprecated napari internals."""
    try:
        from qtpy.QtWidgets import QApplication
        return QApplication.activeWindow()
    except Exception:
        return None

@magicgui(
    call_button="Run Automated Preprocessing",
    r1_dapi_layer={"label": "Round 1 DAPI Layer"},
    r2_dapi_layer={"label": "Round 2 DAPI Layer"},
    match_nuclei={"label": "Create Consensus Nuclei Mask"},
    hide_raw={"label": "Hide Raw Layers After?"}
)
@require_active_session("Please start or load a session first.")
@error_handler("Automated Preprocessing Failed")
def _automated_preprocessing_magic_widget(
    r1_dapi_layer: "napari.layers.Image",
    r2_dapi_layer: "napari.layers.Image",
    match_nuclei: bool = True,
    hide_raw: bool = True
):
    viewer = napari.current_viewer()
    if not r1_dapi_layer or not r2_dapi_layer:
        viewer.status = "Please select both DAPI layers."
        return

    with popups.ProgressDialog(_get_qt_parent(viewer), "Automated Preprocessing...") as dialog:
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

        results, bspline_transform, canvas_offset = generate_global_canvas(
            r1_layers_data, r2_layers_data, shift, aligned_dir,
            apply_warp=True,
            progress_callback=lambda p, t: dialog.update_progress(35 + int(p * 0.30), t)
        )
        session.update_data("canvas_scale", voxels)
        r1_layers_data.clear(); r2_layers_data.clear(); gc.collect()

        # Calculate world-space translate from canvas offset for aligned layers
        canvas_translate = np.array(canvas_offset) * np.array(voxels) if canvas_offset is not None else None
        translate_arg = tuple(canvas_translate) if canvas_translate is not None else None

        n_results = max(len(results), 1)
        for i, layer_info in enumerate(results):
            pct = 65 + int(((i + 1) / n_results) * 5)
            dialog.update_progress(pct, f"Loading layer: {layer_info['name']}...")

            layer_type = layer_info['type']
            meta = layer_info['meta']
            if layer_type == 'labels':
                lyr = viewer.add_labels(
                    layer_info['data'].astype(np.uint32), name=layer_info['name'],
                    scale=meta['scale'], translate=translate_arg, opacity=0.6
                )
                lyr.rendering = 'iso_categorical'
            elif layer_type == 'image':
                viewer.add_image(
                    layer_info['data'], name=layer_info['name'],
                    colormap=meta.get('colormap', 'gray'), scale=meta['scale'],
                    translate=translate_arg,
                    blending=meta.get('blending', 'additive'), opacity=meta.get('opacity', 1.0)
                )
            elif layer_type == 'vectors':
                viewer.add_vectors(
                    layer_info['data'], name=layer_info['name'],
                    scale=meta['scale'], translate=translate_arg,
                    edge_width=0.2, length=2.5, edge_color='cyan'
                )

        # === STEP 4: CONSENSUS NUCLEI ===
        merged_mask = None
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
                    threshold=0,  # Auto-determine from distance distribution
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
                    # Compute world-space translate from canvas offset
                    translate = np.array(canvas_offset) * np.array(voxels) if canvas_offset is not None else (0,) * len(voxels)
                    lyr = viewer.add_labels(
                        merged_mask, name=constants.CONSENSUS_MASKS_NAME,
                        scale=voxels, translate=translate, opacity=0.5
                    )
                    lyr.rendering = 'iso_categorical'

        # === STEP 5: TRANSFORM EXISTING PUNCTA LAYERS ===
        # Find any raw puncta Points layers in the viewer and transform them
        # into aligned/warped space, renaming them accordingly.
        puncta_layers = [
            l for l in list(viewer.layers)
            if isinstance(l, napari.layers.Points)
            and constants.PUNCTA_SUFFIX in l.name
            and constants.ALIGNED_PREFIX not in l.name
            and constants.WARPED_PREFIX not in l.name
        ]
        if puncta_layers:
            import pandas as pd
            reports_dir = output_dir / constants.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)
            n_puncta = max(len(puncta_layers), 1)
            for pi, pts_layer in enumerate(puncta_layers):
                pct = 92 + int(((pi + 1) / n_puncta) * 6)
                dialog.update_progress(pct, f"Transforming puncta: {pts_layer.name}...")

                # Determine round from layer name
                name_upper = pts_layer.name.upper()
                if "R1" in name_upper:
                    round_id = "R1"
                elif "R2" in name_upper:
                    round_id = "R2"
                else:
                    continue

                # Build raw_puncta array: Z, Y, X, Nucleus_ID, Intensity, SNR
                coords = np.array(pts_layer.data)
                if len(coords) == 0:
                    continue
                feats = pts_layer.features if hasattr(pts_layer, 'features') and isinstance(pts_layer.features, pd.DataFrame) and not pts_layer.features.empty else None
                if feats is not None and len(feats) == len(coords):
                    nuc_ids = feats['Nucleus_ID'].values if 'Nucleus_ID' in feats.columns else np.zeros(len(coords))
                    intensity = feats['Intensity'].values if 'Intensity' in feats.columns else np.zeros(len(coords))
                    snr = feats['SNR'].values if 'SNR' in feats.columns else np.zeros(len(coords))
                else:
                    nuc_ids = np.zeros(len(coords))
                    intensity = np.zeros(len(coords))
                    snr = np.zeros(len(coords))
                raw_puncta = np.column_stack([coords, nuc_ids, intensity, snr])

                prefix_str = constants.ALIGNED_PREFIX if round_id == "R1" else constants.WARPED_PREFIX
                base_name = pts_layer.name.replace(constants.PUNCTA_SUFFIX, "")
                aligned_layer_name = f"{prefix_str} {base_name.strip()}{constants.PUNCTA_SUFFIX}"
                csv_out = reports_dir / f"{aligned_layer_name.replace(' ', '_')}.csv"

                transformed = puncta.transform_puncta_to_aligned_space(
                    raw_puncta=raw_puncta,
                    round_id=round_id,
                    shift=shift,
                    canvas_offset=canvas_offset,
                    bspline_transform=bspline_transform if round_id == "R2" else None,
                    consensus_mask=merged_mask if match_nuclei else None,
                    output_path=csv_out,
                    layer_name=aligned_layer_name,
                )

                if transformed is not None and len(transformed) > 0:
                    ref = type('_ref', (), {
                        'name': aligned_layer_name.replace(constants.PUNCTA_SUFFIX, ''),
                        'scale': voxels,
                        'translate': (0,) * len(voxels),
                    })()
                    viewer_helpers.add_or_update_puncta_layer(viewer, ref, transformed)

                # Remove the original raw puncta layer now that it's been transformed
                try:
                    viewer.layers.remove(pts_layer)
                except ValueError:
                    pass

        if hide_raw:
            for layer in viewer.layers:
                layer.visible = any(x in layer.name for x in ["Aligned", "Warped", "Consensus"])

        dialog.update_progress(100, "Complete.")
        viewer.status = "Automated Preprocessing Complete."

# UI Wrapper
from qtpy.QtWidgets import QFrame
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

# Insert "Options" header into the magicgui form before match_nuclei
# Widget order: r1_dapi(0), r2_dapi(1), match_nuclei(2), hide_raw(3), call_button(4)
_inner = _automated_preprocessing_magic_widget.native.layout()
_options_header = Label(value="<b>Options:</b>")
_inner.insertWidget(2, _make_divider())
_inner.insertWidget(3, _options_header.native)

class _AutomatedPreprocessingContainer(Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    _automated_preprocessing_magic_widget = None

    def reset_choices(self):
        _automated_preprocessing_magic_widget.reset_choices()

automated_preprocessing_widget = _AutomatedPreprocessingContainer(labels=False)
automated_preprocessing_widget._automated_preprocessing_magic_widget = _automated_preprocessing_magic_widget
header = Label(value="Automated Preprocessing")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Segmentation, Registration, Warping, and Consensus Nuclei.</i>")
info.native.setObjectName("widgetInfo")

_layout = automated_preprocessing_widget.native.layout()
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_automated_preprocessing_magic_widget.native)