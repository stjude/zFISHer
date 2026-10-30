import logging
import napari
import numpy as np
import tifffile
import gc
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

logger = logging.getLogger(__name__)

from ...core import session, segmentation, registration, puncta, stages
from ...core.params import ConsensusParams, RegistrationParams, default, gui_option
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.registration import generate_global_canvas
from ... import constants


def _get_qt_parent(viewer):
    """Get the Qt parent window for dialogs without using deprecated napari internals."""
    try:
        from qtpy.QtWidgets import QApplication
        return QApplication.activeWindow()
    except Exception as e:
        logger.debug("Could not get Qt parent window: %s", e)
        return None

@magicgui(
    call_button="Run Automated Registration && Warping",
    r1_dapi_layer={"label": "Round 1 Nuclei Layer", "tooltip": "Select the Round 1 nuclear stain layer for registration."},
    r2_dapi_layer={"label": "Round 2 Nuclei Layer", "tooltip": "Select the Round 2 nuclear stain layer for registration."},
    max_distance=gui_option(RegistrationParams, "max_ransac_distance"),
    apply_warp=gui_option(RegistrationParams, "apply_warp"),
    match_nuclei={"label": "Create Consensus Nuclei Mask", "tooltip": "Create a merged nuclei mask by matching R1 and R2 nuclei after alignment."},
    overlap_method=gui_option(ConsensusParams, "overlap_method", widget_type="RadioButtons", orientation="horizontal"),
    match_threshold=gui_option(ConsensusParams, "match_threshold"),
    remove_outliers=gui_option(ConsensusParams, "remove_extranuclear_puncta"),
    remove_border=gui_option(ConsensusParams, "remove_border_nuclei"),
    show_checkerboard={"label": "Show Checkerboard", "value": True, "tooltip": "Show a checkerboard pattern to visually assess alignment quality."},
    show_deformation={"label": "Show Deformation Field", "value": True, "tooltip": "Show a vector field visualizing how the tissue was warped."},
    hide_raw={"label": "Hide Raw Layers After?", "tooltip": "Hide raw input layers after processing, showing only aligned results."}
)
@require_active_session("Please start or load a session first.")
@error_handler("Automated Registration Failed")
def _automated_preprocessing_magic_widget(
    r1_dapi_layer: "napari.layers.Image",
    r2_dapi_layer: "napari.layers.Image",
    max_distance: int = default(RegistrationParams, "max_ransac_distance"),
    apply_warp: bool = default(RegistrationParams, "apply_warp"),
    match_nuclei: bool = True,
    overlap_method: str = default(ConsensusParams, "overlap_method"),
    match_threshold: int = default(ConsensusParams, "match_threshold"),
    remove_outliers: bool = default(ConsensusParams, "remove_extranuclear_puncta"),
    remove_border: bool = default(ConsensusParams, "remove_border_nuclei"),
    show_checkerboard: bool = True,
    show_deformation: bool = True,
    hide_raw: bool = False
):
    viewer = napari.current_viewer()
    if not r1_dapi_layer or not r2_dapi_layer:
        viewer.status = "Please select both nuclei layers."
        return

    with popups.ProgressDialog(_get_qt_parent(viewer), "Automated Registration...") as dialog:
        output_dir = Path(session.get_data("output_dir"))
        voxels = tuple(r1_dapi_layer.scale)

        # === STEP 1: RETRIEVE EXISTING CENTROIDS ===
        # Nuclei segmentation is performed earlier in the pipeline (Nuclei Mapping).
        # Pull centroids from the existing viewer layers for registration.
        r1_centroid_name = f"{r1_dapi_layer.name}{constants.CENTROIDS_SUFFIX}"
        r2_centroid_name = f"{r2_dapi_layer.name}{constants.CENTROIDS_SUFFIX}"

        r1_centroids_layer = viewer.layers[r1_centroid_name] if r1_centroid_name in viewer.layers else None
        r2_centroids_layer = viewer.layers[r2_centroid_name] if r2_centroid_name in viewer.layers else None

        if r1_centroids_layer is None or r2_centroids_layer is None:
            missing = []
            if r1_centroids_layer is None:
                missing.append(r1_centroid_name)
            if r2_centroids_layer is None:
                missing.append(r2_centroid_name)
            popups.show_error_popup(
                None, "Missing Nuclei Segmentation",
                f"Run Nuclei Mapping first. Missing layers: {', '.join(missing)}"
            )
            return

        r1_centroids = np.array(r1_centroids_layer.data)
        r2_centroids = np.array(r2_centroids_layer.data)
        dialog.update_progress(5, "Found existing nuclei centroids.")

        # === STEP 2: REGISTRATION ===
        shift, _ = registration.calculate_session_registration(
            r1_centroids, r2_centroids,
            voxels=voxels,
            max_distance=max_distance,
            progress_callback=lambda p, t: dialog.update_progress(5 + int(p * 0.20), t)
        )
        if shift is None:
            popups.show_error_popup(
                None, "Registration Failed",
                "Could not calculate a valid shift. Check that both rounds have sufficient nuclear signal."
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
                # The image as read, like batch: a float32 copy made the aligned
                # TIFFs differ from batch's in type.
                'data': np.asarray(l.data).astype(np.uint32) if is_label else np.asarray(l.data),
                'colormap': l.colormap.name if hasattr(l, 'colormap') else 'gray',
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
            apply_warp=apply_warp,
            progress_callback=lambda p, t: dialog.update_progress(35 + int(p * 0.30), t),
            # Rounds whose nuclear channels are named differently (2.1.2's batch fix).
            r2_channel_names=stages.nuclear_channel_names(),
        )
        session.update_data("canvas_scale", voxels)
        r1_layers_data.clear(); r2_layers_data.clear(); gc.collect()

        # Freeze canvas for all remaining layer mutations (canvas results,
        # consensus, puncta transform).  __exit__ will unfreeze.
        dialog.freeze_canvas()

        # Calculate world-space translate from canvas offset for aligned layers
        canvas_translate = np.array(canvas_offset) * np.array(voxels) if canvas_offset is not None else None
        translate_arg = tuple(canvas_translate) if canvas_translate is not None else None

        # Remove existing aligned/warped layers to prevent duplicates on rerun
        existing_names = {layer_info['name'] for layer_info in results}
        for old_layer in list(viewer.layers):
            if old_layer.name in existing_names:
                old_layer._locked = False
                try:
                    viewer.layers.remove(old_layer)
                except ValueError:
                    pass

        n_results = max(len(results), 1)
        for i, layer_info in enumerate(results):
            pct = 65 + int(((i + 1) / n_results) * 5)
            dialog.update_progress(pct, f"Loading layer: {layer_info['name']}...")

            layer_type = layer_info['type']
            meta = layer_info['meta']

            # Skip checkerboard/deformation if user disabled them
            if not show_checkerboard and constants.WARPED_CHECKERBOARD_NAME in layer_info['name']:
                continue
            if not show_deformation and constants.DEFORMATION_FIELD_NAME in layer_info['name']:
                continue

            # Skip adding aligned/warped mask labels to the viewer — they are
            # only needed on disk for the consensus step, which reads them
            # directly.  Adding them here triggers duplicate _IDs layers.
            if layer_type == 'labels' and constants.MASKS_SUFFIX in layer_info['name']:
                continue
            if layer_type == 'labels':
                lyr = viewer.add_labels(
                    layer_info['data'].astype(np.uint32), name=layer_info['name'],
                    scale=meta['scale'], translate=translate_arg, opacity=0.6, visible=False,
                )
                lyr.rendering = 'iso_categorical'
            elif layer_type == 'image':
                viewer.add_image(
                    layer_info['data'], name=layer_info['name'],
                    colormap=meta.get('colormap', 'gray'), scale=meta['scale'],
                    translate=translate_arg,
                    blending=meta.get('blending', 'additive'), opacity=meta.get('opacity', 1.0),
                    visible=False,
                )
            elif layer_type == 'vectors':
                viewer.add_vectors(
                    layer_info['data'], name=layer_info['name'],
                    scale=meta['scale'], translate=translate_arg,
                    edge_width=0.2, length=2.5, edge_color='cyan', visible=False,
                )

        # === STEP 4: CONSENSUS NUCLEI ===
        merged_mask = None
        if match_nuclei:
            nuc_ch = session.get_nuclear_channel()
            r1_mask_path = aligned_dir / f"Aligned_R1_{nuc_ch}{constants.MASKS_SUFFIX}.tif"
            r2_mask_path = aligned_dir / f"Warped_R2_{nuc_ch}{constants.MASKS_SUFFIX}.tif"
            if not r2_mask_path.exists():
                r2_mask_path = aligned_dir / f"Aligned_R2_{nuc_ch}{constants.MASKS_SUFFIX}.tif"

            if not (r1_mask_path.exists() and r2_mask_path.exists()):
                popups.show_error_popup(
                    None, "Consensus Failed",
                    "Could not find aligned nuclei masks. Canvas generation may have failed."
                )
            else:
                merged_mask, pts1 = segmentation.process_consensus_nuclei(
                    mask1=tifffile.imread(r1_mask_path),
                    mask2=tifffile.imread(r2_mask_path),
                    output_dir=output_dir,
                    threshold=match_threshold or None,
                    method=overlap_method,
                    voxel_size=voxels,
                    remove_border=remove_border,
                    progress_callback=lambda p, t: dialog.update_progress(70 + int(p * 0.25), t)
                )

                # Build a lightweight reference for scale/translate
                translate = tuple(np.array(canvas_offset) * np.array(voxels)) if canvas_offset is not None else (0,) * len(voxels)
                ref = type('_ref', (), {
                    'scale': voxels,
                    'translate': translate,
                })()
                viewer_helpers.add_consensus_nuclei_to_viewer(viewer, ref, merged_mask, pts1)

        # === STEP 5: TRANSFORM EXISTING PUNCTA LAYERS ===
        # Lock all puncta layers before transform to prevent deletion mid-process
        from .. import events as _events
        for l in viewer.layers:
            if isinstance(l, napari.layers.Points) and constants.PUNCTA_SUFFIX in l.name:
                _events.lock_layer(l)
        # Find any raw puncta Points layers in the viewer and transform them
        # into aligned/warped space, renaming them accordingly.
        puncta_layers = [
            l for l in list(viewer.layers)
            if isinstance(l, napari.layers.Points)
            and constants.PUNCTA_SUFFIX in l.name
            and constants.ALIGNED_PREFIX not in l.name
            and constants.WARPED_PREFIX not in l.name
        ]
        transformed_names = viewer_helpers.transform_raw_puncta_layers(
            viewer, puncta_layers, voxels, remove_extranuclear=bool(remove_outliers and match_nuclei),
            consensus_mask=merged_mask if match_nuclei else None, assign_nuclei=bool(match_nuclei),
            progress=lambda pct, text: dialog.update_progress(92 + int(pct * 0.06), text))

        # Lock all remaining puncta layers after warping — no further deletion
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Points) and constants.PUNCTA_SUFFIX in layer.name:
                _events.lock_layer(layer)

        # === STEP 6: REASSIGN NUCLEUS IDs & REMOVE OUTLIERS ===
        if match_nuclei and merged_mask is not None:
            import pandas as pd
            consensus_layer = next(
                (l for l in viewer.layers
                 if isinstance(l, napari.layers.Labels)
                 and constants.CONSENSUS_MASKS_NAME in l.name),
                None
            )
            consensus_scale = np.array(consensus_layer.scale) if consensus_layer else np.ones(3)
            consensus_translate = np.array(consensus_layer.translate) if consensus_layer else np.zeros(3)

            # The channels just warped already have their nuclei, as batch gives them.
            aligned_puncta = [
                l for l in list(viewer.layers)
                if isinstance(l, napari.layers.Points)
                and constants.PUNCTA_SUFFIX in l.name
                and l.name not in transformed_names
            ]
            removed_total = 0
            for pts_layer in aligned_puncta:
                dialog.update_progress(97, f"Reassigning nuclei: {pts_layer.name}...")
                coords = np.array(pts_layer.data)
                if len(coords) == 0:
                    continue
                pts_scale = np.array(pts_layer.scale)
                pts_translate = np.array(pts_layer.translate)
                world_coords = coords * pts_scale + pts_translate
                voxel_coords = np.round((world_coords - consensus_translate) / consensus_scale).astype(int)
                mask_shape = np.array(merged_mask.shape)
                in_bounds = np.all((voxel_coords >= 0) & (voxel_coords < mask_shape), axis=1)
                clipped = np.clip(voxel_coords, 0, mask_shape - 1)
                new_ids = merged_mask[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
                new_ids[~in_bounds] = 0

                features = pts_layer.features.copy() if hasattr(pts_layer, 'features') and not pts_layer.features.empty else pd.DataFrame()
                if not features.empty and 'Nucleus_ID' in features.columns:
                    features['Nucleus_ID'] = new_ids

                if remove_outliers:
                    # Puncta placed by hand stay where they were put, as in batch.
                    inside_mask = (new_ids > 0) | viewer_helpers.manual_mask(features)
                    n_removed = int((~inside_mask).sum())
                    removed_total += n_removed
                    coords = coords[inside_mask]
                    if not features.empty:
                        features = features[inside_mask].reset_index(drop=True)

                pts_layer._locked = False
                viewer_helpers.set_points_data(pts_layer, coords)
                if not features.empty:
                    pts_layer.features = features
                pts_layer._locked = True

                # Re-save
                out_dir = session.get_data("output_dir")
                if out_dir:
                    reports_dir = Path(out_dir) / constants.REPORTS_DIR
                    reports_dir.mkdir(exist_ok=True, parents=True)
                    csv_path = constants.puncta_csv_path(reports_dir, pts_layer.name)
                    puncta.write_puncta_csv(csv_path, coords, features)
                    session.set_processed_file(
                        pts_layer.name, str(csv_path),
                        layer_type='points', metadata={'subtype': 'puncta_csv'}
                    )

        if hide_raw:
            nuc_ch = session.get_nuclear_channel()
            r1_nuc = f"{constants.ALIGNED_PREFIX} R1 - {nuc_ch}"
            r2_warped = f"{constants.WARPED_PREFIX} R2 - {nuc_ch}"
            r2_aligned = f"{constants.ALIGNED_PREFIX} R2 - {nuc_ch}"
            show = {r1_nuc, r2_warped, r2_aligned}
            for layer in viewer.layers:
                layer.visible = layer.name in show

        dialog.update_progress(100, "Complete.")
        viewer.status = "Automated Registration Complete."

# UI Wrapper
from qtpy.QtWidgets import QLabel, QSizePolicy
from ._shared import make_divider as _make_divider, make_section_header as _make_section_header, make_section_desc as _make_section_desc, make_spacer as _make_spacer

# Insert section headers into the magicgui form
# Widget order: r1_dapi(0), r2_dapi(1), match_nuclei(2), hide_raw(3), call_button(4)
_inner = _automated_preprocessing_magic_widget.native.layout()

_input_header = _make_section_header("Input Layers")
_input_desc = _make_section_desc("Select the nuclear stain layers from each round for registration.")
_inner.insertWidget(0, _input_header)
_inner.insertWidget(1, _input_desc)
# r1_dapi(2), r2_dapi(3)

_inner.insertWidget(4, _make_spacer())
_inner.insertWidget(5, _make_divider())
_options_header = _make_section_header("Options")
_options_desc = _make_section_desc("Configure consensus nuclei generation and raw layer visibility.")
_inner.insertWidget(6, _options_header)
_inner.insertWidget(7, _options_desc)
# match_nuclei(8), hide_raw(9), call_button(10)

_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)
_automated_preprocessing_magic_widget.native.setMinimumWidth(0)
for child in _automated_preprocessing_magic_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox
for child in _automated_preprocessing_magic_widget.native.findChildren(QAbstractSpinBox) + _automated_preprocessing_magic_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)


class _AutomatedPreprocessingContainer(Container):
    _automated_preprocessing_magic_widget = None
    def reset_choices(self):
        _automated_preprocessing_magic_widget.reset_choices()

automated_preprocessing_widget = _AutomatedPreprocessingContainer(labels=False)
automated_preprocessing_widget._automated_preprocessing_magic_widget = _automated_preprocessing_magic_widget
header = Label(value="Automated Registration & Warping")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Registration, warping, and consensus nuclei.</i>")
info.native.setObjectName("widgetInfo")
info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

_layout = automated_preprocessing_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_automated_preprocessing_magic_widget.native)
_layout.addStretch(1)