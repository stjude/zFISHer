import napari
import numpy as np
import pandas as pd
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, segmentation
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_header_divider

@magicgui(
    call_button="Match & Merge Nuclei",
    layout="vertical",
    r1_mask_layer={"label": "R1 Mask (Aligned)", "tooltip": "The segmented nuclei mask from Round 1."},
    r2_mask_layer={"label": "R2 Mask (Warped)", "tooltip": "The segmented nuclei mask from Round 2 (aligned or warped)."},
    method={
        "label": "Overlap Method",
        "widget_type": "RadioButtons", # Changes dropdown to toggle/radio buttons
        "choices": ["Intersection", "Union"], # Intersection is now first
        "orientation": "horizontal", # Places buttons side-by-side
        "tooltip": "Intersection: Keep only overlapping pixels. Union: Keep all pixels from both rounds."
    },
    remove_outliers={"label": "Remove Extranuclear Puncta", "tooltip": "Remove puncta that fall outside the consensus mask boundaries."},
)
@require_active_session("Please start or load a session before matching nuclei.")
@error_handler("Nuclei Matching Failed")
def _nuclei_matching_widget(
    r1_mask_layer: "napari.layers.Labels",
    r2_mask_layer: "napari.layers.Labels",
    method: str = "Intersection", # Intersection is now the default
    remove_outliers: bool = True,
):
    """
    Matches nuclei between two aligned mask layers and syncs their IDs.
    Refactored to delegate math and saving to core.segmentation.
    """
    viewer = napari.current_viewer()

    # 1. Validation Logic
    if not r1_mask_layer or not r2_mask_layer:
        viewer.status = "Please select both mask layers."
        return
    
    if r1_mask_layer == r2_mask_layer:
        viewer.status = "Error: Same layer selected for both."
        return
        
    # 2. Execution with Progress Feedback
    with popups.ProgressDialog(viewer.window._qt_window, "Matching Nuclei...") as dialog:
        viewer.status = f"Matching nuclei ({method})..."
        
        output_dir = session.get_data("output_dir")

        # 3. Call the Core Orchestrator
        merged_mask, pts1 = segmentation.process_consensus_nuclei(
            mask1=r1_mask_layer.data,
            mask2=r2_mask_layer.data,
            output_dir=output_dir,
            threshold=0,  # Auto-determine from distance distribution
            method=method,
            progress_callback=lambda p, m: dialog.update_progress(p, m)
        )
        
        # 4. Update the Viewer
        viewer_helpers.add_consensus_nuclei_to_viewer(
            viewer,
            r1_mask_layer,
            merged_mask,
            pts1
        )

        # 5. Reassign nucleus IDs and optionally remove extranuclear puncta
        removed_total = 0
        if merged_mask is not None:
            consensus_layer = next(
                (l for l in viewer.layers
                 if isinstance(l, napari.layers.Labels)
                 and constants.CONSENSUS_MASKS_NAME in l.name),
                None
            )
            consensus_scale = np.array(consensus_layer.scale) if consensus_layer else np.ones(3)
            consensus_translate = np.array(consensus_layer.translate) if consensus_layer else np.zeros(3)

            puncta_layers = [
                l for l in list(viewer.layers)
                if isinstance(l, napari.layers.Points)
                and constants.PUNCTA_SUFFIX in l.name
            ]
            for pts_layer in puncta_layers:
                dialog.update_progress(90, f"Reassigning nuclei: {pts_layer.name}...")
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

                # Filter out extranuclear puncta if toggled on
                if remove_outliers:
                    inside_mask = new_ids > 0
                    n_removed = int((~inside_mask).sum())
                    removed_total += n_removed
                    coords = coords[inside_mask]
                    if not features.empty:
                        features = features[inside_mask].reset_index(drop=True)

                pts_layer.data = coords
                if not features.empty:
                    pts_layer.features = features

                # Re-save the updated CSV
                out_dir = session.get_data("output_dir")
                if out_dir:
                    reports_dir = Path(out_dir) / constants.REPORTS_DIR
                    reports_dir.mkdir(exist_ok=True, parents=True)
                    csv_path = reports_dir / f"{pts_layer.name}.csv"
                    coords_df = pd.DataFrame(coords, columns=['Z', 'Y', 'X'])
                    full_df = pd.concat([features.reset_index(drop=True), coords_df], axis=1)
                    full_df.to_csv(csv_path, index=False)

        # Hide the input mask layers so only the consensus is visible
        r1_mask_layer.visible = False
        r2_mask_layer.visible = False

        outlier_msg = f" Removed {removed_total} extranuclear puncta." if removed_total > 0 else ""
        viewer.status = f"Matched {len(pts1) if pts1 else 0} nuclei using {method}.{outlier_msg}"
        dialog.update_progress(100, "Done.")

# --- UI Wrapper ---
class _NucleiMatchingContainer(Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    def reset_choices(self):
        _nuclei_matching_widget.reset_choices()

nuclei_matching_widget = _NucleiMatchingContainer(labels=False)
nuclei_matching_widget._nuclei_matching_widget = _nuclei_matching_widget
header = Label(value="Match Nuclei")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Matches nuclei between rounds.</i>")
info.native.setObjectName("widgetInfo")
nuclei_matching_widget.extend([header, info, make_header_divider(), _nuclei_matching_widget])
_nm_layout = nuclei_matching_widget.native.layout()
_nm_layout.setSpacing(2)
_nm_layout.setContentsMargins(0, 0, 0, 0)
_nm_layout.addStretch(1)