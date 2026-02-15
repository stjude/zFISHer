import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.segmentation import (
    match_nuclei_labels,
    merge_labeled_masks,
    get_mask_centroids,
)
from ... import constants

@magicgui(
    call_button="Match & Merge Nuclei",
    r1_mask_layer={"label": "R1 Mask (Aligned)"},
    r2_mask_layer={"label": "R2 Mask (Warped)"},
    threshold={"label": "Max Distance (px)", "min": 0, "max": 100, "step": 1}
)
@require_active_session("Please start or load a session before matching nuclei.")
@error_handler("Nuclei Matching Failed")
def nuclei_matching_widget(
    r1_mask_layer: "napari.layers.Labels",
    r2_mask_layer: "napari.layers.Labels",
    threshold: float = 20.0
):
    """Matches nuclei between two aligned mask layers and syncs their IDs."""
    viewer = napari.current_viewer()

    if not r1_mask_layer or not r2_mask_layer:
        viewer.status = "Please select both mask layers."
        return
    
    if r1_mask_layer == r2_mask_layer:
        viewer.status = "Error: Same layer selected for both. Please select R1 for the first and R2 for the second."
        return
        
    with popups.ProgressDialog(viewer.window._qt_window, "Matching Nuclei...") as dialog:
        viewer.status = "Matching nuclei..."
        
        # 1. Call core segmentation functions
        new_mask2, pts1, pts2 = match_nuclei_labels(r1_mask_layer.data, r2_mask_layer.data, threshold=threshold)
        merged_mask = merge_labeled_masks(r1_mask_layer.data, new_mask2)
        
        # 2. Pass results to UI helper to handle layer creation and saving
        viewer_helpers.add_consensus_nuclei_to_viewer(viewer, r1_mask_layer, merged_mask, pts1)
        
        viewer.status = f"Matched {len(pts1) if pts1 else 0} nuclei."
        dialog.update_progress(100, "Done.")