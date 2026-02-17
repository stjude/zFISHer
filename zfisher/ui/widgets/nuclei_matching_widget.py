import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from qtpy.QtWidgets import QButtonGroup, QRadioButton # For internal toggle logic if needed

from ...core import session, segmentation 
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Match & Merge Nuclei",
    layout="vertical",
    r1_mask_layer={"label": "R1 Mask (Aligned)"},
    r2_mask_layer={"label": "R2 Mask (Warped)"},
    method={
        "label": "Overlap Method",
        "widget_type": "RadioButtons", # Changes dropdown to toggle/radio buttons
        "choices": ["Intersection", "Union"], # Intersection is now first
        "orientation": "horizontal", # Places buttons side-by-side
        "tooltip": "Intersection: Keep only overlapping pixels. Union: Keep all pixels from both rounds."
    },
    threshold={"label": "Max Distance (px)", "min": 0, "max": 100, "step": 1}
)
@require_active_session("Please start or load a session before matching nuclei.")
@error_handler("Nuclei Matching Failed")
def nuclei_matching_widget(
    r1_mask_layer: "napari.layers.Labels",
    r2_mask_layer: "napari.layers.Labels",
    method: str = "Intersection", # Intersection is now the default
    threshold: float = 20.0
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
            threshold=threshold,
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
        
        viewer.status = f"Matched {len(pts1) if pts1 else 0} nuclei using {method}."
        dialog.update_progress(100, "Done.")