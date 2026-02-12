import napari
import numpy as np
from magicgui import magicgui

from zfisher.core.segmentation import match_nuclei_labels, merge_labeled_masks, get_mask_centroids
from .. import popups

@magicgui(
    call_button="Match & Merge Nuclei",
    r1_mask_layer={"label": "R1 Mask (Aligned)"},
    r2_mask_layer={"label": "R2 Mask (Warped)"},
    threshold={"label": "Max Distance (px)", "min": 0, "max": 100, "step": 1}
)
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
        
    viewer.status = "Matching nuclei..."
    dialog = popups.show_busy_popup(viewer.window._qt_window, "Matching Nuclei...")
    
    try:
        # Run matching
        new_mask2, pts1, pts2 = match_nuclei_labels(r1_mask_layer.data, r2_mask_layer.data, threshold=threshold)
        
        # Merge into a single consensus mask
        merged_mask = merge_labeled_masks(r1_mask_layer.data, new_mask2)
        
        # Add merged layer
        viewer.add_labels(
            merged_mask,
            name="Consensus_Nuclei",
            scale=r1_mask_layer.scale,
            opacity=0.5
        )
        
        # Helper to add ID labels
        def add_id_points(pts_data, name, scale):
            if not pts_data: return
            coords = np.array([p['coord'] for p in pts_data])
            labels = np.array([p['label'] for p in pts_data])
            
            viewer.add_points(
                coords,
                name=name,
                size=0, # Invisible points, just text
                scale=scale,
                properties={'label': labels},
                text={'string': '{label}', 'size': 10, 'color': 'cyan', 'translation': np.array([0, -5, 0])},
                blending='translucent_no_depth'
            )

        # Add IDs for the consensus layer
        consensus_pts = get_mask_centroids(merged_mask)
        add_id_points(consensus_pts, "Consensus_IDs", r1_mask_layer.scale)
        
        viewer.status = "Nuclei matched and merged into 'Consensus_Nuclei'."
    finally:
        dialog.close()
