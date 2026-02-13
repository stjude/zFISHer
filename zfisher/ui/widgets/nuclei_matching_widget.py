import napari
import numpy as np
import tifffile
from pathlib import Path
from magicgui import magicgui

import zfisher.core.session as session
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
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before matching nuclei."
        )
        return

    if not r1_mask_layer or not r2_mask_layer:
        viewer.status = "Please select both mask layers."
        return
    
    if r1_mask_layer == r2_mask_layer:
        viewer.status = "Error: Same layer selected for both. Please select R1 for the first and R2 for the second."
        return
        
    viewer.status = "Matching nuclei..."
    dialog = popups.ProgressDialog(viewer.window._qt_window, "Matching Nuclei...")
    
    try:
        # Run matching
        new_mask2, pts1, pts2 = match_nuclei_labels(r1_mask_layer.data, r2_mask_layer.data, threshold=threshold)
        
        # Merge into a single consensus mask
        merged_mask = merge_labeled_masks(r1_mask_layer.data, new_mask2)
        
        layer_name = "Consensus_Nuclei_Masks"
        
        # Add merged layer
        viewer.add_labels(
            merged_mask,
            name=layer_name,
            scale=r1_mask_layer.scale,
            opacity=0.5
        )
        
        # Save to disk
        if output_dir:
            try:
                seg_dir = Path(output_dir) / "segmentation"
                seg_dir.mkdir(exist_ok=True, parents=True)
                
                # Save mask
                mask_save_path = seg_dir / f"{layer_name}.tif"
                tifffile.imwrite(mask_save_path, merged_mask)
                session.set_processed_file(layer_name, str(mask_save_path))
                print(f"Saved consensus mask to {mask_save_path}")

                # Save IDs/points
                if pts1:
                    ids_layer_name = f"{layer_name}_IDs"
                    ids_save_path = seg_dir / f"{ids_layer_name}.npy"
                    
                    # Convert to structured array
                    dtype = [('coord', 'f4', 3), ('label', 'i4')]
                    structured_pts = np.array([(p['coord'], p['label']) for p in pts1], dtype=dtype)
                    
                    np.save(ids_save_path, structured_pts)
                    session.set_processed_file(ids_layer_name, str(ids_save_path))
                    print(f"Saved consensus IDs to {ids_save_path}")

                session.save_session()

            except Exception as e:
                print(f"Failed to save consensus data: {e}")
        
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

        add_id_points(pts1, f"{layer_name}_IDs", r1_mask_layer.scale)
        
        viewer.status = f"Matched {len(pts1)} nuclei."
        dialog.update_progress(100, "Done.")
    except Exception as e:
        print(f"Matching failed: {e}")
        viewer.status = "Matching failed."
    finally:
        dialog.close()