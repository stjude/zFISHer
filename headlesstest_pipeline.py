import numpy as np
from pathlib import Path
import gc
import tifffile
from zfisher.core import session, io, segmentation, registration
from zfisher import constants

def run_headless_alignment_test():
    """
    Validates Steps 1 through 5 of the zFISHer pipeline in a headless environment.
    Updated to handle rigid/deformable fallback and underscore naming.
    """
    # --- STEP 1: SETUP & SESSION INITIALIZATION ---
    input_r1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
    input_r2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")
    output_dir = Path.home() / "zFISHer_Headless_FullTest"
    
    print(f"--- STEP 1: Starting Headless Session ---")
    success = session.initialize_new_session(
        output_dir=output_dir, 
        r1_path=input_r1, 
        r2_path=input_r2,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )
    
    if not success:
        print("Session already exists. Continuing with existing data...")

    # --- STEP 2: DAPI MAPPING (SEGMENTATION) ---
    print(f"\n--- STEP 2: DAPI Mapping (Segmentation) ---")
    r1_path = output_dir / "input" / "R1_converted.ome.tif"
    r2_path = output_dir / "input" / "R2_converted.ome.tif"
    
    r1_session = io.load_image_session(r1_path)
    r2_session = io.load_image_session(r2_path)
    
    r1_dapi = io.get_channel_data(r1_session, constants.DAPI_CHANNEL_NAME)
    r2_dapi = io.get_channel_data(r2_session, constants.DAPI_CHANNEL_NAME)
    
    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, 
        r2_data=r2_dapi, 
        output_dir=output_dir,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    # --- STEP 3: REGISTRATION (RANSAC) ---
    print(f"\n--- STEP 3: Registration (RANSAC) ---")
    r1_centroids = seg_results['R1'][1]
    r2_centroids = seg_results['R2'][1]

    shift, rmsd = registration.calculate_session_registration(
        r1_centroids, 
        r2_centroids, 
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    if shift is None:
        print("❌ FAILED: Registration failed. Aborting.")
        return

    # --- STEP 4: GLOBAL CANVAS (WARPING) ---
    print(f"\n--- STEP 4: Global Canvas (Headless Warping) ---")
    r1_layers_data = []
    r2_layers_data = []

    for i, ch_name in enumerate(r1_session.channels):
        r1_layers_data.append({'name': f"R1 - {ch_name}", 'data': r1_session.data[:, i, :, :], 'scale': r1_session.voxels, 'is_label': False})
    for i, ch_name in enumerate(r2_session.channels):
        r2_layers_data.append({'name': f"R2 - {ch_name}", 'data': r2_session.data[:, i, :, :], 'scale': r2_session.voxels, 'is_label': False})

    seg_dir = output_dir / constants.SEGMENTATION_DIR
    for prefix, layer_list in [("R1", r1_layers_data), ("R2", r2_layers_data)]:
        mask_path = seg_dir / f"{prefix}_masks.tif"
        if mask_path.exists():
            layer_list.append({
                'name': "DAPI_masks", # Simplified to match observed output underscores
                'data': tifffile.imread(mask_path), 
                'scale': r1_session.voxels, 
                'is_label': True
            })

    aligned_dir = output_dir / constants.ALIGNED_DIR
    aligned_dir.mkdir(exist_ok=True, parents=True)

    registration.generate_global_canvas(
        r1_layers_data=r1_layers_data, 
        r2_layers_data=r2_layers_data,
        shift=shift, 
        output_dir=aligned_dir, 
        apply_warp=True,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    # --- STEP 5: MATCH NUCLEI (CONSENSUS) ---
    print(f"\n--- STEP 5: Match Nuclei (Headless Consensus) ---")
    
    # Using underscores to match the actual folder output
    r1_aligned_mask = aligned_dir / "Aligned_R1_DAPI_masks.tif"
    
    # Robust fallback: Try Warped first, then Aligned if RANSAC used rigid shift
    r2_warped_mask = aligned_dir / "Warped_R2_DAPI_masks.tif"
    if not r2_warped_mask.exists():
        r2_warped_mask = aligned_dir / "Aligned_R2_DAPI_masks.tif"

    if r1_aligned_mask.exists() and r2_warped_mask.exists():
        print(f"Loading: {r1_aligned_mask.name} and {r2_warped_mask.name}")
        merged_mask, pts1 = segmentation.process_consensus_nuclei(
            mask1=tifffile.imread(r1_aligned_mask),
            mask2=tifffile.imread(r2_warped_mask),
            output_dir=output_dir,
            threshold=20.0,
            method="Intersection", # Defaulted to Intersection
            progress_callback=lambda p, t: print(f"[{p}%] {t}")
        )
        print(f"✅ SUCCESS: Consensus generated with {len(pts1)} matched nuclei.")
    else:
        print(f"❌ FAILED: File search failed.")
        print(f"Missing: {[f.name for f in [r1_aligned_mask, r2_warped_mask] if not f.exists()]}")

    print(f"\n--- ALL STEPS COMPLETE ---")
    print(f"Results saved to: {output_dir}")
    gc.collect()

if __name__ == "__main__":
    try:
        run_headless_alignment_test()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")