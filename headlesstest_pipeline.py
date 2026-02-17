import numpy as np
from pathlib import Path
import gc
import tifffile
from zfisher.core import session, io, segmentation, registration
from zfisher import constants

def run_headless_alignment_test():
    """
    Validates Steps 1 through 4 of the zFISHer pipeline in a headless environment.
    Updated to include automated loading and warping of segmentation masks.
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
    
    # Building the layer lists manually
    r1_layers_data = []
    r2_layers_data = []

    # 4a. Add Image Channels
    for i, ch_name in enumerate(r1_session.channels):
        r1_layers_data.append({
            'name': f"R1 - {ch_name}",
            'data': r1_session.data[:, i, :, :],
            'scale': r1_session.voxels,
            'is_label': False
        })
    for i, ch_name in enumerate(r2_session.channels):
        r2_layers_data.append({
            'name': f"R2 - {ch_name}",
            'data': r2_session.data[:, i, :, :],
            'scale': r2_session.voxels,
            'is_label': False
        })

    # 4b. Add Masks from Step 2 (The fix for your missing files)
    seg_dir = output_dir / constants.SEGMENTATION_DIR
    for prefix, layer_list in [("R1", r1_layers_data), ("R2", r2_layers_data)]:
        mask_path = seg_dir / f"{prefix}_masks.tif"
        if mask_path.exists():
            print(f"Adding {prefix} masks to warping queue...")
            layer_list.append({
                'name': f"{prefix} - DAPI_masks",
                'data': tifffile.imread(mask_path),
                'scale': r1_session.voxels,
                'is_label': True # Ensures nearest-neighbor interpolation
            })

    aligned_dir = output_dir / constants.ALIGNED_DIR
    aligned_dir.mkdir(exist_ok=True, parents=True)

    # Execute warping
    results = registration.generate_global_canvas(
        r1_layers_data=r1_layers_data,
        r2_layers_data=r2_layers_data,
        shift=shift,
        output_dir=aligned_dir,
        apply_warp=True,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    if results:
        print(f"\n✅ SUCCESS: Full Headless Alignment Complete.")
        print(f"Files saved in: {aligned_dir}")
    
    gc.collect()

if __name__ == "__main__":
    try:
        run_headless_alignment_test()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")