import numpy as np
from pathlib import Path
from zfisher.core import session, io, segmentation

def test_steps_1_and_2():
    # 1. SETUP PATHS
    # Ensure these paths match your local environment
    input_r1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
    input_r2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")
    output_dir = Path.home() / "zFISHer_Headless_FullTest"
    
    print(f"--- STEP 1: Starting Headless Session ---")
    # This creates the directory structure and converts ND2s to OME-TIFFs
    success = session.initialize_new_session(
        output_dir=output_dir, 
        r1_path=input_r1, 
        r2_path=input_r2,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )
    
    if not success:
        print("Session already exists. Continuing with existing data...")

    print(f"\n--- STEP 2: DAPI Mapping (Segmentation) ---")
    
    # Define paths to the converted OME-TIFFs in the 'input' folder
    r1_path = Path(output_dir) / "input" / "R1_converted.ome.tif"
    r2_path = Path(output_dir) / "input" / "R2_converted.ome.tif"
    
    # A. Load the session objects (this keeps metadata like channel names intact)
    r1_session = io.load_image_session(r1_path)
    r2_session = io.load_image_session(r2_path)
    
    # B. EXTRACT 3D DAPI DATA
    # This prevents the 'ValueError' by slicing (Z, C, Y, X) into (Z, Y, X)
    r1_dapi = io.get_channel_data(r1_session) 
    r2_dapi = io.get_channel_data(r2_session)
    
    print(f"Loaded R1 DAPI shape: {r1_dapi.shape}")

    # C. Run the core segmentation orchestrator
    # This runs the watershed math and saves .tif masks and .csv centroids to disk
    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, 
        r2_data=r2_dapi, 
        output_dir=output_dir,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    # 3. VERIFY OUTPUT
    if 'R1' in seg_results:
        masks, centroids = seg_results['R1']
        print(f"\n✅ SUCCESS: Found {len(centroids)} nuclei in R1.")
        print(f"Mask file created: {output_dir}/segmentation/R1_masks.tif")
        print(f"Centroids file created: {output_dir}/segmentation/R1_centroids.csv")
    
    return seg_results

if __name__ == "__main__":
    test_steps_1_and_2()