import numpy as np
from pathlib import Path
from zfisher.core import session, io, segmentation, registration #

def test_steps_1_through_3(): #
    # 1. SETUP PATHS
    input_r1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2") #
    input_r2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2") #
    output_dir = Path.home() / "zFISHer_Headless_FullTest" #
    
    print(f"--- STEP 1: Starting Headless Session ---") #
    success = session.initialize_new_session( #
        output_dir=output_dir, 
        r1_path=input_r1, 
        r2_path=input_r2,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )
    
    if not success: #
        print("Session already exists. Continuing with existing data...") #

    print(f"\n--- STEP 2: DAPI Mapping (Segmentation) ---") #
    r1_path = Path(output_dir) / "input" / "R1_converted.ome.tif" #
    r2_path = Path(output_dir) / "input" / "R2_converted.ome.tif" #
    
    r1_session = io.load_image_session(r1_path) #
    r2_session = io.load_image_session(r2_path) #
    
    r1_dapi = io.get_channel_data(r1_session) #
    r2_dapi = io.get_channel_data(r2_session) #
    
    seg_results = segmentation.process_session_dapi( #
        r1_data=r1_dapi, 
        r2_data=r2_dapi, 
        output_dir=output_dir,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    # --- NEW STEP 3: Registration ---
    print(f"\n--- STEP 3: Registration (RANSAC) ---") #
    
    # Extract centroids from the segmentation results
    # seg_results format: {'RoundPrefix': (masks, centroids)}
    r1_centroids = seg_results['R1'][1] #
    r2_centroids = seg_results['R2'][1] #

    # Call the core orchestrator for registration
    # This automatically updates the session file with the calculated shift
    shift, rmsd = registration.calculate_session_registration( #
        r1_centroids, 
        r2_centroids, 
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    if shift is not None: #
        print(f"\n✅ SUCCESS: Registration Complete.") #
        print(f"Calculated Shift (Z, Y, X): {shift}") #
        print(f"RMSD: {rmsd:.4f} pixels") #
        print(f"Shift stored in: {output_dir}/zfisher_session.json") #
    
    return seg_results, shift

if __name__ == "__main__":
    test_steps_1_through_3() #