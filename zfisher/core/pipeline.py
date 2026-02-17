import numpy as np
import tifffile
from pathlib import Path
import gc
import logging
from . import io, registration, segmentation
from .. import constants

from .registration import (
    align_and_pad_images,
    calculate_deformable_transform,
    apply_deformable_transform
)
from .session import set_processed_file
from .. import constants

def run_full_zfisher_pipeline(input_path: Path, output_dir: Path, params: dict):
    """
    Automated backbone covering the steps in the zFISHer UI.
    """
    # 1. Start Session / I/O
    # Handles .nd2 to .tif conversion and directory setup
    session_data = io.initialize_session(input_path, output_dir)
    
    # Load raw data
    r1_session = io.load_image_session(session_data['r1_path'])
    r2_session = io.load_image_session(session_data['r2_path'])
    
    # 2. DAPI Mapping (Segmentation)
    # This ensures the science steps only receive 3D arrays
    r1_dapi = io.get_channel_data(r1_session, constants.DAPI_CHANNEL_NAME)
    r2_dapi = io.get_channel_data(r2_session, constants.DAPI_CHANNEL_NAME)

    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, 
        r2_data=r2_dapi, 
        output_dir=output_dir
    )

    # 3. Run the DAPI orchestrator with the 3D data
    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, 
        r2_data=r2_dapi, 
        output_dir=output_dir,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )
    # 3. Registration
    # Calculates shifts between imaging rounds
    shifts = registration.calculate_all_shifts(session_data['round_paths'])
    
    # 4. Global Canvas (Warping)
    # Applies shifts to create the aligned 4D/5D stacks
    aligned_stacks = registration.apply_transformations(session_data['round_paths'], shifts)
    
    # 5. Match Nuclei
    # Connects masks across rounds to ensure identity consistency
    matched_labels = segmentation.match_nuclei_labels(masks, shifts)
    
    # 6. Puncta Detection
    # The actual science: finding the FISH signals
    puncta_data = puncta.detect_spots(aligned_stacks, matched_labels)
    
    # 7. & 8. Export & Analysis
    # Generates the final CSVs and Colocalization reports
    analysis.generate_reports(puncta_data, output_dir)
    
    return "Pipeline Complete"

def batch_process_directory(input_dir: Path, output_base_dir: Path, params: dict):
    """
    Scans a directory for .nd2 files and processes them sequentially.
    """
    # 1. Identify and pair files (e.g., finding R1 and R2 for the same FOV)
    # This logic depends on your specific naming convention
    file_pairs = io.discover_nd2_pairs(input_dir)
    
    results = []
    
    for pair in file_pairs:
        fov_name = pair['name']
        fov_output = output_base_dir / fov_name
        fov_output.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"--- Starting Batch: {fov_name} ---")
            # 2. Call the full 8-step pipeline we discussed
            status = run_full_zfisher_pipeline(
                r1_path=pair['r1'], 
                r2_path=pair['r2'], 
                output_dir=fov_output,
                params=params
            )
            results.append({"fov": fov_name, "status": "Success"})
            
        except Exception as e:
            # 3. Robust Error Logging
            logging.error(f"Failed to process {fov_name}: {str(e)}")
            results.append({"fov": fov_name, "status": "Failed", "error": str(e)})
            
    return results



######
