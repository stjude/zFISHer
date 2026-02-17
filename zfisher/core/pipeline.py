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

def generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp=True):


    """
    Orchestrates the alignment and warping of multiple channels, yielding progress.
    
    Yields:
    ------
    tuple
        A tuple containing (progress_int, message_str, result_dict_or_None).
        The result dictionary contains the processed layer data for napari.
    """
    yield 0, "Matching channels...", None
    aligned_pairs = {}
    for r1 in r1_layers_data:
        channel_name = r1['name'].split("-")[-1].strip()
        r2 = next((l for l in r2_layers_data if channel_name in l['name']), None)
        
        if r2:
            is_label = r1.get('is_label', False)
            aligned_r1, aligned_r2 = align_and_pad_images(r1['data'], r2['data'], shift, is_label=is_label)
            aligned_pairs[channel_name] = (aligned_r1, aligned_r2, r1, r2)

    transform = None
    has_dapi = constants.DAPI_CHANNEL_NAME in aligned_pairs
    
    if apply_warp and has_dapi:
        yield 10, "Calculating deformable registration on DAPI...", None
        yield 11, "(This may take several minutes for large images)", None
        dapi_r1, dapi_r2, _, _ = aligned_pairs[constants.DAPI_CHANNEL_NAME]
        transform = calculate_deformable_transform(dapi_r1, dapi_r2)
        yield 40, "Deformable registration complete.", None
    elif apply_warp and not has_dapi:
        print("Warning: No DAPI channel found. Skipping deformable registration.")
        yield 10, "No DAPI channel; skipping deformable warp.", None

    def warp_worker(item):
        channel_name, (r1_data, r2_data, r1_meta, r2_meta) = item
        final_r2 = r2_data
        r2_name_prefix = constants.ALIGNED_PREFIX
        is_label = r1_meta.get('is_label', False)
        
        if transform:
            final_r2 = apply_deformable_transform(r2_data, transform, r1_data, is_label=is_label)
            r2_name_prefix = constants.WARPED_PREFIX
            
        if output_dir:
            try:
                out_name_r1 = output_dir / f"{constants.ALIGNED_PREFIX}_R1_{channel_name}.tif"
                out_name_r2 = output_dir / f"{r2_name_prefix}_R2_{channel_name}.tif"
                tifffile.imwrite(out_name_r1, r1_data)
                tifffile.imwrite(out_name_r2, final_r2)
                set_processed_file(f"{constants.ALIGNED_PREFIX} R1 - {channel_name}", str(out_name_r1), layer_type='labels' if is_label else 'image')
                set_processed_file(f"{r2_name_prefix} R2 - {channel_name}", str(out_name_r2), layer_type='labels' if is_label else 'image')
            except OSError as e:
                print(f"Error saving {channel_name}: {e}. Check disk space.")
            
        return {
            'r1': {'data': r1_data, 'name': f"{constants.ALIGNED_PREFIX} R1 - {channel_name}", 'colormap': r1_meta['colormap'], 'scale': r1_meta['scale'], 'is_label': is_label},
            'r2': {'data': final_r2, 'name': f"{r2_name_prefix} R2 - {channel_name}", 'colormap': r2_meta['colormap'], 'scale': r2_meta['scale'], 'is_label': is_label}
        }

    num_channels = len(aligned_pairs)
    start_progress = 40 if (apply_warp and has_dapi) else 10
    
    if num_channels > 0:
        for i, item in enumerate(aligned_pairs.items()):
            channel_name, _ = item
            
            # Progress at the START of this channel's processing
            base_progress = start_progress + int((i / num_channels) * (100 - start_progress))
            yield base_progress, f"Processing {channel_name} ({i+1}/{num_channels})...", None

            if transform:
                yield base_progress, f"Applying warp to {channel_name} (can be slow)...", None

            result = warp_worker(item)
            
            # Progress at the END of this channel's processing
            end_progress = start_progress + int(((i + 1) / num_channels) * (100 - start_progress))
            yield end_progress, f"Finished {channel_name}", result
            
            gc.collect()
        
    yield 100, "Canvas generation complete.", None

def generate_aligned_data(fixed_stacks, moving_stacks, shift, deformable=False):
    """
    PURE COMPUTE: Aligns and warps images. 
    Use this for both the UI and the Batch Pipeline.
    """
    results = []
    
    # 1. Rigged alignment/padding
    # 2. If deformable, calculate/apply transform
    # 3. Return the processed arrays
    
    return results