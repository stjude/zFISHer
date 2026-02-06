import numpy as np
import tifffile
from pathlib import Path
import concurrent.futures
from zfisher.core.registration import (
    align_and_pad_images,
    calculate_deformable_transform,
    apply_deformable_transform
)
from zfisher.core.session import set_processed_file, save_session

def generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp=True):
    """
    Orchestrates the alignment and warping of multiple channels.
    
    Args:
        r1_layers_data: List of dicts {'name', 'data', 'colormap', 'scale'}
        r2_layers_data: List of dicts (same structure)
        shift: (Z, Y, X) shift vector
        output_dir: Path to save outputs (or None)
        apply_warp: Boolean
        
    Returns:
        List of result dicts to be added to viewer
    """
    # 1. Match Channels
    aligned_pairs = {} # channel: (r1_data, r2_data, r1_meta, r2_meta)
    
    for r1 in r1_layers_data:
        channel_name = r1['name'].split("-")[-1].strip()
        r2 = next((l for l in r2_layers_data if channel_name in l['name']), None)
        
        if r2:
            print(f"Rigid aligning {channel_name}...")
            aligned_r1, aligned_r2 = align_and_pad_images(r1['data'], r2['data'], shift)
            aligned_pairs[channel_name] = (aligned_r1, aligned_r2, r1, r2)

    # 2. Calculate Deformable Transform (on DAPI)
    transform = None
    if apply_warp:
        if "DAPI" in aligned_pairs:
            print("Calculating deformable registration on DAPI...")
            dapi_r1, dapi_r2, _, _ = aligned_pairs["DAPI"]
            transform = calculate_deformable_transform(dapi_r1, dapi_r2)
        else:
            print("Warning: No DAPI channel found. Skipping deformable registration.")

    # 3. Apply Transform (Parallelized)
    def warp_worker(item):
        channel_name, (r1_data, r2_data, r1_meta, r2_meta) = item
        final_r2 = r2_data
        r2_name_prefix = "Aligned"
        
        if transform:
            print(f"Applying warp to {channel_name}...")
            final_r2 = apply_deformable_transform(r2_data, transform, r1_data)
            r2_name_prefix = "Warped"
            
        # Save automatically
        if output_dir:
            out_name_r1 = output_dir / f"Aligned_R1_{channel_name}.tif"
            out_name_r2 = output_dir / f"{r2_name_prefix}_R2_{channel_name}.tif"
            tifffile.imwrite(out_name_r1, r1_data)
            tifffile.imwrite(out_name_r2, final_r2)
            set_processed_file(f"Aligned R1 - {channel_name}", out_name_r1)
            set_processed_file(f"{r2_name_prefix} R2 - {channel_name}", out_name_r2)
            print(f"Saved {out_name_r1}")
            
        return {
            'r1': {'data': r1_data, 'name': f"Aligned R1 - {channel_name}", 'colormap': r1_meta['colormap'], 'scale': r1_meta['scale']},
            'r2': {'data': final_r2, 'name': f"{r2_name_prefix} R2 - {channel_name}", 'colormap': r2_meta['colormap'], 'scale': r2_meta['scale']}
        }

    results = []
    if transform:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(warp_worker, aligned_pairs.items()))
    else:
        results = [warp_worker(item) for item in aligned_pairs.items()]
        
    if output_dir:
        save_session()
        
    return results