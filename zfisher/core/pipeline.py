import numpy as np
import tifffile
from pathlib import Path
import gc
from zfisher.core.registration import (
    align_and_pad_images,
    calculate_deformable_transform,
    apply_deformable_transform
)
from zfisher.core.session import set_processed_file, save_session

def generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp=True):
    """
    Orchestrates the alignment and warping of multiple channels, yielding progress.
    
    Args:
        ... (same as before)
        
    Yields:
        A tuple: (progress_int, message_str, result_dict_or_None)
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
    if apply_warp:
        if "DAPI" in aligned_pairs:
            yield 10, "Calculating deformable registration on DAPI...", None
            dapi_r1, dapi_r2, _, _ = aligned_pairs["DAPI"]
            transform = calculate_deformable_transform(dapi_r1, dapi_r2)
        else:
            print("Warning: No DAPI channel found. Skipping deformable registration.")

    def warp_worker(item):
        channel_name, (r1_data, r2_data, r1_meta, r2_meta) = item
        final_r2 = r2_data
        r2_name_prefix = "Aligned"
        is_label = r1_meta.get('is_label', False)
        
        if transform:
            final_r2 = apply_deformable_transform(r2_data, transform, r1_data, is_label=is_label)
            r2_name_prefix = "Warped"
            
        if output_dir:
            try:
                out_name_r1 = output_dir / f"Aligned_R1_{channel_name}.tif"
                out_name_r2 = output_dir / f"{r2_name_prefix}_R2_{channel_name}.tif"
                tifffile.imwrite(out_name_r1, r1_data)
                tifffile.imwrite(out_name_r2, final_r2)
                set_processed_file(f"Aligned R1 - {channel_name}", str(out_name_r1))
                set_processed_file(f"{r2_name_prefix} R2 - {channel_name}", str(out_name_r2))
            except OSError as e:
                print(f"Error saving {channel_name}: {e}. Check disk space.")
            
        return {
            'r1': {'data': r1_data, 'name': f"Aligned R1 - {channel_name}", 'colormap': r1_meta['colormap'], 'scale': r1_meta['scale'], 'is_label': is_label},
            'r2': {'data': final_r2, 'name': f"{r2_name_prefix} R2 - {channel_name}", 'colormap': r2_meta['colormap'], 'scale': r2_meta['scale'], 'is_label': is_label}
        }

    num_channels = len(aligned_pairs)
    start_progress = 20
    
    if num_channels > 0:
        for i, item in enumerate(aligned_pairs.items()):
            channel_name, _ = item
            progress = start_progress + int(((i + 1) / num_channels) * (100 - start_progress))
            
            yield progress, f"Applying warp to {channel_name} ({i+1}/{num_channels})...", None
            result = warp_worker(item)
            yield progress, f"Applied warp to {channel_name} ({i+1}/{num_channels})", result
            
            gc.collect()
        
    if output_dir:
        save_session()
        
    yield 100, "Canvas generation complete.", None