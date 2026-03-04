import numpy as np
from pathlib import Path
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from skimage import restoration 
from .. import constants

def calculate_spot_quality(image_data, coords, radius=2):
    """
    Calculates Signal-to-Noise Ratio (SNR) and Intensity for each puncta.
    Essential for validating counts in crowded fields.
    """
    stats = []
    # Ensure coords are within image bounds
    z_max, y_max, x_max = image_data.shape
    
    for coord in coords:
        z, y, x = coord.astype(int)
        # Clip to valid image bounds
        z = np.clip(z, 0, z_max - 1)
        y = np.clip(y, 0, y_max - 1)
        x = np.clip(x, 0, x_max - 1)
        # Define local neighborhood for background estimation
        z_slice = image_data[z]
        y_min, y_max_p = max(0, y-radius), min(y_max, y+radius+1)
        x_min, x_max_p = max(0, x-radius), min(x_max, x+radius+1)
        
        local_crop = z_slice[y_min:y_max_p, x_min:x_max_p]
        
        peak_intensity = image_data[z, y, x]
        background = np.median(local_crop) if local_crop.size > 0 else 1.0
        snr = peak_intensity / background if background > 0 else 0
        
        stats.append([peak_intensity, snr])
    return np.array(stats)

def apply_deconvolution(image_data, iterations=10):
    """Sharpens crowded fields using Richardson-Lucy deconvolution."""
    psf = np.ones((3, 3, 3)) / 27 
    img_float = image_data.astype(np.float32)
    return restoration.richardson_lucy(img_float, psf, num_iter=iterations)

def preprocess_white_tophat(image_data, radius=constants.PUNCTA_TOPHAT_RADIUS):
    """Applies slice-by-slice background subtraction."""
    selem = disk(radius)
    processed_slices = [white_tophat(image_slice, selem) for image_slice in image_data]
    return np.stack(processed_slices, axis=0)

def _detect_radial_symmetry(image_data, threshold_rel, sigma):
    """High-precision localization for high-density transcripts."""
    coords = peak_local_max(image_data, min_distance=1, threshold_rel=threshold_rel)
    return coords.astype(float) if len(coords) > 0 else np.empty((0, 3))

def _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma):
    """Standard intensity-based peak finding."""
    if sigma > 0:
        image_data = gaussian(image_data, sigma=sigma, preserve_range=True)
    return peak_local_max(image_data, min_distance=min_distance, 
                          threshold_rel=threshold_rel, exclude_border=False)

def _detect_spots_log(image_data, threshold_rel, sigma, z_scale=1.0):
    """Anisotropic Laplacian of Gaussian for PSF compensation."""
    s = sigma if sigma > 0 else 1.0
    sigma_vec = (s * z_scale, s, s)
    max_val = np.max(image_data)
    image_norm = image_data.astype(np.float32) / max_val if max_val > 0 else image_data
    blobs = blob_log(image_norm, min_sigma=sigma_vec, max_sigma=[sv * 1.5 for sv in sigma_vec], 
                     num_sigma=2, threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def _detect_spots_dog(image_data, threshold_rel, sigma, z_scale=1.0):
    """Anisotropic Difference of Gaussian for fast 3D detection."""
    s = sigma if sigma > 0 else 1.0
    sigma_vec = (s * z_scale, s, s)
    max_val = np.max(image_data)
    image_norm = image_data.astype(np.float32) / max_val if max_val > 0 else image_data
    blobs = blob_dog(image_norm, min_sigma=sigma_vec, max_sigma=[sv * 1.6 for sv in sigma_vec], 
                     threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def detect_spots_3d(image_data, method="Local Maxima", progress_callback=None, **kwargs):
    """Main entry point for Step 6 math."""
    if kwargs.get('use_decon', False):
        if progress_callback: progress_callback(10, "Deconvolving image...")
        image_data = apply_deconvolution(image_data, iterations=kwargs.get('decon_iter', 10))
    if kwargs.get('use_tophat', False):
        if progress_callback: progress_callback(25, "Subtracting background (top-hat)...")
        image_data = preprocess_white_tophat(image_data, radius=kwargs.get('tophat_radius', 10))

    if method == "Radial Symmetry":
        return _detect_radial_symmetry(image_data, kwargs.get('threshold_rel', 0.1), kwargs.get('sigma', 1.0))
    elif method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, kwargs.get('min_distance', 3), 
                                          kwargs.get('threshold_rel', 0.1), kwargs.get('sigma', 1.0))
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, kwargs.get('threshold_rel', 0.1), 
                                 kwargs.get('sigma', 1.0), z_scale=kwargs.get('z_scale', 1.0))
    elif method == "Difference of Gaussian":
        return _detect_spots_dog(image_data, kwargs.get('threshold_rel', 0.1), 
                                 kwargs.get('sigma', 1.0), z_scale=kwargs.get('z_scale', 1.0))
    return np.empty((0, 3))

def process_puncta_detection(image_data, mask_data=None, voxels=None, params=None, output_path=None, progress_callback=None):
    """Orchestrates detection, quality mapping, and session persistence with CSV tags."""
    from . import session
    params = params or {}

    if 'z_scale' not in params and voxels is not None:
        params['z_scale'] = voxels[0] / voxels[2]

    if progress_callback: progress_callback(5, "Detecting spots...")
    coords = detect_spots_3d(image_data, progress_callback=progress_callback, **params)
    if len(coords) == 0:
        if progress_callback: progress_callback(100, "No spots found.")
        return np.empty((0, 6))

    if progress_callback: progress_callback(50, f"Computing quality metrics for {len(coords)} spots...")
    quality_metrics = calculate_spot_quality(image_data, coords)

    if progress_callback: progress_callback(70, "Assigning nucleus IDs...")
    # Clip coordinates to valid mask/image bounds before indexing
    coords_int = coords.astype(int)
    for dim in range(3):
        coords_int[:, dim] = np.clip(coords_int[:, dim], 0, image_data.shape[dim] - 1)
    indices = tuple(coords_int.T)
    nucleus_ids = mask_data[indices] if mask_data is not None else np.zeros(len(coords))

    # Filter out extranuclear puncta if requested
    nuclei_only = params.get('nuclei_only', True)
    if nuclei_only and mask_data is not None:
        if progress_callback: progress_callback(80, "Filtering extranuclear puncta...")
        keep = nucleus_ids > 0
        coords = coords[keep]
        nucleus_ids = nucleus_ids[keep]
        quality_metrics = quality_metrics[keep]
        if len(coords) == 0:
            if progress_callback: progress_callback(100, "No nuclear puncta found.")
            return np.empty((0, 6))

    # Combined Data: Z, Y, X, Nucleus_ID, Intensity, SNR
    final_data = np.column_stack([coords, nucleus_ids, quality_metrics])

    if output_path:
        if progress_callback: progress_callback(90, "Saving results...")
        header = "Z,Y,X,Nucleus_ID,Intensity,SNR"
        np.savetxt(output_path, final_data, delimiter=",", header=header, comments='')

        # FIX: Pass format metadata so UI knows this is a text CSV, not a pickle
        session.set_processed_file(
            Path(output_path).stem,
            str(output_path),
            "points",
            metadata={
                'format': 'csv',
                'subtype': 'puncta_csv'
            }
        )

    if progress_callback: progress_callback(100, f"Done. Found {len(final_data)} spots.")
    return final_data