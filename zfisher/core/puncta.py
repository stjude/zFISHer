import numpy as np
import tifffile
from pathlib import Path
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from .. import constants


def preprocess_white_tophat(image_data, radius=constants.PUNCTA_TOPHAT_RADIUS):
    """Slice-by-slice background subtraction for puncta detection."""
    selem = disk(radius)
    processed_slices = [white_tophat(image_slice, selem) for image_slice in image_data]
    return np.stack(processed_slices, axis=0)

def _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma):
    """Internal helper for Local Maxima spot finding."""
    if sigma > 0:
        image_data = gaussian(image_data, sigma=sigma, preserve_range=True)
    return peak_local_max(image_data, min_distance=min_distance, 
                          threshold_rel=threshold_rel, exclude_border=False)

def _detect_spots_log(image_data, threshold_rel, sigma):
    """Internal helper for Laplacian of Gaussian spot finding."""
    s = sigma if sigma > 0 else 1.0
    abs_threshold = threshold_rel * np.max(image_data)
    blobs = blob_log(image_data, min_sigma=s, max_sigma=s * 1.5, 
                     num_sigma=2, threshold=abs_threshold)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def merge_puncta(existing_coords, new_coords):
    """Combines coordinates and removes duplicates."""
    if new_coords is None or len(new_coords) == 0:
        return existing_coords
    if existing_coords is None or len(existing_coords) == 0:
        return new_coords
    combined = np.vstack((existing_coords, new_coords))
    return np.unique(combined, axis=0)

def detect_spots_3d(image_data, min_distance=constants.PUNCTA_MIN_DISTANCE, 
                    threshold_rel=constants.PUNCTA_THRESHOLD_REL, 
                    sigma=constants.PUNCTA_SIGMA, method="Local Maxima", 
                    use_tophat=False, tophat_radius=constants.PUNCTA_TOPHAT_RADIUS):
    """Main spot detection entry point."""
    if use_tophat:
        image_data = preprocess_white_tophat(image_data, radius=tophat_radius)

    if method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma)
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, threshold_rel, sigma)
    else:
        raise ValueError(f"Unknown spot detection method: {method}")

def process_puncta_detection(image_data, mask_data=None, params=None, output_path=None):
    """Orchestrates detection and mapping to Nuclei IDs."""
    params = params or {}
    coords = detect_spots_3d(image_data, **params)

    if len(coords) == 0:
        return np.empty((0, 4))

    if mask_data is not None:
        z, y, x = coords.astype(int).T
        nucleus_ids = mask_data[z, y, x]
        final_data = np.column_stack([coords, nucleus_ids])
    else:
        final_data = np.column_stack([coords, np.zeros(len(coords))])

    if output_path:
        np.savetxt(output_path, final_data, delimiter=",", 
                   header="Z,Y,X,Nucleus_ID", comments='')
        
    return final_data