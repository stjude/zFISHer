import numpy as np
import tifffile
from pathlib import Path
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from .. import constants

def preprocess_white_tophat(image_data, radius=constants.PUNCTA_TOPHAT_RADIUS):
    """
    Applies a slice-by-slice background subtraction to enhance bright spots.
    """
    selem = disk(radius)
    processed_slices = [white_tophat(image_slice, selem) for image_slice in image_data]
    return np.stack(processed_slices, axis=0)

def _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma):
    """
    Internal helper for the Local Maxima detection method.
    """
    if sigma > 0:
        image_data = gaussian(image_data, sigma=sigma, preserve_range=True)
    return peak_local_max(image_data, min_distance=min_distance, 
                          threshold_rel=threshold_rel, exclude_border=False)

def _detect_spots_log(image_data, threshold_rel, sigma):
    """
    Internal helper for the Laplacian of Gaussian (LoG) method.
    """
    s = sigma if sigma > 0 else 1.0
    # The threshold for blob_log is absolute and should be applied to a
    # normalized image for the 'threshold_rel' (sensitivity) to be meaningful.
    # We normalize the image to [0, 1] float, so threshold_rel can be used directly.
    max_val = np.max(image_data)
    min_val = np.min(image_data)
    if max_val == min_val:
        # Avoid division by zero for blank images
        image_norm = np.zeros_like(image_data, dtype=np.float32)
    else:
        image_norm = (image_data.astype(np.float32) - min_val) / (max_val - min_val)

    blobs = blob_log(image_norm, min_sigma=s, max_sigma=s * 1.5, 
                     num_sigma=2, threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def _detect_spots_dog(image_data, threshold_rel, sigma):
    """
    Internal helper for the Difference of Gaussian (DoG) method.
    Provides a faster alternative to LoG for large 3D volumes.
    """
    s = sigma if sigma > 0 else 1.0
    # The threshold for blob_dog is absolute and should be applied to a
    # normalized image for the 'threshold_rel' (sensitivity) to be meaningful.
    # We normalize the image to [0, 1] float, so threshold_rel can be used directly.
    max_val = np.max(image_data)
    min_val = np.min(image_data)
    if max_val == min_val:
        # Avoid division by zero for blank images
        image_norm = np.zeros_like(image_data, dtype=np.float32)
    else:
        image_norm = (image_data.astype(np.float32) - min_val) / (max_val - min_val)

    # Uses a sigma ratio of 1.6 to approximate the LoG
    blobs = blob_dog(image_norm, min_sigma=s, max_sigma=s * 1.6, 
                     threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def merge_puncta(existing_coords, new_coords):
    """
    Combines coordinate arrays and removes any overlapping duplicates.
    """
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
    """
    Main entry point for 3D spot detection.
    Dispatches to Local Maxima, LoG, or DoG based on the 'method' parameter.
    """
    if use_tophat:
        image_data = preprocess_white_tophat(image_data, radius=tophat_radius)

    if method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma)
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, threshold_rel, sigma)
    elif method == "Difference of Gaussian":
        return _detect_spots_dog(image_data, threshold_rel, sigma)
    else:
        raise ValueError(f"Unknown spot detection method: {method}")

def process_puncta_detection(image_data, mask_data=None, params=None, output_path=None):
    """
    Core Orchestrator for Step 6.
    Detects spots, maps them to Nucleus IDs, and saves results to the session reports.
    """
    from . import session # Local import to avoid circular dependencies
    
    params = params or {}
    coords = detect_spots_3d(image_data, **params)

    if len(coords) == 0:
        return np.empty((0, 4))

    # Assign each spot to a Nucleus ID using the Consensus Mask
    if mask_data is not None:
        z, y, x = coords.astype(int).T
        nucleus_ids = mask_data[z, y, x]
        final_data = np.column_stack([coords, nucleus_ids])
    else:
        # Default to 0 if no mask is provided
        final_data = np.column_stack([coords, np.zeros(len(coords))])

    # Persistence: Automated CSV saving for headless runs
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(output_path, final_data, delimiter=",", 
                   header="Z,Y,X,Nucleus_ID", comments='')
        
        # Log the file in zfisher_session.json
        session.set_processed_file(
            layer_name=output_path.stem,
            path=str(output_path),
            layer_type="points"
        )
        
    return final_data