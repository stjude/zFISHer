import numpy as np
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian

def detect_spots_3d(image_data, min_distance=2, threshold_rel=0.05, sigma=0.0, method="Local Maxima"):
    """
    Detects diffraction-limited spots (puncta) in a 3D image.
    
    Args:
        image_data: 3D numpy array (Z, Y, X)
        min_distance: Minimum pixel distance between spots
        threshold_rel: Relative intensity threshold (0.0 to 1.0)
        sigma: Standard deviation for Gaussian kernel (smoothing). 0 to disable.
        method: "Local Maxima" (fast) or "Laplacian of Gaussian" (robust but slower).
        
    Returns:
        coords: (N, 3) array of (Z, Y, X) coordinates
    """
    coords = np.empty((0, 3))
    
    if method == "Local Maxima":
        # Smooth image to enhance spots of expected size and reduce noise
        if sigma > 0:
            image_data = gaussian(image_data, sigma=sigma, preserve_range=True)

        # Find peaks in the image
        coords = peak_local_max(
            image_data, 
            min_distance=min_distance, 
            threshold_rel=threshold_rel,
            exclude_border=False
        )
        
    elif method == "Laplacian of Gaussian":
        # LoG requires a sigma range. If sigma is 0 (off), we default to 1.0.
        s = sigma if sigma > 0 else 1.0
        
        # blob_log uses absolute threshold. We estimate it from the relative threshold.
        # Note: This is an approximation.
        abs_threshold = threshold_rel * np.max(image_data)
        
        blobs = blob_log(
            image_data,
            min_sigma=s,
            max_sigma=s * 1.5, # Small range to target specific spot size
            num_sigma=2,
            threshold=abs_threshold
        )
        
        if len(blobs) > 0:
            # blob_log returns (z, y, x, sigma), we only need coordinates
            coords = blobs[:, :3].astype(int)
            
    return coords