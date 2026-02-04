import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import gaussian

def detect_spots_3d(image_data, min_distance=2, threshold_rel=0.05):
    """
    Detects diffraction-limited spots (puncta) in a 3D image.
    
    Args:
        image_data: 3D numpy array (Z, Y, X)
        min_distance: Minimum pixel distance between spots
        threshold_rel: Relative intensity threshold (0.0 to 1.0)
        
    Returns:
        coords: (N, 3) array of (Z, Y, X) coordinates
    """
    # Find peaks in the image
    # We use a low min_distance because puncta can be close together
    coords = peak_local_max(
        image_data, 
        min_distance=min_distance, 
        threshold_rel=threshold_rel,
        exclude_border=False
    )
    return coords