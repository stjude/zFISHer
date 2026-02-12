import numpy as np
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian
from skimage.measure import regionprops
from scipy.spatial import cKDTree

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

def match_nuclei_labels(mask1, mask2, threshold=20, progress_callback=None):
    """
    Matches nuclei in mask2 to mask1 based on centroid distance.
    Relabels mask2 to match mask1 IDs where possible.
    
    Args:
        mask1: Reference labels (Z, Y, X)
        mask2: Moving labels (Z, Y, X)
        threshold: Max distance in pixels to consider a match
        progress_callback: Optional function for progress reporting.
        
    Returns:
        new_mask2: Relabeled mask2
        points1: List of dicts {'coord', 'label'} for mask1
        points2: List of dicts {'coord', 'label'} for new_mask2
    """
    if progress_callback: progress_callback(0, "Analyzing reference nuclei...")
    props1 = regionprops(mask1)
    
    if progress_callback: progress_callback(15, "Analyzing moving nuclei...")
    props2 = regionprops(mask2)
    
    if not props1 or not props2:
        if progress_callback: progress_callback(100, "No nuclei found.")
        return mask2, [], []

    c1 = np.array([p.centroid for p in props1])
    l1 = np.array([p.label for p in props1])
    
    c2 = np.array([p.centroid for p in props2])
    l2 = np.array([p.label for p in props2])
    
    if progress_callback: progress_callback(30, "Finding nearest neighbors...")
    tree = cKDTree(c1)
    dists, idxs = tree.query(c2)
    
    if progress_callback: progress_callback(50, "Creating ID map...")
    mapping = {}
    next_id = np.max(l1) + 1
    
    for i, (d, idx) in enumerate(zip(dists, idxs)):
        old_label = l2[i]
        if d < threshold:
            target_label = l1[idx]
            # Handle potential merges: if multiple mask2 labels map to one mask1 label
            if target_label in mapping.values():
                # This old_label should be considered unmatched and get a new ID
                mapping[old_label] = next_id
                next_id += 1
            else:
                mapping[old_label] = target_label
        else:
            mapping[old_label] = next_id
            next_id += 1
            
    if progress_callback: progress_callback(75, "Applying new labels to mask...")
    max_val = np.max(mask2)
    lookup = np.zeros(max_val + 1, dtype=mask2.dtype)
    for old, new in mapping.items():
        lookup[old] = new
        
    new_mask2 = lookup[mask2]
    
    if progress_callback: progress_callback(90, "Generating visualization points...")
    points1_data = [{'coord': p.centroid, 'label': p.label} for p in props1]
    points2_data = [{'coord': c2[i], 'label': mapping[l2[i]]} for i in range(len(c2))]
        
    if progress_callback: progress_callback(100, "Done.")
    return new_mask2, points1_data, points2_data

def merge_labeled_masks(mask1, mask2):
    """
    Merges two labeled masks. Assumes IDs are already matched/synchronized.
    Prioritizes mask1, fills gaps with mask2.
    """
    merged = mask1.copy()
    # Where mask1 is 0 (background) and mask2 has a label, use mask2
    fill_indices = (merged == 0) & (mask2 > 0)
    merged[fill_indices] = mask2[fill_indices]
    return merged

def get_mask_centroids(mask):
    """Returns list of dicts {'coord': (z,y,x), 'label': id} for a mask."""
    props = regionprops(mask)
    return [{'coord': p.centroid, 'label': p.label} for p in props]