import numpy as np
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import logging
from skimage.transform import rescale, resize
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects
from cellpose import models, core

# Set logging to see the progress bar in terminal
logging.basicConfig(level=logging.INFO)

def segment_nuclei_3d(image_data, gpu=True):
    # 1. SETUP
    use_gpu = core.use_gpu() if gpu else False
    model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')

    # 2. SUBSAMPLE Z
    z_step = 2
    subsampled_data = image_data[::z_step, :, :]
    
    # 3. DOWNSAMPLE X/Y
    scale_factor = 0.25
    small_data = rescale(
        subsampled_data, 
        (1, scale_factor, scale_factor), 
        preserve_range=True, 
        anti_aliasing=True
    ).astype(np.float32) # Cellpose prefers float32

    # 4. EVALUATE (Fixing the ValueError)
    masks_small, flows, styles = model.eval(
        small_data,
        channels=[0,0],      # Grayscale DAPI
        diameter=None,       # We already handled scaling manually
        rescale=1.0,         # <--- CRITICAL: Prevents internal resizing
        do_3D=True,
        stitch_threshold=0.5,       
        z_axis=0,
        batch_size=16,               
        progress=True,
        resample=False       # <--- CRITICAL: Prevents internal resampling
    )

    # 5. SCALE CENTROIDS
    props = regionprops(masks_small)
    centroids = np.array([
        [p.centroid[0] * z_step, 
         p.centroid[1] / scale_factor, 
         p.centroid[2] / scale_factor] 
        for p in props
    ])
    
    # Resize masks back to original shape
    masks = resize(
        masks_small, 
        image_data.shape, 
        order=0, 
        preserve_range=True, 
        anti_aliasing=False
    ).astype(np.uint32)
    
    return masks, centroids

def segment_nuclei_classical(image_data, progress_callback=None):
    """
    Fast 3D nuclei segmentation using classical image processing.
    Much faster than Cellpose, suitable for registration landmarks.
    """
    # 1. Downsample for speed
    if progress_callback: progress_callback(0, "Downsampling...")
    z_step = 2
    scale_factor = 0.25
    
    small_data = rescale(
        image_data[::z_step], 
        (1, scale_factor, scale_factor), 
        preserve_range=True,
        anti_aliasing=False
    )

    # 2. Smooth to reduce noise
    if progress_callback: progress_callback(20, "Smoothing...")
    smoothed = gaussian(small_data, sigma=3)
    
    # 3. Threshold (Otsu)
    if progress_callback: progress_callback(40, "Thresholding...")
    try:
        thresh = threshold_otsu(smoothed)
        binary = smoothed > thresh
        binary = remove_small_objects(binary, min_size=50)
    except ValueError:
        return None, None # Handle empty images

    # 4. Distance Transform & Peak Finding (Separates touching nuclei)
    if progress_callback: progress_callback(60, "Finding nuclei centers...")
    distance = ndi.distance_transform_edt(binary)
    # min_distance=7 corresponds to ~28 pixels in original image (7 / 0.25)
    coords = peak_local_max(distance, min_distance=7, labels=binary)
    
    # 5. Extract Centroids directly from peaks
    centroids = np.array([
        [c[0] * z_step, c[1] / scale_factor, c[2] / scale_factor] 
        for c in coords
    ])

    # Generate markers for watershed
    if progress_callback: progress_callback(80, "Expanding centers (Watershed)...")
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(len(coords)) + 1
    
    # Watershed to get labeled regions
    labels_small = watershed(-distance, markers, mask=binary)
    
    # Resize labels back to original shape
    labels = resize(
        labels_small, 
        image_data.shape, 
        order=0, 
        preserve_range=True, 
        anti_aliasing=False
    ).astype(np.uint32)
    
    if progress_callback: progress_callback(100, "Done.")
    return labels, centroids

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