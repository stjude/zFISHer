import numpy as np
from skimage.feature import peak_local_max, blob_log
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import logging
from skimage.transform import rescale, resize
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, white_tophat, disk
from cellpose import models, core

from .. import constants

# Set logging to see the progress bar in terminal
logging.basicConfig(level=logging.INFO)

def segment_nuclei_3d(image_data, gpu=True):
    """
    Segments 3D nuclei using the Cellpose model.

    This function downsamples the data for performance, runs the 'nuclei'
    model, and then scales the resulting masks and centroids back to the
    original image dimensions.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D (Z, Y, X) image data containing nuclei (e.g., DAPI channel).
    gpu : bool, optional
        Whether to use a GPU for computation if available. Defaults to True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - masks: A (Z, Y, X) labeled integer mask of the segmented nuclei.
        - centroids: A (N, 3) array of the (Z, Y, X) centroids for each nucleus.
    """
    # 1. SETUP
    use_gpu = core.use_gpu() if gpu else False
    model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')

    # 2. SUBSAMPLE Z
    z_step = constants.NUC_SEG_3D_Z_STEP
    subsampled_data = image_data[::z_step, :, :]
    
    # 3. DOWNSAMPLE X/Y
    scale_factor = constants.NUC_SEG_3D_SCALE_FACTOR
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
        stitch_threshold=constants.NUC_SEG_3D_STITCH_THRESH,
        z_axis=0,
        batch_size=constants.NUC_SEG_3D_BATCH_SIZE,
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
    Segment 3D nuclei using a classical image processing workflow.

    This method is significantly faster than deep learning models like Cellpose
    and is well-suited for generating landmarks for registration. The workflow
    involves downsampling, smoothing, thresholding, and watershed segmentation.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D (Z, Y, X) image data containing nuclei.
    progress_callback : callable, optional
        A function to report progress, e.g., `lambda p, m: print(f"{p}%: {m}")`.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        A tuple containing:
        - labels: A (Z, Y, X) labeled integer mask of the segmented nuclei.
        - centroids: A (N, 3) array of the (Z, Y, X) centroids.
        Returns (None, None) if the input image is empty or segmentation fails.
    """
    # 1. Downsample for speed
    if progress_callback: progress_callback(0, "Downsampling...")
    z_step = constants.NUC_SEG_Z_STEP
    scale_factor = constants.NUC_SEG_SCALE_FACTOR
    
    small_data = rescale(
        image_data[::z_step], 
        (1, scale_factor, scale_factor), 
        preserve_range=True,
        anti_aliasing=False
    )

    # 2. Smooth to reduce noise
    if progress_callback: progress_callback(20, "Smoothing...")
    smoothed = gaussian(small_data, sigma=constants.NUC_SEG_GAUSSIAN_SIGMA)
    
    # 3. Threshold (Otsu)
    if progress_callback: progress_callback(40, "Thresholding...")
    try:
        thresh = threshold_otsu(smoothed)
        binary = smoothed > thresh
        binary = remove_small_objects(binary, min_size=constants.NUC_SEG_OTSU_MIN_SIZE)
    except ValueError:
        return None, None # Handle empty images

    # 4. Distance Transform & Peak Finding (Separates touching nuclei)
    if progress_callback: progress_callback(60, "Finding nuclei centers...")
    distance = ndi.distance_transform_edt(binary)
    # min_distance corresponds to ~28 pixels in original image (7 / 0.25)
    coords = peak_local_max(distance, min_distance=constants.NUC_SEG_PEAK_MIN_DIST, labels=binary)
    
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

def preprocess_white_tophat(image_data, radius=constants.PUNCTA_TOPHAT_RADIUS):
    """
    Applies a white top-hat transform to subtract uneven background.

    This is useful for enhancing bright spots before detection. The operation
    is applied slice-by-slice in 3D.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D (Z, Y, X) image data.
    radius : int, optional
        The radius of the disk-shaped structuring element for the morphological
        opening. This should be larger than the radius of the spots.

    Returns
    -------
    np.ndarray
        The background-subtracted image.
    """
    selem = disk(radius)
    # Apply to each Z-slice. This is generally faster and effective for spots.
    processed_slices = [white_tophat(image_slice, selem) for image_slice in image_data]
    return np.stack(processed_slices, axis=0)

def _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma):
    """
    Finds spots using the local maxima method.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D image data.
    min_distance, threshold_rel, sigma :
        Parameters for `skimage.feature.peak_local_max`.

    Returns
    -------
    np.ndarray
        (N, 3) array of spot coordinates.
    """
    if sigma > 0:
        image_data = gaussian(image_data, sigma=sigma, preserve_range=True)
    
    return peak_local_max(
        image_data,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        exclude_border=False
    )

def _detect_spots_log(image_data, threshold_rel, sigma):
    """
    Finds spots using the Laplacian of Gaussian (LoG) method.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D image data.
    threshold_rel, sigma :
        Parameters for `skimage.feature.blob_log`.

    Returns
    -------
    np.ndarray
        (N, 3) array of spot coordinates.
    """
    s = sigma if sigma > 0 else 1.0
    abs_threshold = threshold_rel * np.max(image_data)
    
    blobs = blob_log(
        image_data,
        min_sigma=s,
        max_sigma=s * 1.5,
        num_sigma=2,
        threshold=abs_threshold
    )
    
    if len(blobs) > 0:
        return blobs[:, :3].astype(int)
    return np.empty((0, 3))

def detect_spots_3d(image_data, min_distance=constants.PUNCTA_MIN_DISTANCE, threshold_rel=constants.PUNCTA_THRESHOLD_REL, sigma=constants.PUNCTA_SIGMA, method="Local Maxima", use_tophat=False, tophat_radius=constants.PUNCTA_TOPHAT_RADIUS):
    """
    Detects diffraction-limited spots (puncta) in a 3D image.
    
    Parameters
    ----------
    image_data : np.ndarray
        The 3D (Z, Y, X) image data.
    min_distance : int, optional
        Minimum pixel distance between spots for the "Local Maxima" method.
    threshold_rel : float, optional
        Relative intensity threshold (0.0 to 1.0).
    sigma : float, optional
        Standard deviation for Gaussian kernel (smoothing). 0 to disable.
    method : {"Local Maxima", "Laplacian of Gaussian"}, optional
        The algorithm to use for spot detection.
    use_tophat : bool, optional
        If True, applies a white top-hat filter for background subtraction
        before spot detection.
    tophat_radius : int, optional
        The radius for the white top-hat structuring element.

    Returns
    -------
    np.ndarray
        (N, 3) array of (Z, Y, X) spot coordinates.
    """
    if use_tophat:
        image_data = preprocess_white_tophat(image_data, radius=tophat_radius)

    if method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma)
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, threshold_rel, sigma)
    else:
        raise ValueError(f"Unknown spot detection method: {method}")

def merge_puncta(existing_coords, new_coords):
    """
    Combines new puncta coordinates with existing ones and removes duplicates.

    Parameters
    ----------
    existing_coords : np.ndarray
        Array of existing (N, D) coordinates.
    new_coords : np.ndarray
        Array of new (M, D) coordinates to add.

    Returns
    -------
    np.ndarray
        A new array of unique combined coordinates.
    """
    if new_coords is None or len(new_coords) == 0:
        return existing_coords
    if existing_coords is None or len(existing_coords) == 0:
        return new_coords
        
    combined = np.vstack((existing_coords, new_coords))
    return np.unique(combined, axis=0)

def match_nuclei_labels(mask1, mask2, threshold=20, progress_callback=None):
    """
    Matches nuclei in mask2 to mask1 based on centroid distance.

    Relabels mask2 to match mask1 IDs where possible.
    
    Parameters
    ----------
    mask1 : np.ndarray
        The reference labels (Z, Y, X) array.
    mask2 : np.ndarray
        The moving labels (Z, Y, X) array to be relabeled.
    threshold : float, optional
        Maximum distance in pixels to consider two centroids a match.
    progress_callback : callable, optional
        A function to report progress.

    Returns
    -------
    tuple[np.ndarray, list, list]
        A tuple containing (new_mask2, points1, points2), where `points1` and
        `points2` are lists of dictionaries containing centroid coordinates and labels.
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
    Merges two labeled masks, prioritizing mask1.

    Assumes that the label IDs in both masks have already been synchronized.
    This function fills in the background of `mask1` with labeled regions
    from `mask2`.

    Parameters
    ----------
    mask1, mask2 : np.ndarray
        The two (Z, Y, X) label arrays to merge.

    Returns
    -------
    np.ndarray
        The merged label mask.
    """
    merged = mask1.copy()
    # Where mask1 is 0 (background) and mask2 has a label, use mask2
    fill_indices = (merged == 0) & (mask2 > 0)
    merged[fill_indices] = mask2[fill_indices]
    return merged

def get_mask_centroids(mask):
    """
    Calculates the centroid for each labeled region in a mask.

    Parameters
    ----------
    mask : np.ndarray
        A (Z, Y, X) labeled integer mask.

    Returns
    -------
    list[dict]
        A list of dictionaries, each with 'coord' and 'label' keys.
    """
    props = regionprops(mask)
    return [{'coord': p.centroid, 'label': p.label} for p in props]

def merge_labels(mask_data, source_id, target_id):
    """
    Merges one label ID into another within a mask array.

    Parameters
    ----------
    mask_data : np.ndarray
        The label mask data.
    source_id : int
        The label ID to be replaced.
    target_id : int
        The label ID to replace with.

    Returns
    -------
    np.ndarray
        A new mask array with the labels merged.
    """
    if source_id == target_id:
        return mask_data
    new_data = mask_data.copy()
    new_data[new_data == source_id] = target_id
    return new_data

def delete_label(mask_data, label_id):
    """
    Deletes a label from a mask by setting its pixels to 0.

    Parameters
    ----------
    mask_data : np.ndarray
        The label mask data.
    label_id : int
        The ID of the label to delete.

    Returns
    -------
    np.ndarray
        A new mask array with the label removed.
    """
    if label_id == 0:
        return mask_data
    new_data = mask_data.copy()
    new_data[new_data == label_id] = 0
    return new_data

def extrude_label(mask_data, z_index, label_id):
    """
    Extrudes a 2D label mask on a given Z-slice through all other Z-slices.

    Parameters
    ----------
    mask_data : np.ndarray
        The 3D label mask data.
    z_index : int
        The index of the Z-slice containing the 2D mask to extrude.
    label_id : int
        The ID of the label to extrude.

    Returns
    -------
    np.ndarray
        A new 3D mask array with the label extruded.
    """
    if mask_data.ndim != 3 or label_id == 0:
        return mask_data
    
    current_slice = mask_data[z_index]
    mask_2d = (current_slice == label_id)
    
    if not np.any(mask_2d):
        return mask_data
        
    new_data = mask_data.copy()
    new_data[:, mask_2d] = label_id
    return new_data