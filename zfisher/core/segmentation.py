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
from pathlib import Path
import tifffile

from .. import constants

logger = logging.getLogger(__name__)


# zfisher/core/segmentation.py
# zfisher/core/segmentation.py

# zfisher/core/segmentation.py (Partial: focus on process_consensus_nuclei)
def process_consensus_nuclei(mask1, mask2, output_dir, threshold=20.0, method="Union", progress_callback=None):
    """
    Updated Core Orchestrator to support Union vs Intersection and 3D coordinate integrity.
    """
    if progress_callback: 
        progress_callback(10, f"Matching nuclei labels ({method})...")
    
    # The original pts1 is not needed, only the relabeled mask2
    new_mask2, _, _ = match_nuclei_labels(mask1, mask2, threshold=threshold)
    merged_mask = merge_labeled_masks(mask1, new_mask2, method=method)
    
    # DEFINITIVE FIX: Clean the merged mask to remove any tiny, non-contiguous artifacts
    # created by the intersection. This prevents regionprops from generating invalid (NaN)
    # centroids, which was the root cause of the napari rendering warning.
    # The 'in_place' argument is not available in all scikit-image versions.
    # Reassigning the result is the backward-compatible way to achieve the same outcome.
    merged_mask = remove_small_objects(merged_mask, min_size=constants.NUC_SEG_OTSU_MIN_SIZE)

    # CRITICAL FIX: Calculate centroids from the FINAL merged mask, not the original R1 mask.
    # This ensures the IDs displayed in napari correspond to the actual consensus objects.
    final_pts = get_mask_centroids(merged_mask)

    if output_dir:
        from . import session 
        seg_dir = Path(output_dir) / constants.SEGMENTATION_DIR
        seg_dir.mkdir(exist_ok=True, parents=True)
        
        # Save the .tif mask
        mask_path = seg_dir / f"{constants.CONSENSUS_MASKS_NAME}.tif"
        tifffile.imwrite(mask_path, merged_mask.astype(np.uint32), compression='zlib')
        session.set_processed_file(constants.CONSENSUS_MASKS_NAME, str(mask_path), 'labels')
        
        # Save the structured .npy IDs
        ids_path = seg_dir / f"{constants.CONSENSUS_MASKS_NAME}{constants.CONSENSUS_IDS_SUFFIX}.npy"
        
        # FIX: Explicitly enforce 3D coordinates (Z, Y, X) for napari restoration
        dtype = [('coord', 'f4', 3), ('label', 'i4')]
        
        # Ensure coordinates are captured as 3-element arrays
        structured_pts = np.array([
            (np.array(p['coord'], dtype='f4'), int(p['label'])) 
            for p in final_pts
        ], dtype=dtype)
        
        np.save(ids_path, structured_pts)
        
        # Use 'structured_ids' subtype to trigger correct unpacking in viewer_helpers
        session.set_processed_file(
            f"{constants.CONSENSUS_MASKS_NAME}_IDs", 
            str(ids_path), 
            'points', 
            metadata={'subtype': 'structured_ids'}
        )

    return merged_mask, final_pts
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

def get_label_volumes(labels):
    """Return an array of voxel counts for each label in the mask."""
    props = regionprops(labels)
    return np.array([p.area for p in props]) if props else np.array([])


def compute_min_volume_threshold(all_volumes):
    """
    Compute a minimum volume threshold using median - 2*MAD.

    Uses MAD (median absolute deviation) instead of std for robustness
    against skewed distributions with many small fragments.

    Parameters
    ----------
    all_volumes : np.ndarray
        Pooled label volumes from one or more rounds.

    Returns
    -------
    int
        Minimum volume threshold. Labels smaller than this are fragments.
    """
    if len(all_volumes) < 3:
        return 1
    med = np.median(all_volumes)
    mad = np.median(np.abs(all_volumes - med))
    return max(1, int(med - 2 * mad))


def filter_small_labels(labels, min_vol, progress_callback=None):
    """
    Remove labels below a volume threshold and recompute centroids.

    Parameters
    ----------
    labels : np.ndarray
        A (Z, Y, X) labeled integer mask.
    min_vol : int
        Minimum voxel count for a label to survive.
    progress_callback : callable, optional
        Progress reporting function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (filtered_labels, centroids) with small labels zeroed out.
    """
    if progress_callback: progress_callback(0, f"Filtering labels below {min_vol} voxels...")
    filtered = remove_small_objects(labels, min_size=min_vol)
    props = regionprops(filtered)
    centroids = np.array([p.centroid for p in props]) if props else np.empty((0, 3))
    if progress_callback: progress_callback(100, f"Kept {len(props)} nuclei.")
    return filtered, centroids


def _merge_oversegmented_labels(labels, progress_callback=None):
    """
    Merge labels that were erroneously split along any axis.

    Finds all pairs of spatially adjacent labels and computes their shared
    boundary surface area. If the shared boundary is large relative to the
    smaller label's total surface, the pair is merged (indicating
    over-segmentation rather than two distinct nuclei touching).

    Parameters
    ----------
    labels : np.ndarray
        A (Z, Y, X) labeled integer mask.
    progress_callback : callable, optional
        Progress reporting function.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (merged_labels, centroids) with merged labels and recomputed centroids.
    """
    from collections import Counter

    if progress_callback: progress_callback(0, "Checking for over-segmented nuclei...")

    props = regionprops(labels)
    if len(props) < 2:
        centroids = np.array([p.centroid for p in props]) if props else np.empty((0, 3))
        return labels, centroids

    # Compute total surface voxels for each label
    label_surface = {}
    for p in props:
        # Surface area approximated by the difference between the region
        # and its erosion — but for speed, just use the bounding-box sliced region
        label_surface[p.label] = p.area  # will refine below

    # Find adjacent label pairs and count shared boundary voxels
    # by shifting the label array in each of 6 directions
    if progress_callback: progress_callback(20, "Finding adjacent label pairs...")
    boundary_counts = Counter()
    for axis in range(3):
        # Compare each voxel with its neighbor along +axis
        slc_a = [slice(None)] * 3
        slc_b = [slice(None)] * 3
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)
        lab_a = labels[tuple(slc_a)]
        lab_b = labels[tuple(slc_b)]
        # Where labels differ and both non-zero = shared boundary
        mask = (lab_a != lab_b) & (lab_a > 0) & (lab_b > 0)
        if np.any(mask):
            pairs = np.column_stack((lab_a[mask], lab_b[mask]))
            # Normalize pair order
            pairs = np.sort(pairs, axis=1)
            for a, b in pairs:
                boundary_counts[(a, b)] += 1

    if not boundary_counts:
        if progress_callback: progress_callback(100, "No adjacent labels found.")
        centroids = np.array([p.centroid for p in props])
        return labels, centroids

    # Compute total boundary voxels per label (sum of all its shared boundaries + exposed surface)
    # For the ratio, we use each label's total boundary voxel count
    if progress_callback: progress_callback(50, "Computing boundary ratios...")
    total_boundary = Counter()
    for (a, b), count in boundary_counts.items():
        total_boundary[a] += count
        total_boundary[b] += count

    # Also count exposed (background-facing) boundary voxels per label
    for axis in range(3):
        slc_a = [slice(None)] * 3
        slc_b = [slice(None)] * 3
        slc_a[axis] = slice(None, -1)
        slc_b[axis] = slice(1, None)
        lab_a = labels[tuple(slc_a)]
        lab_b = labels[tuple(slc_b)]
        # Voxel faces background
        mask_a = (lab_a > 0) & (lab_b == 0)
        mask_b = (lab_b > 0) & (lab_a == 0)
        if np.any(mask_a):
            for lab in lab_a[mask_a]:
                total_boundary[lab] += 1
        if np.any(mask_b):
            for lab in lab_b[mask_b]:
                total_boundary[lab] += 1

    # Union-find for merging
    all_labels = {p.label for p in props}
    parent = {lab: lab for lab in all_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    merge_count = 0
    threshold = constants.NUC_SEG_MERGE_BOUNDARY_RATIO
    for (a, b), shared in boundary_counts.items():
        if find(a) == find(b):
            continue
        # Ratio of shared boundary to the smaller label's total surface
        smaller_surface = min(total_boundary[a], total_boundary[b])
        if smaller_surface > 0 and (shared / smaller_surface) > threshold:
            union(a, b)
            merge_count += 1

    if merge_count == 0:
        if progress_callback: progress_callback(100, "No over-segmentation detected.")
        centroids = np.array([p.centroid for p in props])
        return labels, centroids

    if progress_callback: progress_callback(75, f"Merging {merge_count} over-segmented pair(s)...")

    # Relabel the mask
    label_map = {}
    next_label = 1
    for lab in sorted(all_labels):
        root = find(lab)
        if root not in label_map:
            label_map[root] = next_label
            next_label += 1
        label_map[lab] = label_map[root]

    merged = np.zeros_like(labels)
    for old_lab, new_lab in label_map.items():
        merged[labels == old_lab] = new_lab

    merged_props = regionprops(merged)
    centroids = np.array([p.centroid for p in merged_props]) if merged_props else np.empty((0, 3))

    if progress_callback: progress_callback(100, f"Merged into {len(merged_props)} nuclei.")
    return merged, centroids


def segment_nuclei_classical(image_data, voxel_spacing=None, merge_splits=True, progress_callback=None):
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
    # Account for anisotropic voxel spacing so the EDT doesn't over-split in Z.
    # voxel_spacing is (dz, dy, dx) in physical units (e.g. micrometers).
    # After downsampling: Z voxels are z_step× larger, XY voxels are 1/scale_factor× larger.
    if voxel_spacing is not None:
        dz, dy, dx = voxel_spacing
        edt_spacing = (dz * z_step, dy / scale_factor, dx / scale_factor)
    else:
        # Fallback: assume typical FISH anisotropy
        z_ratio = constants.NUC_SEG_Z_XY_RATIO * z_step * scale_factor
        edt_spacing = (z_ratio, 1, 1)
    distance = ndi.distance_transform_edt(binary, sampling=edt_spacing)
    coords = peak_local_max(distance, min_distance=constants.NUC_SEG_PEAK_MIN_DIST, labels=binary)
    
    # Generate markers for watershed
    if progress_callback: progress_callback(80, "Expanding centers (Watershed)...")
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(len(coords)) + 1

    # Watershed to get labeled regions
    labels_small = watershed(-distance, markers, mask=binary)

    # Extract centroids from watershed result
    props = regionprops(labels_small)
    centroids = np.array([
        [p.centroid[0] * z_step, p.centroid[1] / scale_factor, p.centroid[2] / scale_factor]
        for p in props
    ]) if props else np.empty((0, 3))

    # Resize labels back to original shape
    if progress_callback: progress_callback(85, "Rescaling masks...")
    labels = resize(
        labels_small,
        image_data.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False
    ).astype(np.uint32)

    # Post-processing: merge nuclei split along Z-axis
    if merge_splits:
        def merge_progress(val, text):
            if progress_callback: progress_callback(90 + int(val * 0.10), text)
        labels, centroids = _merge_oversegmented_labels(labels, progress_callback=merge_progress)
        labels = labels.astype(np.uint32)

    if progress_callback: progress_callback(100, "Done.")
    return labels, centroids

def segment_nuclei_cellpose(image_data, gpu=True, merge_splits=True, progress_callback=None):
    """
    Segment 3D nuclei using Cellpose in 2D-per-slice mode with cross-slice stitching.

    Runs the 'nuclei' model on each Z-slice independently, then stitches labels
    across Z using IoU overlap. Slower than classical watershed but more accurate
    for dense or touching nuclei.

    Parameters
    ----------
    image_data : np.ndarray
        The 3D (Z, Y, X) image data containing nuclei.
    gpu : bool, optional
        Whether to use GPU if available. Defaults to True.
    progress_callback : callable, optional
        A function to report progress, e.g., ``lambda p, m: ...``.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        (labels, centroids) matching the format of segment_nuclei_classical.
    """
    # 1. Downsample
    if progress_callback: progress_callback(0, "Downsampling...")
    z_step = constants.NUC_SEG_CP2D_Z_STEP
    scale_factor = constants.NUC_SEG_CP2D_SCALE_FACTOR

    small_data = rescale(
        image_data[::z_step],
        (1, scale_factor, scale_factor),
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    # 2. Init model
    if progress_callback: progress_callback(10, "Loading Cellpose model...")
    use_gpu = core.use_gpu() if gpu else False
    model = models.CellposeModel(gpu=use_gpu, model_type='nuclei')

    # 3. Run 2D eval with stitching
    if progress_callback: progress_callback(15, "Running Cellpose (2D + stitch)...")
    masks_small, _, _ = model.eval(
        small_data,
        channels=[0, 0],
        diameter=constants.NUC_SEG_CP2D_DIAMETER,
        do_3D=False,
        stitch_threshold=constants.NUC_SEG_CP2D_STITCH_THRESH,
        z_axis=0,
        progress=True,
    )

    if progress_callback: progress_callback(85, "Cleaning up...")
    masks_small = remove_small_objects(masks_small, min_size=constants.NUC_SEG_OTSU_MIN_SIZE)

    # 4. Extract centroids
    if progress_callback: progress_callback(90, "Extracting centroids...")
    props = regionprops(masks_small)
    if not props:
        return None, None

    centroids = np.array([
        [p.centroid[0] * z_step, p.centroid[1] / scale_factor, p.centroid[2] / scale_factor]
        for p in props
    ])

    # 5. Resize labels back to original shape
    if progress_callback: progress_callback(90, "Rescaling masks...")
    labels = resize(
        masks_small,
        image_data.shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.uint32)

    # Post-processing: merge nuclei split along Z-axis
    if merge_splits:
        def merge_progress(val, text):
            if progress_callback: progress_callback(93 + int(val * 0.07), text)
        labels, centroids = _merge_oversegmented_labels(labels, progress_callback=merge_progress)
        labels = labels.astype(np.uint32)

    if progress_callback: progress_callback(100, "Done.")
    return labels, centroids

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

    # Auto-determine threshold from the distance distribution if not explicitly set
    if threshold is None or threshold <= 0:
        median_dist = np.median(dists[np.isfinite(dists)])
        mad = np.median(np.abs(dists[np.isfinite(dists)] - median_dist))
        threshold = median_dist + 3 * max(mad, 1.0)
        if progress_callback: progress_callback(40, f"Auto threshold: {threshold:.1f} px")

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

def merge_labeled_masks(mask1, mask2, method="Intersection"):
    """
    Safely merges masks by expanding both to a shared 'Union' shape.
    This prevents broadcasting errors when stacks have different slice counts.
    """
    # 1. Determine the largest required dimension in Z, Y, and X
    max_z = max(mask1.shape[0], mask2.shape[0])
    max_y = max(mask1.shape[1], mask2.shape[1])
    max_x = max(mask1.shape[2], mask2.shape[2])
    union_shape = (max_z, max_y, max_x)

    # 2. Helper to pad a mask to the union shape without shifting its data
    def pad_to_union(mask, target_shape):
        if mask.shape == target_shape:
            return mask
        padded = np.zeros(target_shape, dtype=mask.dtype)
        padded[:mask.shape[0], :mask.shape[1], :mask.shape[2]] = mask
        return padded

    # 3. Align the array sizes (Computational alignment)
    m1_padded = pad_to_union(mask1, union_shape)
    m2_padded = pad_to_union(mask2, union_shape)

    # 4. Perform the biological matching
    if method == "Intersection":
        overlap_mask = (m1_padded > 0) & (m2_padded > 0)
        merged = np.zeros(union_shape, dtype=mask1.dtype)
        merged[overlap_mask] = m1_padded[overlap_mask]
        return merged
    
    # Union Method
    merged = m1_padded.copy()
    fill_indices = (merged == 0) & (m2_padded > 0)
    merged[fill_indices] = m2_padded[fill_indices]
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

def prepare_id_points(mask_data):
    """Calculates centroids and prepares arrays for ID visualization."""
    pts_data = get_mask_centroids(mask_data)
    if not pts_data:
        return np.empty((0, mask_data.ndim)), np.empty(0)
        
    coords = np.array([p['coord'] for p in pts_data])
    labels = np.array([p['label'] for p in pts_data])
    return coords, labels

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


# zfisher/core/segmentation.py

def process_session_dapi(r1_data, r2_data=None, output_dir=None, progress_callback=None, method="classical", voxel_spacing=None):
    """
    Core Orchestrator: Runs segmentation on one or both rounds and saves results.
    """
    # Pass 1: Segment all rounds
    raw_results = {}
    data_map = [("R1", r1_data)]
    if r2_data is not None:
        data_map.append(("R2", r2_data))

    for i, (prefix, data) in enumerate(data_map):
        if data is None: continue

        def on_step_progress(val, text, _i=i):
            if progress_callback:
                base = (_i / len(data_map)) * 80
                progress_callback(int(base + (val / len(data_map)) * 0.8), f"{prefix}: {text}")

        if method == "cellpose":
            masks, centroids = segment_nuclei_cellpose(data, progress_callback=on_step_progress)
        else:
            masks, centroids = segment_nuclei_classical(data, voxel_spacing=voxel_spacing, progress_callback=on_step_progress)
        raw_results[prefix] = (masks, centroids)

    # Pass 2: Pool volumes from all rounds, compute shared threshold, filter
    if progress_callback: progress_callback(80, "Computing volume threshold...")
    all_volumes = np.concatenate([
        get_label_volumes(masks) for masks, _ in raw_results.values() if masks is not None
    ])
    min_vol = compute_min_volume_threshold(all_volumes)

    results = {}
    for prefix, (masks, centroids) in raw_results.items():
        if masks is not None:
            masks, centroids = filter_small_labels(masks, min_vol)
            masks = masks.astype(np.uint32)
        results[prefix] = (masks, centroids)

    if progress_callback: progress_callback(90, "Saving results...")

    # Save to disk if output_dir exists
    if output_dir:
        from . import session
        seg_dir = Path(output_dir) / constants.SEGMENTATION_DIR
        seg_dir.mkdir(exist_ok=True, parents=True)

        for prefix, (masks, centroids) in results.items():
            dapi_layer_name = f"{prefix} - {constants.DAPI_CHANNEL_NAME}"
            mask_layer_name = f"{dapi_layer_name}{constants.MASKS_SUFFIX}"
            centroid_layer_name = f"{dapi_layer_name}{constants.CENTROIDS_SUFFIX}"

            mask_path = seg_dir / f"{mask_layer_name}.tif"
            tifffile.imwrite(mask_path, masks.astype(np.uint32), compression='zlib')
            session.set_processed_file(mask_layer_name, str(mask_path), layer_type='labels', metadata={'subtype': 'mask'})

            cent_path = seg_dir / f"{centroid_layer_name}.npy"
            np.save(cent_path, centroids)
            session.set_processed_file(centroid_layer_name, str(cent_path), layer_type='points', metadata={'subtype': 'centroids'})

    if progress_callback: progress_callback(100, "Done.")
    return results