import numpy as np
import logging
from skimage.measure import ransac, regionprops
from skimage.transform import AffineTransform, rescale, resize
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import SimpleITK as sitk
import warnings
from skimage.filters import gaussian, threshold_otsu
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
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

def align_centroids_ransac(fixed_points, moving_points, max_distance=None, progress_callback=None):
    """
    Calculates the rigid shift between two point clouds using Vector Voting + RANSAC.
    This is robust to large shifts where nearest-neighbor matching fails.
    
    Args:
        fixed_points: (N, 3) array of centroids from Round 1 (Z, Y, X)
        moving_points: (M, 3) array of centroids from Round 2 (Z, Y, X)
        max_distance: Ignored (kept for compatibility).
        progress_callback: Optional function to report progress (value, message).
        
    Returns:
        shift_vector: (Z, Y, X) translation needed to move Round 2 to Round 1
    """
    # 1. Brute-force Vector Voting to find rough shift
    if progress_callback: progress_callback(10, "Finding rough alignment...")
    n_limit = 2000
    fp = fixed_points
    mp = moving_points
    
    if len(fp) > n_limit:
        fp = fp[np.random.choice(len(fp), n_limit, replace=False)]
    if len(mp) > n_limit:
        mp = mp[np.random.choice(len(mp), n_limit, replace=False)]
        
    if len(fp) < 1 or len(mp) < 1:
        return np.array([0.0, 0.0, 0.0])

    diffs = fp[:, np.newaxis, :] - mp[np.newaxis, :, :]
    diffs = diffs.reshape(-1, 3)
    
    bin_size = 5.0
    binned_diffs = np.round(diffs / bin_size).astype(int)
    
    unique_bins, counts = np.unique(binned_diffs, axis=0, return_counts=True)
    best_bin = unique_bins[np.argmax(counts)]
    rough_shift = best_bin * bin_size
    
    # 2. Refine using Nearest Neighbors
    if progress_callback: progress_callback(40, "Matching nearest neighbors...")
    moving_shifted = moving_points + rough_shift
    
    search_radius = 100.0
    tree = cKDTree(fixed_points)
    distances, indices = tree.query(moving_shifted, distance_upper_bound=search_radius)
    
    valid_mask = distances < float('inf')
    src = moving_points[valid_mask]
    dst = fixed_points[indices[valid_mask]]
    
    if len(src) < 4:
        if progress_callback: progress_callback(100, "Done.")
        return rough_shift

    # 3. Run RANSAC to refine the fit
    if progress_callback: progress_callback(70, "Running RANSAC to find best fit...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No inliers found")
        model, inliers = ransac(
            (src, dst), 
            AffineTransform, 
            min_samples=4, 
            residual_threshold=50,
            max_trials=2000
        )
    
    if model is None:
        if progress_callback: progress_callback(100, "Done.")
        return rough_shift

    shift = model.translation
    
    deviation = np.linalg.norm(shift - rough_shift)
    if deviation > 500.0:
        if progress_callback: progress_callback(100, "Done (reverted to rough shift).")
        return rough_shift
    
    if np.any(np.isnan(shift)):
        if progress_callback: progress_callback(100, "Done (reverted to rough shift).")
        return rough_shift
        
    if progress_callback: progress_callback(100, "Done.")
    return shift

def align_and_pad_images(fixed_data, moving_data, shift_vector, is_label=False):
    """
    Aligns two 3D volumes based on a shift vector by zero-padding.
    Returns two volumes of the same shape (Union Bounding Box).
    
    Args:
        fixed_data: (Z, Y, X) numpy array
        moving_data: (Z, Y, X) numpy array
        shift_vector: (Z, Y, X) shift to apply to moving_data to match fixed_data
        
    Returns:
        padded_fixed, padded_moving
    """
    # Round shift to nearest integer for lossless padding
    shift = np.round(shift_vector).astype(int)
    dz, dy, dx = shift
    # 1. Separate Integer and Fractional Shift
    # We round to the nearest integer for the grid placement
    shift_int = np.round(shift_vector).astype(int)
    shift_frac = shift_vector - shift_int
    
    dz, dy, dx = shift_int
    # Dimensions
    fz, fy, fx = fixed_data.shape
    mz, my, mx = moving_data.shape
    
    # Calculate Union Bounding Box
    # We calculate the min/max extent of both images in the new coordinate space
    min_z, max_z = min(0, dz), max(fz, dz + mz)
    min_y, max_y = min(0, dy), max(fy, dy + my)
    min_x, max_x = min(0, dx), max(fx, dx + mx)
    
    out_z, out_y, out_x = max_z - min_z, max_y - min_y, max_x - min_x
    
    # Create empty canvases
    padded_fixed = np.zeros((out_z, out_y, out_x), dtype=fixed_data.dtype)
    padded_moving = np.zeros((out_z, out_y, out_x), dtype=moving_data.dtype)
    
    # Paste Fixed Image (Offset by -min_bound)
    padded_fixed[-min_z:-min_z+fz, -min_y:-min_y+fy, -min_x:-min_x+fx] = fixed_data
    
    # Paste Moving Image (Offset by shift - min_bound)
    # Apply Sub-pixel Shift to Moving Image
    # We use linear interpolation (order=1) to avoid ringing artifacts on puncta
    if np.any(np.abs(shift_frac) > 0.01):
        print(f"Applying sub-pixel shift: {shift_frac}")
        order = 0 if is_label else 1
        moving_data_subpixel = ndi.shift(moving_data.astype(np.float32), shift_frac, order=order)
        # Cast back to original dtype
        if np.issubdtype(fixed_data.dtype, np.integer):
             moving_data_subpixel = np.clip(moving_data_subpixel, 0, np.iinfo(fixed_data.dtype).max).astype(fixed_data.dtype)
    else:
        moving_data_subpixel = moving_data

    # Paste Moving Image (Offset by integer shift - min_bound)
    mz_start, my_start, mx_start = dz - min_z, dy - min_y, dx - min_x
    padded_moving[mz_start:mz_start+mz, my_start:my_start+my, mx_start:mx_start+mx] = moving_data_subpixel
    
    return padded_fixed, padded_moving

def calculate_deformable_transform(fixed_data, moving_data, downsample_factor=16):
    """
    Calculates B-Spline deformable transform using SimpleITK.
    Returns the transform object, does not return the image.
    
    Args:
        fixed_data: (Z, Y, X) numpy array
        moving_data: (Z, Y, X) numpy array
        downsample_factor: Factor to downsample XY dimensions for speed
        
    Returns:
        transform: SimpleITK Transform object
    """
    print(f"Downsampling data by {downsample_factor}x (BinShrink) for registration calculation...")
    
    # Convert to SimpleITK images (Full Resolution)
    # ITK handles dimensions as (X, Y, Z) from numpy (Z, Y, X)
    fixed_img_full = sitk.GetImageFromArray(fixed_data.astype(np.float32))
    moving_img_full = sitk.GetImageFromArray(moving_data.astype(np.float32))
    
    # Use BinShrink for proper downsampling (averaging) and spacing handling
    # Shrink X and Y by factor, keep Z resolution (1)
    shrink_factors = [downsample_factor, downsample_factor, 1]
    fixed_img = sitk.BinShrink(fixed_img_full, shrink_factors)
    moving_img = sitk.BinShrink(moving_img_full, shrink_factors)
    
    # Initialize B-Spline Transform
    # Mesh size determines flexibility (lower = more rigid, higher = more flexible)
    transformDomainMeshSize = [8] * fixed_img.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_img, transformDomainMeshSize)
    
    # Set up Registration Method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation() 
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=10)
    R.SetInitialTransform(tx, True)
    R.SetInterpolator(sitk.sitkLinear)
    
    print(f"Starting B-Spline optimization on {fixed_img.GetSize()} volume (Sampling 1%, 10 iters)...")
    # Execute Registration
    outTx = R.Execute(fixed_img, moving_img)
    print("Registration finished.")
    
    return outTx

def apply_deformable_transform(moving_data, transform, fixed_reference_data, is_label=False):
    """
    Applies a calculated SimpleITK transform to an image array.
    """
    # Convert inputs to SimpleITK
    moving_img = sitk.GetImageFromArray(moving_data.astype(np.float32))
    
    # Optimization: We only need the grid (size/spacing), not the pixel data.
    # Using uint8 zeros saves massive memory compared to casting input to float32.
    fixed_ref = sitk.GetImageFromArray(np.zeros(fixed_reference_data.shape, dtype=np.uint8))
    
    # Setup Resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_ref)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    # Execute
    out_img = resampler.Execute(moving_img)
    
    result = sitk.GetArrayFromImage(out_img)
    
    if is_label:
        if np.issubdtype(moving_data.dtype, np.integer):
            return result.astype(moving_data.dtype)
        else:
            return result.astype(np.uint32)
        
    return result