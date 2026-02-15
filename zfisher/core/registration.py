import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import SimpleITK as sitk
import warnings

def _find_rough_shift_vector_voting(fixed_points, moving_points, n_limit=2000, bin_size=5.0):
    """Finds a rough alignment shift using brute-force vector voting."""
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
    
    binned_diffs = np.round(diffs / bin_size).astype(int)
    
    unique_bins, counts = np.unique(binned_diffs, axis=0, return_counts=True)
    best_bin = unique_bins[np.argmax(counts)]
    return best_bin * bin_size

def _get_nearest_neighbor_pairs(fixed_points, moving_points, rough_shift, search_radius=100.0):
    """Matches points based on nearest neighbors after applying a rough shift."""
    moving_shifted = moving_points + rough_shift
    
    tree = cKDTree(fixed_points)
    distances, indices = tree.query(moving_shifted, distance_upper_bound=search_radius)
    
    valid_mask = distances < float('inf')
    src = moving_points[valid_mask]
    dst = fixed_points[indices[valid_mask]]
    
    return src, dst

def _refine_shift_with_ransac(src_points, dst_points, residual_threshold=50, max_trials=2000):
    """Refines the transformation using RANSAC."""
    if len(src_points) < 4:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No inliers found")
        model, inliers = ransac(
            (src_points, dst_points), AffineTransform, min_samples=4, 
            residual_threshold=residual_threshold, max_trials=max_trials
        )
    
    return model.translation if model else None

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
    rough_shift = _find_rough_shift_vector_voting(fixed_points, moving_points)
    
    # 2. Refine using Nearest Neighbors
    if progress_callback: progress_callback(40, "Matching nearest neighbors...")
    src, dst = _get_nearest_neighbor_pairs(fixed_points, moving_points, rough_shift)
    
    if len(src) < 4:
        if progress_callback: progress_callback(100, "Done (not enough matches for RANSAC).")
        return rough_shift

    # 3. Run RANSAC to refine the fit
    if progress_callback: progress_callback(70, "Running RANSAC to find best fit...")
    refined_shift = _refine_shift_with_ransac(src, dst)
    
    if refined_shift is None:
        if progress_callback: progress_callback(100, "Done (RANSAC failed, using rough shift).")
        return rough_shift

    # Sanity checks on the refined shift
    deviation = np.linalg.norm(refined_shift - rough_shift)
    if deviation > 500.0 or np.any(np.isnan(refined_shift)):
        if progress_callback: progress_callback(100, "Done (RANSAC result invalid, reverted to rough shift).")
        return rough_shift
        
    if progress_callback: progress_callback(100, "Done.")
    return refined_shift

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