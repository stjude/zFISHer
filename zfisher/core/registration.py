import numpy as np
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import SimpleITK as sitk
import warnings

from .. import constants

def _find_rough_shift_vector_voting(fixed_points, moving_points, n_limit=constants.RANSAC_N_LIMIT, bin_size=constants.RANSAC_BIN_SIZE):
    """
    Finds a rough alignment shift using brute-force vector voting.

    This method is robust to large displacements and is used to get a
    coarse initial alignment before refinement.

    Parameters
    ----------
    fixed_points : np.ndarray
        (N, 3) array of reference points.
    moving_points : np.ndarray
        (M, 3) array of points to be aligned.
    n_limit : int, optional
        The maximum number of points to use for voting to limit computation.
    bin_size : float, optional
        The size of the bins for discretizing displacement vectors.

    Returns
    -------
    np.ndarray
        A (3,) array representing the coarse shift vector (Z, Y, X).
    """
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

def _get_nearest_neighbor_pairs(fixed_points, moving_points, rough_shift, search_radius=constants.RANSAC_SEARCH_RADIUS):
    """
    Matches points based on nearest neighbors after applying a rough shift.

    Parameters
    ----------
    fixed_points : np.ndarray
        (N, 3) array of reference points.
    moving_points : np.ndarray
        (M, 3) array of points to be aligned.
    rough_shift : np.ndarray
        A (3,) coarse shift vector to apply to `moving_points` before matching.
    search_radius : float, optional
        The maximum distance to search for a nearest neighbor.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing (src, dst) points, where `src` are from `moving_points`
        and `dst` are their corresponding matches in `fixed_points`.
    """
    moving_shifted = moving_points + rough_shift
    
    tree = cKDTree(fixed_points)
    distances, indices = tree.query(moving_shifted, distance_upper_bound=search_radius)
    
    valid_mask = distances < float('inf')
    src = moving_points[valid_mask]
    dst = fixed_points[indices[valid_mask]]
    
    return src, dst

def _calculate_rmsd(src_points, dst_points, model, inliers):
    """
    Calculates the Root Mean Square Deviation (RMSD) for inlier points.

    Parameters
    ----------
    src_points : np.ndarray
        The source points used for RANSAC.
    dst_points : np.ndarray
        The destination points used for RANSAC.
    model : skimage.transform.AffineTransform
        The transformation model estimated by RANSAC.
    inliers : np.ndarray
        A boolean mask indicating the inlier points.

    Returns
    -------
    float
        The calculated RMSD value, or 0.0 if no inliers are found.
    """
    if inliers is None or not np.any(inliers):
        return 0.0

    inlier_src = src_points[inliers]
    inlier_dst = dst_points[inliers]
    transformed_src = model.predict(inlier_src)
    squared_distances = np.sum((transformed_src - inlier_dst) ** 2, axis=1)
    return np.sqrt(np.mean(squared_distances))

def _refine_shift_with_ransac(src_points, dst_points, residual_threshold=constants.RANSAC_RESIDUAL_THRESHOLD, max_trials=constants.RANSAC_MAX_TRIALS):
    """
    Refines the transformation using RANSAC to find a robust affine shift.

    Parameters
    ----------
    src_points : np.ndarray
        (K, 3) array of source points (from the moving set).
    dst_points : np.ndarray
        (K, 3) array of corresponding destination points (from the fixed set).
    residual_threshold : float, optional
        Maximum distance for a data point to be considered an inlier.
    max_trials : int, optional
        The maximum number of RANSAC iterations.

    Returns
    -------
    tuple[AffineTransform, np.ndarray] or tuple[None, None]
        The estimated model and a boolean mask of the inliers, or (None, None).
    """
    if len(src_points) < 4:
        return None

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No inliers found")
        model, inliers = ransac(
            (src_points, dst_points), AffineTransform, min_samples=4, 
            residual_threshold=residual_threshold, max_trials=max_trials
        )
    
    return model, inliers

def align_centroids_ransac(fixed_points, moving_points, max_distance=None, progress_callback=None):
    """
    Calculates the rigid shift between two point clouds.

    This function uses a two-step process:
    1. Coarse alignment via vector voting.
    2. Fine alignment using RANSAC on nearest-neighbor pairs.

    Parameters
    ----------
    fixed_points : np.ndarray
        (N, 3) array of centroids from the reference image (e.g., Round 1).
    moving_points : np.ndarray
        (M, 3) array of centroids from the image to be aligned (e.g., Round 2).
    progress_callback : callable, optional
        A function to report progress, e.g., `lambda p, m: print(f"{p}%: {m}")`.

    Returns
    -------
    tuple[np.ndarray, float]
        A tuple containing:
        - The calculated (Z, Y, X) shift vector.
        - The calculated RMSD of the inlier points.
    """
    # 1. Brute-force Vector Voting to find rough shift
    if progress_callback: progress_callback(10, "Finding rough alignment...")
    rough_shift = _find_rough_shift_vector_voting(fixed_points, moving_points)
    
    # 2. Refine using Nearest Neighbors
    if progress_callback: progress_callback(40, "Matching nearest neighbors...")
    src, dst = _get_nearest_neighbor_pairs(fixed_points, moving_points, rough_shift)
    
    if len(src) < 4:
        if progress_callback: progress_callback(100, "Done (not enough matches for RANSAC).")
        return rough_shift, 0.0

    # 3. Run RANSAC to refine the fit
    if progress_callback: progress_callback(70, "Running RANSAC to find best fit...")
    model, inliers = _refine_shift_with_ransac(src, dst)
    
    if model is None:
        if progress_callback: progress_callback(100, "Done (RANSAC failed, using rough shift).")
        return rough_shift, 0.0

    rmsd = _calculate_rmsd(src, dst, model, inliers)
    refined_shift = model.translation
    
    # Sanity checks on the refined shift
    deviation = np.linalg.norm(refined_shift - rough_shift)
    if deviation > constants.RANSAC_DEVIATION_THRESHOLD or np.any(np.isnan(refined_shift)):
        if progress_callback: progress_callback(100, "Done (RANSAC result invalid, reverted to rough shift).")
        return rough_shift, 0.0
        
    if progress_callback: progress_callback(100, "Done.")
    return refined_shift, rmsd

def align_and_pad_images(fixed_data, moving_data, shift_vector, is_label=False):
    """
    Aligns two 3D volumes based on a shift vector by padding.

    Creates a new canvas large enough to contain both volumes after alignment
    and pastes them into the correct positions. Handles sub-pixel shifts via
    interpolation.

    Parameters
    ----------
    fixed_data : np.ndarray
        The reference (Z, Y, X) image data.
    moving_data : np.ndarray
        The (Z, Y, X) image data to be aligned.
    shift_vector : np.ndarray
        The (Z, Y, X) shift to apply to `moving_data`.
    is_label : bool, optional
        If True, uses nearest-neighbor interpolation for label images.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing (padded_fixed, padded_moving), both of the same shape.
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
    Calculates a B-Spline deformable transform using SimpleITK.

    The registration is performed on downsampled images for speed.

    Parameters
    ----------
    fixed_data : np.ndarray
        The reference (Z, Y, X) image data.
    moving_data : np.ndarray
        The (Z, Y, X) image data to be warped.
    downsample_factor : int, optional
        The factor by which to downsample XY dimensions for faster registration.

    Returns
    -------
    SimpleITK.Transform
        The calculated B-spline transform.
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
    transformDomainMeshSize = [constants.DEFORMABLE_MESH_SIZE] * fixed_img.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed_img, transformDomainMeshSize)
    
    # Set up Registration Method
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation() 
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(constants.DEFORMABLE_SAMPLING_PERC)
    R.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=constants.DEFORMABLE_ITERATIONS)
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

    Parameters
    ----------
    moving_data : np.ndarray
        The (Z, Y, X) image data to warp.
    transform : SimpleITK.Transform
        The transform to apply.
    fixed_reference_data : np.ndarray
        A reference image defining the output grid (size, spacing, origin).
    is_label : bool, optional
        If True, uses nearest-neighbor interpolation.

    Returns
    -------
    np.ndarray
        The warped image data.
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