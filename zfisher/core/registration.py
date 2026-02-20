import numpy as np
from pathlib import Path
from skimage.measure import ransac
from skimage.transform import AffineTransform
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import SimpleITK as sitk
import warnings
import gc
import tifffile
from .session import set_processed_file

from . import session
from .. import constants

def calculate_session_registration(r1_centroids, r2_centroids, progress_callback=None):
    """
    Headless Orchestrator for Step 3.
    Calculates the shift and updates the global session data.
    """
    if r1_centroids is None or r2_centroids is None:
        return None, 0.0

    # 1. Execute the RANSAC math (already pure compute!)
    shift, rmsd = align_centroids_ransac(
        r1_centroids, 
        r2_centroids, 
        progress_callback=progress_callback
    )

    # 2. Update Session State
    # Note: Using .tolist() because JSON doesn't support numpy arrays
    session.update_data("shift", shift.tolist())
    session.update_data("registration_rmsd", float(rmsd))

    return shift, rmsd

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
    transformed_src = model(inlier_src)
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
        return None, None

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

def create_deformation_field(fixed_image_shape, transform, grid_spacing=50):
    """
    Generates a vector field representing the deformation.

    Creates a grid of points and transforms them to show the warp.

    Parameters
    ----------
    fixed_image_shape : tuple
        The shape (Z, Y, X) of the reference image space.
    transform : SimpleITK.Transform
        The calculated B-spline transform.
    grid_spacing : int
        The spacing between points in the grid in pixels.

    Returns
    -------
    np.ndarray
        (N, 2, 3) array for napari's Vectors layer. [start_point, vector]
    """
    # Create a grid of points in Z, Y, X order
    z_coords = np.arange(0, fixed_image_shape[0], grid_spacing)
    y_coords = np.arange(0, fixed_image_shape[1], grid_spacing)
    x_coords = np.arange(0, fixed_image_shape[2], grid_spacing)
    
    if len(z_coords) == 0: z_coords = [fixed_image_shape[0] // 2]
    if len(y_coords) == 0: y_coords = [fixed_image_shape[1] // 2]
    if len(x_coords) == 0: x_coords = [fixed_image_shape[2] // 2]

    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    start_points = np.vstack([zz.ravel(), yy.ravel(), xx.ravel()]).T

    start_points_sitk = start_points[:, ::-1].astype(float)
    transformed_points_sitk = [transform.TransformPoint(p) for p in start_points_sitk]
    end_points = np.array(transformed_points_sitk)[:, ::-1]

    vectors = end_points - start_points
    napari_vectors = np.stack([start_points, vectors], axis=1)
    return napari_vectors

# zfisher/core/registration.py

# ... (existing imports)

def generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp=True, progress_callback=None):
    """
    Core Orchestrator: Aligns, warps, and saves all channels.
    Headless-ready (uses callback instead of yield).
    """
    def update(val, msg):
        if progress_callback:
            progress_callback(val, msg)

    results = []
    # 1. Rigid Alignment
    update(0, "Matching and aligning channels...")
    aligned_pairs = _match_and_align_channels(r1_layers_data, r2_layers_data, shift)

    # 2. Deformable Transform Calculation
    transform = None
    has_dapi = constants.DAPI_CHANNEL_NAME in aligned_pairs
    
    if apply_warp and has_dapi:
        update(10, "Calculating deformable registration on DAPI...")
        dapi_pair = aligned_pairs[constants.DAPI_CHANNEL_NAME]
        transform = calculate_deformable_transform(dapi_pair['r1_data'], dapi_pair['r2_data'])
        update(35, "Deformable registration complete.")

        update(40, "Generating deformation field...")
        deformation_vectors = create_deformation_field(dapi_pair['r1_data'].shape, transform, grid_spacing=constants.DEFORMATION_GRID_SPACING)
        vector_layer_name = constants.DEFORMATION_FIELD_NAME
        vector_path = output_dir / f"{vector_layer_name}.npy"
        np.save(vector_path, deformation_vectors)
        set_processed_file(vector_layer_name, str(vector_path), layer_type='vectors')
        
        vector_meta = {'scale': dapi_pair['r1_meta']['scale']}
        results.append({
            'data': deformation_vectors, 'name': vector_layer_name, 'meta': vector_meta, 'type': 'vectors'
        })
    
    # 3. Per-channel Warping and Saving
    num_channels = len(aligned_pairs)
    start_progress = 40 if (apply_warp and has_dapi) else 10

    for i, (channel_name, pair_data) in enumerate(aligned_pairs.items()):
        prog = start_progress + int((i / num_channels) * (100 - start_progress))
        update(prog, f"Warping {channel_name}...")

        # Process the pair (Math + Save)
        result_pair = _process_channel_pair(channel_name, pair_data, transform, output_dir)
        results.extend(result_pair)
        gc.collect()
        
    update(100, "Canvas generation complete.")
    return results

def _match_and_align_channels(r1_layers_data, r2_layers_data, shift):
    """Pure math/logic: matches R1/R2 channels and performs initial rigid shift."""
    aligned_pairs = {}
    for r1 in r1_layers_data:
        channel_name = r1['name'].split("-")[-1].strip()
        r2 = next((l for l in r2_layers_data if channel_name in l['name']), None)
        if r2:
            is_label = r1.get('is_label', False)
            aligned_r1, aligned_r2 = align_and_pad_images(r1['data'], r2['data'], shift, is_label=is_label)
            aligned_pairs[channel_name] = {
                'r1_data': aligned_r1, 'r2_data': aligned_r2,
                'r1_meta': r1, 'r2_meta': r2, 'is_label': is_label
            }
    return aligned_pairs

def _process_channel_pair(channel_name, pair_data, transform, output_dir):
    """Applies deformable warp if needed and triggers the save."""
    r1_data, r2_data = pair_data['r1_data'], pair_data['r2_data']
    is_label = pair_data['is_label']
    
    final_r2 = r2_data
    r2_prefix = constants.ALIGNED_PREFIX
    if transform:
        final_r2 = apply_deformable_transform(r2_data, transform, r1_data, is_label=is_label)
        r2_prefix = constants.WARPED_PREFIX
        
    # Trigger the Save logic (The I/O part)
    _save_aligned_layer(r1_data, constants.ALIGNED_PREFIX, "R1", channel_name, output_dir, is_label)
    _save_aligned_layer(final_r2, r2_prefix, "R2", channel_name, output_dir, is_label)
        
    r1_result = {'data': r1_data, 'name': f"{constants.ALIGNED_PREFIX} R1 - {channel_name}", 'meta': pair_data['r1_meta'], 'type': 'labels' if is_label else 'image'}
    r2_result = {'data': final_r2, 'name': f"{r2_prefix} R2 - {channel_name}", 'meta': pair_data['r2_meta'], 'type': 'labels' if is_label else 'image'}
    return [r1_result, r2_result]

def _save_aligned_layer(data, prefix, round_id, channel_name, output_dir, is_label):
    """Internal helper to write warped files to disk and track in session."""
    if not output_dir: return
    layer_name = f"{prefix} {round_id} - {channel_name}"
    out_path = Path(output_dir) / f"{prefix}_{round_id}_{channel_name}.tif"
    tifffile.imwrite(out_path, data, compression='zlib')
    set_processed_file(layer_name, str(out_path), layer_type='labels' if is_label else 'image')