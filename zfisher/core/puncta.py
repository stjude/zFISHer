import logging
import numpy as np
from pathlib import Path
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from .. import constants

logger = logging.getLogger(__name__)


def _save_puncta_csv(final_data, output_path, layer_name):
    """Persist a puncta array to CSV in canonical column order and register it.

    Parameters
    ----------
    final_data : np.ndarray
        (N, 7) array with columns Z, Y, X, Nucleus_ID, Intensity, SNR, puncta_id.
    output_path : str or Path
        Destination CSV path.
    layer_name : str or None
        Layer name used to derive the session-registry key (falls back to the
        file stem when None).

    Notes
    -----
    The on-disk column order follows ``constants.PUNCTA_CSV_COLUMNS`` so every
    writer produces an identical schema (puncta_id first). Z/Y/X/Nucleus_ID/
    Intensity/SNR keep their original float representation; ``puncta_id`` is
    written as an integer.
    """
    from . import session
    import pandas as pd

    df = pd.DataFrame(final_data[:, :6], columns=['Z', 'Y', 'X', 'Nucleus_ID', 'Intensity', 'SNR'])
    df['puncta_id'] = final_data[:, 6].astype(int)
    df['Source'] = 'auto'
    df = df[constants.PUNCTA_CSV_COLUMNS]
    df.to_csv(output_path, index=False)

    session_key = layer_name if layer_name else Path(output_path).stem
    session.set_processed_file(
        session_key, str(output_path), "points",
        metadata={'format': 'csv', 'subtype': 'puncta_csv'}
    )


def lookup_label_ids(coords, points_scale, points_translate,
                     mask_data, mask_scale, mask_translate):
    """Return the mask label under each point, in the mask's voxel space.

    Converts points from their own layer (data) coordinates to world
    coordinates (``coords * points_scale + points_translate``) and then into the
    mask's voxel indices (``(world - mask_translate) / mask_scale``), so the
    lookup is correct even when the points and the mask have different scale /
    translate. Points outside the mask bounds get label 0.

    Parameters
    ----------
    coords : array-like, shape (N, 3)
        Point coordinates in the points layer's data space (Z, Y, X).
    points_scale, points_translate : array-like, shape (3,)
        The points layer's scale and translate.
    mask_data : np.ndarray
        3D label array.
    mask_scale, mask_translate : array-like, shape (3,)
        The mask layer's scale and translate.

    Returns
    -------
    np.ndarray, shape (N,)
        Integer label at each point (0 = background or out of bounds).
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.zeros(0, dtype=int)
    world = coords * np.asarray(points_scale, dtype=float) + np.asarray(points_translate, dtype=float)
    voxel = np.round((world - np.asarray(mask_translate, dtype=float)) / np.asarray(mask_scale, dtype=float)).astype(int)
    shape = np.asarray(mask_data.shape)
    in_bounds = np.all((voxel >= 0) & (voxel < shape), axis=1)
    clipped = np.clip(voxel, 0, shape - 1)
    labels = mask_data[clipped[:, 0], clipped[:, 1], clipped[:, 2]]
    return np.where(in_bounds, labels, 0).astype(int)


def calculate_spot_quality(image_data, coords, radius=2, progress_callback=None,
                           progress_range=(50, 70)):
    """
    Calculates Signal-to-Noise Ratio (SNR) and Intensity for each puncta.
    Essential for validating counts in crowded fields.
    """
    stats = []
    z_max, y_max, x_max = image_data.shape
    n_coords = len(coords)
    pmin, pmax = progress_range
    last_pct = -1

    for i, coord in enumerate(coords):
        z, y, x = coord.astype(int)
        z = np.clip(z, 0, z_max - 1)
        y = np.clip(y, 0, y_max - 1)
        x = np.clip(x, 0, x_max - 1)
        z_slice = image_data[z]
        y_min, y_max_p = max(0, y-radius), min(y_max, y+radius+1)
        x_min, x_max_p = max(0, x-radius), min(x_max, x+radius+1)

        local_crop = z_slice[y_min:y_max_p, x_min:x_max_p]

        peak_intensity = image_data[z, y, x]
        background = np.median(local_crop) if local_crop.size > 0 else 1.0
        snr = peak_intensity / background if background > 0 else 0

        stats.append([peak_intensity, snr])

        if progress_callback and n_coords > 0:
            pct = pmin + int((i + 1) / n_coords * (pmax - pmin))
            if pct != last_pct:
                last_pct = pct
                progress_callback(pct, f"Quality metrics: {i + 1}/{n_coords} spots...")

    return np.array(stats)

def preprocess_white_tophat(image_data, radius=constants.PUNCTA_TOPHAT_RADIUS,
                            progress_callback=None, progress_range=(25, 40)):
    """Applies slice-by-slice background subtraction."""
    selem = disk(radius)
    n_slices = len(image_data)
    pmin, pmax = progress_range
    processed_slices = []
    last_pct = -1
    for i, image_slice in enumerate(image_data):
        processed_slices.append(white_tophat(image_slice, selem))
        if progress_callback and n_slices > 0:
            pct = pmin + int((i + 1) / n_slices * (pmax - pmin))
            if pct != last_pct:
                last_pct = pct
                progress_callback(pct, f"Background subtraction: slice {i + 1}/{n_slices}...")
    return np.stack(processed_slices, axis=0)

def _refine_radial_symmetry_3d(image_data, coord, radius, spacing):
    """Refine a single candidate point using 3D radial symmetry (Parthasarathy, 2012).

    Computes intensity gradients in a local patch, then solves a weighted
    least-squares problem to find the point where gradient rays converge,
    yielding sub-voxel localization.
    """
    z, y, x = np.round(coord).astype(int)
    nz, ny, nx = image_data.shape

    z0, z1 = max(0, z - radius), min(nz, z + radius + 1)
    y0, y1 = max(0, y - radius), min(ny, y + radius + 1)
    x0, x1 = max(0, x - radius), min(nx, x + radius + 1)

    patch = image_data[z0:z1, y0:y1, x0:x1].astype(np.float64)
    if patch.size < 8:
        return coord.astype(float)

    # Compute gradients scaled by voxel spacing
    gz, gy, gx = np.gradient(patch, spacing[0], spacing[1], spacing[2])
    mag = np.sqrt(gz**2 + gy**2 + gx**2)

    # Threshold: only use voxels with meaningful gradients
    mask = mag > (np.max(mag) * 0.1)
    if np.sum(mask) < 4:
        return coord.astype(float)

    # Coordinates of each voxel in the patch (physical units)
    pz, py, px = np.mgrid[0:patch.shape[0], 0:patch.shape[1], 0:patch.shape[2]]
    pz = pz.astype(float) * spacing[0]
    py = py.astype(float) * spacing[1]
    px = px.astype(float) * spacing[2]

    # Extract masked gradient vectors and positions
    gz_m = gz[mask]
    gy_m = gy[mask]
    gx_m = gx[mask]
    mag_m = mag[mask]
    pz_m = pz[mask]
    py_m = py[mask]
    px_m = px[mask]

    # Normalize gradient directions
    gz_n = gz_m / mag_m
    gy_n = gy_m / mag_m
    gx_n = gx_m / mag_m

    # Weights: gradient magnitude squared (stronger gradients = more reliable)
    w = mag_m ** 2

    # Weighted least-squares: find point minimizing distance to all gradient rays.
    # For each voxel i at position p_i with unit gradient direction d_i,
    # the projection matrix M_i = I - d_i * d_i^T projects onto the plane
    # perpendicular to d_i. The center minimizes Σ w_i * |M_i * (c - p_i)|^2.
    # Solution: (Σ w_i * M_i) * c = Σ w_i * M_i * p_i
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(len(gz_n)):
        d = np.array([gz_n[i], gy_n[i], gx_n[i]])
        M = np.eye(3) - np.outer(d, d)
        wM = w[i] * M
        p = np.array([pz_m[i], py_m[i], px_m[i]])
        A += wM
        b += wM @ p

    try:
        center_physical = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return coord.astype(float)

    # Convert back to voxel coordinates and offset to global position
    center_voxel = center_physical / spacing
    result = center_voxel + np.array([z0, y0, x0], dtype=float)

    # Reject if refined position is outside the patch (bad convergence)
    if (result[0] < z0 or result[0] >= z1 or
        result[1] < y0 or result[1] >= y1 or
        result[2] < x0 or result[2] >= x1):
        return coord.astype(float)

    return result


def _detect_radial_symmetry(image_data, threshold_rel, sigma, z_scale=1.0):
    """3D radial symmetry detection (Parthasarathy, 2012) with sub-voxel precision.

    Uses local maxima as initial candidates, then refines each to sub-voxel
    accuracy by solving a weighted least-squares gradient ray intersection.
    """
    if sigma > 0:
        smoothed = gaussian(image_data, sigma=(sigma * z_scale, sigma, sigma), preserve_range=True)
    else:
        smoothed = image_data

    # Find initial candidates via local maxima
    candidates = peak_local_max(smoothed, min_distance=1, threshold_rel=threshold_rel)
    if len(candidates) == 0:
        return np.empty((0, 3))

    spacing = np.array([z_scale, 1.0, 1.0])
    radius = max(2, int(np.ceil(sigma))) if sigma > 0 else 2

    refined = np.empty((len(candidates), 3), dtype=float)
    for i, coord in enumerate(candidates):
        refined[i] = _refine_radial_symmetry_3d(image_data, coord, radius, spacing)

    return refined

def _detect_spots_local_maxima(image_data, min_distance, threshold_rel, sigma):
    """Standard intensity-based peak finding."""
    if sigma > 0:
        image_data = gaussian(image_data, sigma=sigma, preserve_range=True)
    return peak_local_max(image_data, min_distance=min_distance, 
                          threshold_rel=threshold_rel, exclude_border=False)

def _detect_spots_log(image_data, threshold_rel, sigma, z_scale=1.0):
    """Anisotropic Laplacian of Gaussian for PSF compensation."""
    s = sigma if sigma > 0 else 1.0
    sigma_vec = (s * z_scale, s, s)
    max_val = np.max(image_data)
    image_norm = image_data.astype(np.float32) / max_val if max_val > 0 else image_data
    blobs = blob_log(image_norm, min_sigma=sigma_vec, max_sigma=[sv * 1.5 for sv in sigma_vec], 
                     num_sigma=2, threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def _detect_spots_dog(image_data, threshold_rel, sigma, z_scale=1.0):
    """Anisotropic Difference of Gaussian for fast 3D detection."""
    s = sigma if sigma > 0 else 1.0
    sigma_vec = (s * z_scale, s, s)
    max_val = np.max(image_data)
    image_norm = image_data.astype(np.float32) / max_val if max_val > 0 else image_data
    blobs = blob_dog(image_norm, min_sigma=sigma_vec, max_sigma=[sv * 1.6 for sv in sigma_vec], 
                     threshold=threshold_rel)
    return blobs[:, :3].astype(int) if len(blobs) > 0 else np.empty((0, 3))

def detect_spots_3d(image_data, method="Local Maxima", progress_callback=None, **kwargs):
    """Main entry point for puncta detection on a single image volume."""
    logger.info("Puncta detection: method=%s, shape=%s, params=%s", method, image_data.shape, kwargs)
    if kwargs.get('use_tophat', False):
        if progress_callback: progress_callback(25, "Subtracting background (top-hat)...")
        image_data = preprocess_white_tophat(image_data, radius=kwargs.get('tophat_radius', 10),
                                             progress_callback=progress_callback, progress_range=(25, 40))

    if method == "Radial Symmetry":
        return _detect_radial_symmetry(image_data, kwargs.get('threshold_rel', 0.1), kwargs.get('sigma', 1.0), z_scale=kwargs.get('z_scale', 1.0))
    elif method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, kwargs.get('min_distance', 3), 
                                          kwargs.get('threshold_rel', 0.1), kwargs.get('sigma', 1.0))
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, kwargs.get('threshold_rel', 0.1), 
                                 kwargs.get('sigma', 1.0), z_scale=kwargs.get('z_scale', 1.0))
    elif method == "Difference of Gaussian":
        return _detect_spots_dog(image_data, kwargs.get('threshold_rel', 0.1), 
                                 kwargs.get('sigma', 1.0), z_scale=kwargs.get('z_scale', 1.0))
    return np.empty((0, 3))

def transform_puncta_to_aligned_space(raw_puncta, round_id, shift, canvas_offset,
                                       bspline_transform=None, consensus_mask=None,
                                       remove_extranuclear=True,
                                       output_path=None, layer_name=None,
                                       puncta_ids=None,
                                       progress_callback=None):
    """
    Transform raw-space puncta coordinates into aligned canvas space and
    reassign nucleus IDs from the consensus mask.

    Parameters
    ----------
    raw_puncta : np.ndarray
        (N, 6) array from process_puncta_detection: Z, Y, X, Nucleus_ID,
        Intensity, SNR. A 7th column, if present, is treated as the stable
        ``puncta_id`` and carried through (takes precedence over ``puncta_ids``).
    round_id : str
        "R1" or "R2".
    shift : np.ndarray
        (3,) registration shift vector (Z, Y, X).
    canvas_offset : np.ndarray
        (3,) canvas offset from align_and_pad_images (typically negative or zero).
    bspline_transform : SimpleITK.Transform, optional
        The B-spline transform for R2 warping (fixed->moving direction). Ignored for R1.
    consensus_mask : np.ndarray, optional
        3D label array in aligned space for nucleus ID reassignment.
    output_path : Path, optional
        If provided, saves the transformed puncta CSV.
    layer_name : str, optional
        Session key for the saved file.
    puncta_ids : array-like, optional
        Stable per-punctum identifiers aligned 1:1 with ``raw_puncta`` rows.
        Used when ``raw_puncta`` has no 7th id column. If neither is supplied
        (or the length does not match), a positional ``arange`` is assigned.
        The same keep-mask used for extranuclear filtering is applied to the
        ids, so surviving puncta retain their original identity.
    progress_callback : callable, optional
        ``callback(pct, msg)``

    Returns
    -------
    np.ndarray
        (M, 7) array: Z, Y, X, Nucleus_ID, Intensity, SNR, puncta_id in aligned space.
    """
    if progress_callback:
        progress_callback(0, "Transforming puncta coordinates...")

    if len(raw_puncta) == 0:
        if progress_callback:
            progress_callback(100, "No puncta to transform.")
        return np.empty((0, 7))

    coords = raw_puncta[:, :3].copy()  # Z, Y, X
    quality = raw_puncta[:, 4:6].copy()  # Intensity, SNR (keep from raw detection)

    # Stable puncta_id: prefer a 7th column on raw_puncta, else the explicit
    # puncta_ids arg, else a positional fallback. Carried through filtering so a
    # surviving punctum keeps the identity it was assigned at detection.
    if raw_puncta.shape[1] >= 7:
        ids = raw_puncta[:, 6].astype(float).copy()
    elif puncta_ids is not None and len(puncta_ids) == len(raw_puncta):
        ids = np.asarray(puncta_ids, dtype=float).copy()
    else:
        ids = np.arange(len(raw_puncta), dtype=float)

    # Apply rigid transform to canvas space
    shift = np.asarray(shift, dtype=float)
    canvas_offset = np.asarray(canvas_offset, dtype=float)

    # Coordinate space transformations:
    # 1. Raw puncta coords are in raw image voxel space
    # 2. For R1: subtract canvas_offset to move into padded canvas coordinates
    # 3. For R2: add shift (rigid alignment), subtract canvas_offset, then
    #    optionally apply inverse B-spline warp (moving->fixed space)
    # After transform, coords are in the consensus mask's coordinate system.
    if round_id == "R1":
        aligned_coords = coords - canvas_offset
    else:  # R2
        aligned_coords = coords + shift - canvas_offset

        # Apply inverse B-spline warp
        if bspline_transform is not None:
            if progress_callback:
                progress_callback(20, f"Inverting B-spline for {len(aligned_coords)} points...")
            from .registration import transform_points_inverse_bspline
            aligned_coords = transform_points_inverse_bspline(aligned_coords, bspline_transform)

    if progress_callback:
        progress_callback(60, "Reassigning nucleus IDs from consensus mask...")

    # Reassign nucleus IDs from consensus mask
    if consensus_mask is not None:
        mask_shape = np.array(consensus_mask.shape)
        coords_int = np.round(aligned_coords).astype(int)

        # Bounds check
        in_bounds = np.all((coords_int >= 0) & (coords_int < mask_shape), axis=1)

        # Clip for safe indexing, then zero out-of-bounds
        for dim in range(3):
            coords_int[:, dim] = np.clip(coords_int[:, dim], 0, mask_shape[dim] - 1)

        indices = tuple(coords_int.T)
        nucleus_ids = consensus_mask[indices]
        nucleus_ids[~in_bounds] = 0

        # Filter to nuclear puncta only (if enabled)
        if remove_extranuclear:
            keep = nucleus_ids > 0
            aligned_coords = aligned_coords[keep]
            nucleus_ids = nucleus_ids[keep]
            quality = quality[keep]
            ids = ids[keep]

        if len(aligned_coords) == 0:
            if progress_callback:
                progress_callback(100, "No puncta in consensus nuclei.")
            return np.empty((0, 7))
    else:
        nucleus_ids = np.zeros(len(aligned_coords))

    final_data = np.column_stack([aligned_coords, nucleus_ids, quality, ids])

    if output_path:
        if progress_callback:
            progress_callback(90, "Saving transformed puncta...")
        _save_puncta_csv(final_data, output_path, layer_name)

    if progress_callback:
        progress_callback(100, f"Done. {len(final_data)} puncta in aligned space.")
    return final_data


def process_puncta_detection(image_data, mask_data=None, voxels=None, params=None, output_path=None, progress_callback=None, layer_name=None):
    """Orchestrates detection, quality mapping, and session persistence with CSV tags."""
    params = params or {}

    if 'z_scale' not in params and voxels is not None:
        params['z_scale'] = voxels[0] / voxels[2]

    method = params.get('method', 'Local Maxima')
    if progress_callback: progress_callback(-1, f"Detecting spots ({method})...")
    coords = detect_spots_3d(image_data, progress_callback=progress_callback, **params)
    if progress_callback: progress_callback(45, f"Detection complete. Found {len(coords)} candidates.")
    if len(coords) == 0:
        if progress_callback: progress_callback(100, "No spots found.")
        return np.empty((0, 7))

    # Assign a stable puncta_id once, at detection, BEFORE any extranuclear
    # filtering. Surviving puncta keep this identity through filtering, the
    # world-space transform, and reloads (the id is persisted in every CSV).
    puncta_ids = np.arange(len(coords), dtype=float)

    if progress_callback: progress_callback(50, f"Computing quality metrics for {len(coords)} spots...")
    quality_metrics = calculate_spot_quality(image_data, coords,
                                             progress_callback=progress_callback,
                                             progress_range=(50, 70))

    if progress_callback: progress_callback(70, "Assigning nucleus IDs...")
    # Clip coordinates to valid mask/image bounds before indexing
    coords_int = coords.astype(int)
    for dim in range(3):
        coords_int[:, dim] = np.clip(coords_int[:, dim], 0, image_data.shape[dim] - 1)
    indices = tuple(coords_int.T)
    nucleus_ids = mask_data[indices] if mask_data is not None else np.zeros(len(coords))

    # Filter out extranuclear puncta if requested
    nuclei_only = params.get('nuclei_only', True)
    if nuclei_only and mask_data is not None:
        if progress_callback: progress_callback(80, "Filtering extranuclear puncta...")
        keep = nucleus_ids > 0
        coords = coords[keep]
        nucleus_ids = nucleus_ids[keep]
        quality_metrics = quality_metrics[keep]
        puncta_ids = puncta_ids[keep]
        if len(coords) == 0:
            if progress_callback: progress_callback(100, "No nuclear puncta found.")
            return np.empty((0, 7))

    # Combined Data: Z, Y, X, Nucleus_ID, Intensity, SNR, puncta_id
    final_data = np.column_stack([coords, nucleus_ids, quality_metrics, puncta_ids])

    if output_path:
        if progress_callback: progress_callback(90, "Saving results...")
        _save_puncta_csv(final_data, output_path, layer_name)

    if progress_callback: progress_callback(100, f"Done. Found {len(final_data)} spots.")
    return final_data