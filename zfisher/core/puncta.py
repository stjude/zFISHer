import logging
import numpy as np
from pathlib import Path
from skimage.feature import peak_local_max, blob_log, blob_dog
from skimage.filters import gaussian
from skimage.morphology import white_tophat, disk
from .. import constants
from . import outputs
from .errors import SchemaError
from .timing import timed_stage, n_rows

logger = logging.getLogger(__name__)


class PunctaTableError(SchemaError):
    """A puncta table whose feature rows do not match its points one to one.
    Written, its values would sit on the wrong puncta, so it never is."""


def repeated_puncta_ids(features):
    """The ``puncta_id`` values that more than one row of ``features`` carries."""
    import pandas as pd
    if features is None or 'puncta_id' not in getattr(features, 'columns', ()):
        return []
    ids = pd.to_numeric(features['puncta_id'], errors='coerce')
    return sorted({int(i) for i in ids[ids.notna() & ids.duplicated(keep=False)]})


def puncta_table(coords, features=None):
    """The CSV table of a puncta layer: ``constants.PUNCTA_CSV_COLUMNS`` first,
    in that order, then any other feature columns.

    ``coords`` is (N, 3) Z, Y, X; ``features`` a DataFrame with one row per
    point. A missing ``Source`` column, or a row without a value, is written as
    ``unknown``: nothing recorded how that punctum was made.

    Raises ``PunctaTableError`` when ``features`` has a different number of
    rows than there are points: the two are joined by position, so every value
    after the first missing or extra row would land on the wrong punctum.
    """
    import pandas as pd

    coords_df = pd.DataFrame(np.asarray(coords, dtype=float).reshape(-1, 3), columns=['Z', 'Y', 'X'])
    if features is not None and len(features.columns) and len(features) != len(coords_df):
        raise PunctaTableError(f"{len(coords_df)} points but {len(features)} feature rows")
    if features is None or len(features.columns) == 0:
        features = pd.DataFrame(index=range(len(coords_df)))
    features = features.reset_index(drop=True).drop(columns=['Z', 'Y', 'X'], errors='ignore')
    df = pd.concat([features, coords_df], axis=1)
    if 'Source' not in df.columns:
        df['Source'] = constants.PUNCTA_SOURCE_UNKNOWN
    else:
        df['Source'] = df['Source'].where(df['Source'].notna() & (df['Source'] != ''),
                                          constants.PUNCTA_SOURCE_UNKNOWN)
    ordered = [c for c in constants.PUNCTA_CSV_COLUMNS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def write_puncta_csv(path, coords, features=None):
    """Write a puncta layer's points and features to ``path`` as ``puncta_table``,
    and record its new hash if the file is in the session's provenance."""
    outputs.write_csv(puncta_table(coords, features), path)
    _record_rewrite(path)


# --- Provenance ------------------------------------------------------------------
#
# Puncta CSVs are recorded in the session's provenance (core.provenance): round,
# channel, coordinate space (raw:R1, raw:R2 or aligned), image and mask, detection
# parameters, what the file was derived from, who ran it, and the puncta counts by
# Source. A raw CSV and the aligned CSV made from it share puncta_id, so each
# aligned punctum can be traced to its detection.

def points_fingerprint(csv_path, with_source=True):
    """A hash of the puncta in a puncta CSV: each row's ``puncta_id``, Z, Y, X
    (to 1e-6 voxel) and, with ``with_source``, ``Source``, in id order.

    Two files with the same puncta in the same places have the same
    fingerprint, whatever their ``Nucleus_ID``, ``Intensity`` and ``SNR``:
    ``Nucleus_ID`` is re-derived from the consensus mask whenever a session is
    opened or a report exported, and that rewrite is not an edit."""
    import hashlib
    import pandas as pd
    df = pd.read_csv(csv_path, float_precision="round_trip")
    ids = pd.to_numeric(df["puncta_id"], errors="coerce") if "puncta_id" in df.columns else pd.Series(range(len(df)))
    order = np.argsort(ids.to_numpy(float), kind="stable")
    source = (df["Source"].fillna(constants.PUNCTA_SOURCE_UNKNOWN).astype(str) if "Source" in df.columns
              else pd.Series([constants.PUNCTA_SOURCE_UNKNOWN] * len(df)))
    digest = hashlib.sha256()
    for i in order:
        row = f"{ids.iloc[i]:.0f},{df['Z'].iloc[i]:.6f},{df['Y'].iloc[i]:.6f},{df['X'].iloc[i]:.6f}"
        digest.update((row + (f",{source.iloc[i]}" if with_source else "") + "\n").encode())
    return digest.hexdigest()


def record_puncta(csv_path, event, **fields):
    """Record the puncta CSV that zFISHer just wrote at ``csv_path``, with its
    puncta counts by ``Source``; see ``provenance.record_artifact``."""
    import pandas as pd
    from . import provenance

    counts = {}
    try:
        df = pd.read_csv(csv_path)
        source = (df["Source"].fillna(constants.PUNCTA_SOURCE_UNKNOWN) if "Source" in df.columns
                  else pd.Series([constants.PUNCTA_SOURCE_UNKNOWN] * len(df), dtype=object))
        counts = {"n_puncta": int(len(df)),
                  "n_by_source": {str(k): int(v) for k, v in source.value_counts().items()}}
    except (OSError, ValueError) as exc:
        logger.warning("Could not count puncta in %s: %s", csv_path, exc)
    return provenance.record_artifact(csv_path, event, kind="puncta_csv", **counts, **fields)


def _record_rewrite(csv_path):
    """After zFISHer rewrote a recorded puncta CSV, record its new hash and
    counts, so ``provenance.verify`` keeps meaning "changed outside zFISHer"."""
    from . import provenance
    if provenance.has_record(csv_path):
        record_puncta(csv_path, "saved")


def _save_puncta_csv(final_data, output_path, layer_name, sources=None):
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
    sources : array-like of str, optional
        The ``Source`` of each row. Without it every row is ``auto``, which is
        right only for rows a detector just produced.

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
    df['Source'] = constants.PUNCTA_SOURCE_AUTO if sources is None else np.asarray(sources, dtype=object)
    df = df[constants.PUNCTA_CSV_COLUMNS]
    outputs.write_csv(df, output_path)
    _record_rewrite(output_path)

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


def nucleus_ids_from_mask(coords, mask):
    """The nucleus of each punctum: the label of ``mask`` at the punctum's
    nearest voxel, 0 outside every nucleus or outside the mask. ``coords`` are
    in the mask's voxel grid, as aligned puncta CSVs are for the consensus
    mask. The rule the GUI's resync uses (``lookup_label_ids``)."""
    return lookup_label_ids(coords, (1, 1, 1), (0, 0, 0), mask, (1, 1, 1), (0, 0, 0))


def check_nucleus_ids(csv_path, mask):
    """How well the ``Nucleus_ID`` column of the puncta CSV at ``csv_path``
    agrees with ``mask``: a dict with ``file``, ``rows``, ``agree`` (rows whose
    stored ID is the mask's), ``fraction``, ``zero_but_inside`` (stored 0, yet
    inside a nucleus) and ``repeated_ids``. A file without the column agrees
    nowhere."""
    import pandas as pd
    table = pd.read_csv(csv_path, float_precision="round_trip")
    derived = nucleus_ids_from_mask(table[["Z", "Y", "X"]].to_numpy(float), mask)
    stored = (pd.to_numeric(table["Nucleus_ID"], errors="coerce").fillna(-1).to_numpy().astype(np.int64)
              if "Nucleus_ID" in table.columns else np.full(len(table), -1, dtype=np.int64))
    agree = int((stored == derived).sum())
    return {"file": Path(csv_path).name, "rows": len(table), "agree": agree,
            "fraction": agree / len(table) if len(table) else 1.0,
            "zero_but_inside": int(((stored == 0) & (derived > 0)).sum()),
            "repeated_ids": repeated_puncta_ids(table)}


def repair_nucleus_ids(csv_path, mask, out_path):
    """Write the puncta CSV at ``csv_path`` to ``out_path`` with every
    ``Nucleus_ID`` read again from ``mask`` (``nucleus_ids_from_mask``); every
    other column is copied unchanged. Returns ``check_nucleus_ids`` of the
    original."""
    import pandas as pd
    before = check_nucleus_ids(csv_path, mask)
    table = pd.read_csv(csv_path, float_precision="round_trip")
    coords = table[["Z", "Y", "X"]].to_numpy(float)
    features = table.drop(columns=["Z", "Y", "X"]).assign(Nucleus_ID=nucleus_ids_from_mask(coords, mask))
    outputs.write_csv(puncta_table(coords, features), out_path)
    return before


def aligned_puncta_csvs(reports_dir):
    """The aligned and warped puncta CSVs in ``reports_dir``: the files whose
    puncta live in the consensus mask's space."""
    prefixes = tuple(constants.puncta_csv_stem(f"{p} ") for p in (constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))
    return sorted(p for p in Path(reports_dir).glob(f"*{constants.PUNCTA_SUFFIX}.csv") if p.name.startswith(prefixes))


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

def preprocess_white_tophat(image_data, radius=15,
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

# What the detectors use for a setting they are not given. These are not the
# widget's starting values (constants.PUNCTA_*): a batch row that leaves a
# setting empty gets these. process_puncta_detection fills them in before it
# detects, so whatever records its parameters records the values used.
DETECTOR_DEFAULTS = {
    'method': 'Local Maxima',
    'threshold_rel': 0.1,
    'min_distance': 3,
    'sigma': 1.0,
    'use_tophat': False,
    'tophat_radius': 10,
    'nuclei_only': True,
    'z_edge_exclude': 0,
}


def fill_detection_defaults(params, voxels=None):
    """Add to ``params``, in place, every setting it lacks, with the value
    detection uses for it (``DETECTOR_DEFAULTS``; ``z_scale`` from ``voxels``,
    else 1.0). Returns ``params``."""
    for key, value in DETECTOR_DEFAULTS.items():
        params.setdefault(key, value)
    if 'z_scale' not in params:
        params['z_scale'] = voxels[0] / voxels[2] if voxels is not None else 1.0
    return params


def detect_spots_3d(image_data, method="Local Maxima", progress_callback=None, **kwargs):
    """Main entry point for puncta detection on a single image volume."""
    logger.info("Puncta detection: method=%s, shape=%s, params=%s", method, image_data.shape, kwargs)
    kwargs = fill_detection_defaults(dict(kwargs, method=method))
    if kwargs['use_tophat']:
        if progress_callback: progress_callback(25, "Subtracting background (top-hat)...")
        image_data = preprocess_white_tophat(image_data, radius=kwargs['tophat_radius'],
                                             progress_callback=progress_callback, progress_range=(25, 40))

    if method == "Radial Symmetry":
        return _detect_radial_symmetry(image_data, kwargs['threshold_rel'], kwargs['sigma'], z_scale=kwargs['z_scale'])
    elif method == "Local Maxima":
        return _detect_spots_local_maxima(image_data, kwargs['min_distance'], kwargs['threshold_rel'], kwargs['sigma'])
    elif method == "Laplacian of Gaussian":
        return _detect_spots_log(image_data, kwargs['threshold_rel'], kwargs['sigma'], z_scale=kwargs['z_scale'])
    elif method == "Difference of Gaussian":
        return _detect_spots_dog(image_data, kwargs['threshold_rel'], kwargs['sigma'], z_scale=kwargs['z_scale'])
    return np.empty((0, 3))

@timed_stage("puncta_transform",
             fields=lambda a: {"round": a["round_id"], "layer": a["layer_name"]},
             result_fields=n_rows)
def transform_puncta_to_aligned_space(raw_puncta, round_id, shift, canvas_offset,
                                       bspline_transform=None, consensus_mask=None,
                                       remove_extranuclear=True,
                                       output_path=None, layer_name=None,
                                       puncta_ids=None, sources=None,
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
    sources : array-like of str, optional
        The ``Source`` of each ``raw_puncta`` row, filtered with the rows and
        written to the CSV. Without it every row is saved as ``auto``, so pass
        it whenever the puncta may include manual ones (``sources_of`` maps
        the returned rows back to it).
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
    if sources is not None:
        sources = np.asarray(sources, dtype=object)
        if len(sources) != len(raw_puncta):
            raise ValueError(f"{len(sources)} sources for {len(raw_puncta)} puncta")

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
            if sources is not None:
                sources = sources[keep]

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
        _save_puncta_csv(final_data, output_path, layer_name, sources=sources)

    if progress_callback:
        progress_callback(100, f"Done. {len(final_data)} puncta in aligned space.")
    return final_data


def transform_puncta_csv(csv_path, round_id, registration, consensus_mask=None,
                         remove_extranuclear=True, output_path=None, layer_name=None):
    """Transform the raw-space puncta in ``csv_path`` into aligned space with a
    saved registration (``registration.load_registration()``), without
    registering again. Returns ``(data, sources)``: the (M, 7) array of
    ``transform_puncta_to_aligned_space`` and each row's ``Source``.

    The CSV is a puncta table (``puncta_table``) in that round's raw voxel
    coordinates; ``puncta_id`` and ``Source`` are carried to the output."""
    import pandas as pd

    # round_trip: the default parser can change a float's last bit, so a CSV
    # rewritten from what it read would differ from the one it was given.
    df = pd.read_csv(csv_path, float_precision="round_trip")
    n = len(df)

    def column(name, default):
        return df[name].to_numpy(dtype=float) if name in df.columns else np.full(n, default, dtype=float)

    raw = np.column_stack([df[['Z', 'Y', 'X']].to_numpy(dtype=float).reshape(-1, 3),
                           column('Nucleus_ID', 0), column('Intensity', 0), column('SNR', 0)])
    ids = column('puncta_id', np.nan) if 'puncta_id' in df.columns else np.arange(n, dtype=float)
    sources = puncta_table(raw[:, :3], df.drop(columns=['Z', 'Y', 'X']))['Source'].to_numpy(dtype=object)
    data = transform_puncta_to_aligned_space(
        raw, round_id, registration.shift, registration.canvas_offset,
        bspline_transform=registration.bspline if round_id == "R2" else None,
        consensus_mask=consensus_mask, remove_extranuclear=remove_extranuclear,
        output_path=output_path, layer_name=layer_name, puncta_ids=ids, sources=sources)
    return data, sources_of(data, ids, sources) if len(data) else np.empty(0, dtype=object)


def sources_of(puncta_data, ids, sources):
    """The ``Source`` of each row of ``puncta_data`` (column 6 is ``puncta_id``),
    looked up in ``sources`` by the matching entry of ``ids``. Use it after
    ``transform_puncta_to_aligned_space``, which may drop rows."""
    by_id = dict(zip(np.asarray(ids, dtype=float), sources))
    return np.array([by_id.get(float(pid), constants.PUNCTA_SOURCE_UNKNOWN)
                     for pid in np.asarray(puncta_data)[:, 6]], dtype=object)


def brightest_along_ray(image, start, direction, scale, refine_um=None):
    """The voxel the Fishing Hook places a punctum on: the brightest voxel of
    ``image`` (Z, Y, X) on the line through ``start`` (data coordinates, the
    cursor) along ``direction`` (world, the camera's view direction), then,
    with ``refine_um``, the brightest voxel within that radius of it. Returns
    data coordinates, or None when the line misses the image.

    The line is sampled every half voxel across the whole image: at a step of
    a voxel or more, rounding to planes skips some, and in a 2D view it skips
    the plane under the cursor, so a punctum went to the plane above or below
    the spot."""
    image = np.asarray(image)
    scale = np.asarray(scale, dtype=float)
    start = np.asarray(start, dtype=float)
    step = np.asarray(direction, dtype=float) / scale            # world direction -> voxels
    if not np.isfinite(step).all() or not np.linalg.norm(step):
        return None
    step = step / np.linalg.norm(step)
    reach = float(np.linalg.norm(image.shape)) + 1
    points = start + np.outer(np.arange(-reach, reach, 0.5), step)
    voxels = np.round(points).astype(int)
    inside = np.all((voxels >= 0) & (voxels < np.asarray(image.shape)), axis=1)
    if not inside.any():
        return None
    points, voxels = points[inside], voxels[inside]
    values = np.nan_to_num(image[tuple(voxels.T)].astype(float))
    target = voxels[int(np.argmax(values))]
    if refine_um:
        radius = [int(max(1, refine_um / v)) for v in scale]
        lo = np.maximum(target - radius, 0)
        local = image[tuple(slice(l, t + r + 1) for l, t, r in zip(lo, target, radius))]
        if local.size:
            target = lo + np.array(np.unravel_index(int(np.argmax(local)), local.shape))
    return target.astype(float)


def far_from(coords, kept, scale):
    """Which of ``coords`` are further than ``PUNCTA_DUPLICATE_DISTANCE_UM``
    from every punctum in ``kept``: new detections, not duplicates of a kept
    punctum. Both are (N, 3) voxel coordinates; ``scale`` is the voxel size in
    µm, so the distance is the same in Z as in XY."""
    from scipy.spatial import cKDTree
    coords = np.asarray(coords, dtype=float).reshape(-1, 3)
    kept = np.asarray(kept, dtype=float).reshape(-1, 3)
    if len(kept) == 0 or len(coords) == 0:
        return np.ones(len(coords), dtype=bool)
    scale = np.asarray(scale, dtype=float)
    dists, _ = cKDTree(kept * scale).query(coords * scale)
    return dists > constants.PUNCTA_DUPLICATE_DISTANCE_UM


def channel_key(layer_name):
    """``'R2 - FITC'`` for any image or puncta layer of round R2's FITC channel
    (``'R2 - FITC'``, ``'Warped R2 - FITC_puncta'``, ``'Aligned R2 - FITC'``),
    or None if the name does not name a round and channel."""
    import re
    name = str(layer_name)
    if name.endswith(constants.PUNCTA_SUFFIX):
        name = name[:-len(constants.PUNCTA_SUFFIX)]
    match = re.fullmatch(rf"(?:(?:{constants.ALIGNED_PREFIX}|{constants.WARPED_PREFIX})\s+)?"
                         r"(R\d+)\s+-\s+(.+?)\s*", name)
    return f"{match.group(1)} - {match.group(2)}" if match else None


def _channel_ids_on_disk(key):
    """Every ``puncta_id`` in the session's puncta CSVs of channel ``key``,
    raw and aligned: the two share ids, so a new id must be new to both."""
    import pandas as pd
    from . import session
    ids = []
    for name, info in (session.get_data("processed_files", default={}) or {}).items():
        if channel_key(name) != key or not isinstance(info, dict) or info.get("subtype") != "puncta_csv":
            continue
        try:
            column = pd.read_csv(info["path"], usecols=["puncta_id"])["puncta_id"]
        except (OSError, KeyError, ValueError):
            continue
        ids.extend(pd.to_numeric(column, errors="coerce").dropna().tolist())
    return ids


def allocate_puncta_ids(layer_name, count, existing_ids=()):
    """``count`` new ``puncta_id`` values for the channel of ``layer_name``,
    higher than any id the channel has had, so an id never names two puncta.

    The next free id per channel (``channel_key``) is kept in the session's
    ``puncta_next_id``, so ids deleted or replaced earlier, in this run or a
    previous one, are not handed out again. Sessions without it start above
    ``existing_ids`` and the ids in the channel's CSVs. A layer whose name
    names no round and channel gets ids above ``existing_ids`` only."""
    import pandas as pd
    from . import session
    known = pd.to_numeric(pd.Series(list(existing_ids), dtype=object), errors="coerce").dropna()
    start = int(known.max()) + 1 if len(known) else 0
    key = channel_key(layer_name)
    if key is not None:
        # Read, advance and store the counter as one step: a detection thread
        # and a point added by hand must not be handed the same ids.
        with session.current().lock:
            counters = dict(session.get_data("puncta_next_id", default={}) or {})
            if key in counters:
                start = max(start, int(counters[key]))
            else:
                on_disk = _channel_ids_on_disk(key)
                start = max([start] + [int(i) + 1 for i in on_disk])
            counters[key] = start + count
            session.update_data("puncta_next_id", counters)
    return np.arange(start, start + count, dtype=np.int64)


def record_detection_settings(layer_name, settings):
    """Store the settings that made the puncta of ``layer_name``'s channel in
    the session's ``puncta_params`` under ``channel_key`` (``'R2 - FITC'``),
    the key batch uses, and drop older entries for the same channel, which GUI
    detection used to key by image layer (``'Warped R2 - FITC'``). One channel
    then has one row in the report's Parameters sheet."""
    from . import session
    from . import decisions
    key = channel_key(layer_name) or str(layer_name)
    stored = session.get_data("puncta_params", default={}) or {}
    old = next((dict(v) for k, v in reversed(list(stored.items())) if (channel_key(k) or k) == key), None)
    params = {k: v for k, v in stored.items() if (channel_key(k) or k) != key}
    params[key] = dict(settings)
    session.update_data("puncta_params", params)
    # Who chose these settings, beside what they produced.
    outcome = ("num_puncta", "detected_in")
    round_id, _, channel = key.partition(" - ")
    decisions.record("detection", {k: v for k, v in settings.items() if k not in outcome},
                     old={k: v for k, v in old.items() if k not in outcome} if old else None,
                     scope={"round": round_id, "channel": channel} if channel else {"layer": key},
                     metrics_after={k: settings[k] for k in outcome if k in settings})


def detection_settings_row(params, n_puncta, detected_in):
    """The ``puncta_params`` entry for a detection with ``process_puncta_detection``'s
    ``params``: the settings the report's Parameters sheet lists. A setting
    missing from ``params`` is given the value detection used for it."""
    given_z_scale = 'z_scale' in params
    params = fill_detection_defaults(dict(params))
    row = {
        'algorithm': params['method'],
        'sensitivity': params['threshold_rel'],
        'min_distance': params['min_distance'],
        'sigma': params['sigma'],
    }
    if given_z_scale:
        row['z_scale'] = params['z_scale']
    row.update({
        'nuclei_only': params['nuclei_only'],
        'tophat': params['use_tophat'],
        'tophat_radius': params['tophat_radius'],
        **({'z_edge_exclude': int(params['z_edge_exclude'])} if params.get('z_edge_exclude') else {}),
        'num_puncta': int(n_puncta),
        'detected_in': detected_in,
    })
    return row


def detection_settings():
    """The session's ``puncta_params`` keyed by ``channel_key``. For sessions
    that hold several entries for one channel (GUI detection on an aligned
    image before ``record_detection_settings``), the one stored last wins: it
    describes the detection the channel's current puncta came from."""
    from . import session
    out = {}
    for name, settings in (session.get_data("puncta_params", default={}) or {}).items():
        key = channel_key(name) or name
        out.pop(key, None)                  # keep the order in which entries were stored
        out[key] = settings
    return out


@timed_stage("puncta_detection",
             fields=lambda a: {"layer": a["layer_name"], "method": (a["params"] or {}).get("method")},
             result_fields=n_rows)
def process_puncta_detection(image_data, mask_data=None, voxels=None, params=None, output_path=None, progress_callback=None, layer_name=None):
    """Orchestrates detection, quality mapping, and session persistence with CSV tags."""
    params = params if params is not None else {}
    # In place: callers record these parameters after detecting, and must
    # record the values used, defaults included.
    fill_detection_defaults(params, voxels)

    method = params['method']
    if progress_callback: progress_callback(-1, f"Detecting spots ({method})...")
    coords = detect_spots_3d(image_data, progress_callback=progress_callback,
                             **{k: v for k, v in params.items() if k != 'method'}, method=method)
    if progress_callback: progress_callback(45, f"Detection complete. Found {len(coords)} candidates.")
    if len(coords) == 0:
        if progress_callback: progress_callback(100, "No spots found.")
        return np.empty((0, 7))

    # Assign a stable puncta_id once, at detection, BEFORE any extranuclear
    # filtering. Surviving puncta keep this identity through filtering, the
    # world-space transform, and reloads (the id is persisted in every CSV).
    puncta_ids = np.arange(len(coords), dtype=float)

    # Spots in the first or last planes are cut by the stack's ends.
    edge = int(params.get('z_edge_exclude') or 0)
    if edge > 0:
        keep = (coords[:, 0] >= edge) & (coords[:, 0] <= image_data.shape[0] - 1 - edge)
        coords, puncta_ids = coords[keep], puncta_ids[keep]
        if len(coords) == 0:
            if progress_callback: progress_callback(100, "No spots away from the Z edges.")
            return np.empty((0, 7))

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
    nuclei_only = params['nuclei_only']
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

# --- Choosing the sensitivity: count puncta over a range of thresholds ----------

def _keep_count(coords, image_shape, mask, params):
    """How many of ``coords`` detection would keep: inside nuclei (with
    ``nuclei_only`` and a ``mask``) and away from the Z edges."""
    if len(coords) == 0:
        return 0
    keep = np.ones(len(coords), dtype=bool)
    edge = int(params.get("z_edge_exclude") or 0)
    if edge > 0:
        keep &= (coords[:, 0] >= edge) & (coords[:, 0] <= image_shape[0] - 1 - edge)
    if params.get("nuclei_only", True) and mask is not None:
        idx = np.clip(coords.astype(int), 0, np.asarray(image_shape) - 1)
        keep &= mask[tuple(idx.T)] > 0
    return int(keep.sum())


def threshold_sweep(image, mask=None, voxels=None, params=None, thresholds=None, progress_callback=None):
    """Puncta detection would keep at each sensitivity of ``thresholds``
    (default: 25 steps from 0.02 to 0.5), with the other settings of
    ``params``: ``(thresholds, counts)``. For Local Maxima one detection at the
    lowest threshold serves every step (the peaks at a higher threshold are
    exactly those of the lowest above it); other methods detect once per step."""
    from skimage.feature import peak_local_max
    params = fill_detection_defaults(dict(params or {}), voxels)
    thresholds = np.asarray(thresholds if thresholds is not None else np.linspace(0.02, 0.5, 25), dtype=float)
    counts = []
    if params["method"] == "Local Maxima":
        data = image
        if params["use_tophat"]:
            data = preprocess_white_tophat(data, radius=params["tophat_radius"])
        if params["sigma"] > 0:
            from skimage.filters import gaussian
            data = gaussian(data, sigma=params["sigma"], preserve_range=True)
        peaks = peak_local_max(data, min_distance=params["min_distance"], threshold_rel=float(thresholds.min()),
                               exclude_border=False)
        values = data[tuple(peaks.T)] if len(peaks) else np.empty(0)
        top = float(data.max())
        for t in thresholds:
            counts.append(_keep_count(peaks[values > t * top], image.shape, mask, params))
    else:
        for i, t in enumerate(thresholds):
            if progress_callback:
                progress_callback(int(100 * i / len(thresholds)), f"Sensitivity {t:.3f}...")
            coords = detect_spots_3d(image, **dict(params, threshold_rel=float(t)))
            counts.append(_keep_count(np.asarray(coords).reshape(-1, 3), image.shape, mask, params))
    return thresholds, np.asarray(counts, dtype=int)


def find_plateau(thresholds, counts, tolerance=0.1):
    """The widest run of consecutive thresholds over which the count changes by
    at most ``tolerance`` (a fraction) per step: ``{"start", "end", "width",
    "suggested", "count"}``, ``suggested`` its middle threshold; None when no
    step is that stable or every count is 0."""
    thresholds, counts = np.asarray(thresholds, float), np.asarray(counts, float)
    stable = []
    for a, b in zip(counts[:-1], counts[1:]):
        stable.append(max(a, b) > 0 and abs(a - b) <= tolerance * max(a, b))
    best = None
    i = 0
    while i < len(stable):
        if not stable[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(stable) and stable[j + 1]:
            j += 1
        width = thresholds[j + 1] - thresholds[i]
        if best is None or width > best[2]:
            best = (i, j + 1, width)
        i = j + 1
    if best is None:
        return None
    lo, hi, width = best
    middle = (lo + hi) // 2
    return {"start": float(thresholds[lo]), "end": float(thresholds[hi]), "width": float(width),
            "suggested": float(thresholds[middle]), "count": int(counts[middle])}
