import numpy as np
import tifffile
from pathlib import Path
import gc
import logging

from . import io, registration, segmentation, puncta, session
from .. import constants

logger = logging.getLogger(__name__)


def run_full_zfisher_pipeline(r1_path, r2_path, output_dir, progress_callback=None):
    """
    Runs the complete zFISHer automated pipeline without viewer interaction.

    Pipeline order:
      1. Session init
      2. Load & convert raw images
      3. DAPI segmentation (per-round masks + centroids)
      4. Puncta detection on raw images (masked by per-round DAPI)
      5. Registration (RANSAC on centroids)
      6. Canvas generation (rigid align + B-spline warp of images)
      7. Consensus nuclei (merge aligned masks)
      8. Transform puncta coordinates into aligned space + reassign nucleus IDs

    Parameters
    ----------
    r1_path : Path
        Path to the Round 1 image file (.nd2 / .tif / .ome.tif).
    r2_path : Path
        Path to the Round 2 image file.
    output_dir : Path
        Root output directory for this session.
    progress_callback : callable, optional
        ``callback(value: int, text: str)`` called with 0-100 progress.

    Raises
    ------
    RuntimeError
        If session initialisation, registration, or consensus fails.
    """
    r1_path = Path(r1_path)
    r2_path = Path(r2_path)
    output_dir = Path(output_dir)

    def _update(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    # --- 1. Session Init ---
    _update(2, "Initializing session...")
    ok = session.initialize_new_session(output_dir, r1_path, r2_path)
    if not ok:
        raise RuntimeError(
            f"Session already exists in {output_dir}. "
            "Remove the existing session or choose a different output directory."
        )

    # --- 2. Load & Convert Raw Images ---
    _update(5, "Loading Round 1...")
    r1_sess = io.load_image_session(r1_path)
    _update(8, "Loading Round 2...")
    r2_sess = io.load_image_session(r2_path)

    input_dir = output_dir / constants.INPUT_DIR
    if str(r1_path).lower().endswith('.nd2'):
        _update(10, "Converting R1 to OME-TIF...")
        io.convert_nd2_to_ome(r1_sess, input_dir, "R1")
    if str(r2_path).lower().endswith('.nd2'):
        _update(12, "Converting R2 to OME-TIF...")
        io.convert_nd2_to_ome(r2_sess, input_dir, "R2")

    # --- 3. DAPI Segmentation ---
    r1_dapi = io.get_channel_data(r1_sess, constants.DAPI_CHANNEL_NAME)
    r2_dapi = io.get_channel_data(r2_sess, constants.DAPI_CHANNEL_NAME)
    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, r2_data=r2_dapi, output_dir=output_dir,
        progress_callback=lambda p, t: _update(15 + int(p * 0.10), t)
    )

    # --- 4. Puncta Detection on Raw Images (per-round masks) ---
    seg_dir = output_dir / constants.SEGMENTATION_DIR
    puncta_channels = [
        ch for ch in r1_sess.channels
        if ch.upper() != constants.DAPI_CHANNEL_NAME.upper()
    ]
    puncta_params = {
        'threshold_rel': constants.PUNCTA_THRESHOLD_REL,
        'min_distance': constants.PUNCTA_MIN_DISTANCE,
        'method': "Local Maxima"
    }

    # Load per-round DAPI masks for puncta filtering
    r1_mask_path = seg_dir / f"R1 - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    r2_mask_path = seg_dir / f"R2 - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    r1_mask = tifffile.imread(r1_mask_path) if r1_mask_path.exists() else None
    r2_mask = tifffile.imread(r2_mask_path) if r2_mask_path.exists() else None

    raw_puncta_results = {}  # {("R1", ch): ndarray, ("R2", ch): ndarray}
    channel_jobs = [(rnd, mask) for rnd, mask in [("R1", r1_mask), ("R2", r2_mask)] for _ in puncta_channels]
    job_count = max(len(channel_jobs), 1)
    job_i = 0

    for rnd, rnd_mask, sess_obj in [("R1", r1_mask, r1_sess), ("R2", r2_mask, r2_sess)]:
        for ch in puncta_channels:
            ch_idx = list(sess_obj.channels).index(ch)
            ch_data = sess_obj.data[:, ch_idx, :, :]
            job_base = 25 + int((job_i / job_count) * 10)
            _update(job_base, f"Detecting puncta (raw): {rnd} {ch}...")

            result = puncta.process_puncta_detection(
                image_data=ch_data,
                mask_data=rnd_mask,
                voxels=sess_obj.voxels,
                params=puncta_params,
                progress_callback=lambda p, t, _b=job_base: _update(
                    _b + int(p / 100 * 2), f"{rnd} {ch}: {t}"
                )
            )
            raw_puncta_results[(rnd, ch)] = result
            job_i += 1

    # --- 5. Registration (RANSAC) ---
    shift, _ = registration.calculate_session_registration(
        seg_results['R1'][1], seg_results['R2'][1],
        voxels=r1_sess.voxels,
        progress_callback=lambda p, t: _update(35 + int(p * 0.1), t)
    )
    if shift is None:
        raise RuntimeError(
            "Registration failed: could not calculate a valid shift. "
            "Check that both rounds have sufficient DAPI signal."
        )

    # --- 6. Canvas Generation (Rigid + Deformable Warping) ---
    r1_layers = [
        {'name': f"R1 - {ch}", 'data': r1_sess.data[:, i, :, :],
         'scale': r1_sess.voxels, 'is_label': False}
        for i, ch in enumerate(r1_sess.channels)
    ]
    r2_layers = [
        {'name': f"R2 - {ch}", 'data': r2_sess.data[:, i, :, :],
         'scale': r2_sess.voxels, 'is_label': False}
        for i, ch in enumerate(r2_sess.channels)
    ]

    for prefix, sess_obj, layer_list in [("R1", r1_sess, r1_layers),
                                          ("R2", r2_sess, r2_layers)]:
        mask_name = f"{prefix} - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}"
        mask_path = seg_dir / f"{mask_name}.tif"
        if mask_path.exists():
            layer_list.append({
                'name': mask_name,
                'data': tifffile.imread(mask_path),
                'scale': sess_obj.voxels,
                'is_label': True
            })

    aligned_dir = output_dir / constants.ALIGNED_DIR
    _, bspline_transform, canvas_offset = registration.generate_global_canvas(
        r1_layers_data=r1_layers, r2_layers_data=r2_layers,
        shift=shift, output_dir=aligned_dir, apply_warp=True,
        progress_callback=lambda p, t: _update(45 + int(p * 0.15), t)
    )
    session.update_data("canvas_scale", r1_sess.voxels)

    # --- 7. Consensus Nuclei ---
    r1_aligned_mask = aligned_dir / f"Aligned_R1_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    r2_warped_mask = aligned_dir / f"Warped_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    if not r2_warped_mask.exists():
        r2_warped_mask = aligned_dir / f"Aligned_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"

    if not (r1_aligned_mask.exists() and r2_warped_mask.exists()):
        raise RuntimeError(
            "Consensus failed: aligned masks not found. "
            "Canvas generation may have failed."
        )

    merged_mask, _ = segmentation.process_consensus_nuclei(
        mask1=tifffile.imread(r1_aligned_mask),
        mask2=tifffile.imread(r2_warped_mask),
        output_dir=output_dir, threshold=20.0, method="Intersection",
        progress_callback=lambda p, t: _update(60 + int(p * 0.1), t)
    )

    # --- 8. Transform Puncta to Aligned Space ---
    job_count = max(len(raw_puncta_results), 1)
    job_i = 0
    for (rnd, ch), raw_data in raw_puncta_results.items():
        prefix_str = constants.ALIGNED_PREFIX if rnd == "R1" else constants.WARPED_PREFIX
        job_base = 70 + int((job_i / job_count) * 25)
        job_span = max(int(25 / job_count), 1)
        _update(job_base, f"Transforming puncta: {rnd} {ch}...")

        csv_out = seg_dir / f"{prefix_str}_{rnd}_{ch}{constants.PUNCTA_SUFFIX}.csv"
        puncta_layer_name = f"{prefix_str} {rnd} - {ch}{constants.PUNCTA_SUFFIX}"

        puncta.transform_puncta_to_aligned_space(
            raw_puncta=raw_data,
            round_id=rnd,
            shift=shift,
            canvas_offset=canvas_offset,
            bspline_transform=bspline_transform if rnd == "R2" else None,
            consensus_mask=merged_mask,
            output_path=csv_out,
            layer_name=puncta_layer_name,
            progress_callback=lambda p, t, _b=job_base, _s=job_span: _update(
                _b + int(p / 100 * _s), f"{rnd} {ch}: {t}"
            )
        )
        job_i += 1

    # --- Cleanup ---
    del r1_sess, r2_sess, r1_dapi, r2_dapi, merged_mask, raw_puncta_results
    gc.collect()

    # Ensure the session JSON is fully written before exiting
    session.save_session()

    _update(100, "Done.")
    logger.info("Pipeline complete for %s", output_dir)
