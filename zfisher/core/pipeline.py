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
    Runs the complete zFISHer automated pipeline (steps 1-7) without viewer
    interaction.  Mirrors the logic in NewSessionWidget._on_autorun but saves
    results to disk only — no napari layers are created.

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
        progress_callback=lambda p, t: _update(15 + int(p * 0.2), t)
    )

    # --- 4. Registration (RANSAC) ---
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

    # --- 5. Canvas Generation (Rigid + Deformable Warping) ---
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

    seg_dir = output_dir / constants.SEGMENTATION_DIR
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
    registration.generate_global_canvas(
        r1_layers_data=r1_layers, r2_layers_data=r2_layers,
        shift=shift, output_dir=aligned_dir, apply_warp=True,
        progress_callback=lambda p, t: _update(45 + int(p * 0.15), t)
    )
    session.update_data("canvas_scale", r1_sess.voxels)

    # --- 6. Consensus Nuclei ---
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

    # --- 7. Puncta Detection ---
    puncta_channels = [
        ch for ch in r1_sess.channels
        if ch.upper() != constants.DAPI_CHANNEL_NAME.upper()
    ]
    puncta_params = {
        'threshold_rel': constants.PUNCTA_THRESHOLD_REL,
        'min_distance': constants.PUNCTA_MIN_DISTANCE,
        'method': "Local Maxima"
    }
    channel_jobs = [
        (rnd, pfx)
        for rnd, pfx in [("R1", "Aligned"), ("R2", "Warped")]
        for _ in puncta_channels
    ]
    job_count = max(len(channel_jobs), 1)
    job_i = 0
    for rnd, prefix_str in [("R1", "Aligned"), ("R2", "Warped")]:
        for ch in puncta_channels:
            ch_path = aligned_dir / f"{prefix_str}_{rnd}_{ch}.tif"
            if not ch_path.exists():
                ch_path = aligned_dir / f"Aligned_{rnd}_{ch}.tif"
            if ch_path.exists():
                _update(
                    70 + int((job_i / job_count) * 25),
                    f"Detecting puncta: {rnd} {ch}..."
                )
                csv_out = seg_dir / f"{prefix_str}_{rnd}_{ch}{constants.PUNCTA_SUFFIX}.csv"
                puncta.process_puncta_detection(
                    image_data=tifffile.imread(ch_path),
                    mask_data=merged_mask,
                    params=puncta_params,
                    output_path=csv_out
                )
            job_i += 1

    # --- Cleanup ---
    del r1_sess, r2_sess, r1_dapi, r2_dapi, merged_mask
    gc.collect()

    _update(100, "Done.")
    logger.info("Pipeline complete for %s", output_dir)
