import numpy as np
import tifffile
from pathlib import Path
import gc
import logging

from . import io, registration, segmentation, puncta, analysis, session
from .. import constants

logger = logging.getLogger(__name__)


def run_full_zfisher_pipeline(
    r1_path, r2_path, output_dir,
    seg_method="Classical", merge_splits=True,
    puncta_config=None, pairwise_rules=None, tri_rules=None,
    progress_callback=None,
):
    """
    Runs the complete zFISHer automated pipeline without viewer interaction.

    Pipeline order:
      1. Session init
      2. Load & convert raw images
      3. Nuclei segmentation (per-round masks + centroids)
      4. Puncta detection on raw images (masked by per-round nuclei)
      5. Registration (RANSAC on centroids)
      6. Canvas generation (rigid align + B-spline warp of images)
      7. Consensus nuclei (merge aligned masks)
      8. Transform puncta coordinates into aligned space + reassign nucleus IDs
      9. Colocalization analysis (pairwise + tri)

    Parameters
    ----------
    r1_path : Path
        Path to the Round 1 image file (.nd2 / .tif / .ome.tif).
    r2_path : Path
        Path to the Round 2 image file.
    output_dir : Path
        Root output directory for this session.
    seg_method : str
        "Classical" or "Cellpose".
    merge_splits : bool
        Whether to merge over-segmented nuclei.
    puncta_config : dict, optional
        {channel_name: params_dict} for per-channel puncta detection.
        params_dict keys: method, threshold_rel, min_distance, sigma,
        nuclei_only, use_tophat, tophat_radius.
        If None, all non-nuclear channels are detected with defaults.
    pairwise_rules : list[dict], optional
        Pairwise colocalization rules. Each dict has 'source', 'target', 'threshold'.
    tri_rules : list[dict], optional
        Tri-colocalization rules. Each dict has 'source'(anchor), 'target'(channel_a),
        'channel_b', 'threshold'.
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

    # --- 2b. Resolve nuclear channel (headless: auto-detect or fallback to index 0) ---
    nuc_ch = io.find_nuclear_channel(r1_sess.channels)
    if nuc_ch is None:
        nuc_ch = r1_sess.channels[0]
        logger.warning("No known nuclear stain found. Falling back to first channel: %s", nuc_ch)
    session.update_data("nuclear_channel", nuc_ch)

    # --- 3. Nuclear Segmentation ---
    r1_dapi = io.get_channel_data(r1_sess, nuc_ch)
    r2_dapi = io.get_channel_data(r2_sess, nuc_ch)

    # Map batch config method names to process_session_dapi convention
    method_key = "cellpose" if seg_method.lower() == "cellpose" else "classical"

    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi, r2_data=r2_dapi, output_dir=output_dir,
        progress_callback=lambda p, t: _update(15 + int(p * 0.10), t),
        method=method_key, merge_splits=merge_splits,
        voxel_spacing=r1_sess.voxels,
    )

    # --- 4. Puncta Detection on Raw Images (per-round masks) ---
    seg_dir = output_dir / constants.SEGMENTATION_DIR

    # Determine which channels to detect puncta on and their parameters
    if puncta_config:
        # Use channels and params from batch config
        puncta_channels = list(puncta_config.keys())
    else:
        # Default: all non-nuclear channels with default params
        puncta_channels = [
            ch for ch in r1_sess.channels
            if ch.upper() != nuc_ch.upper()
        ]

    # Load per-round nuclear masks for puncta filtering
    r1_mask_path = seg_dir / f"R1 - {nuc_ch}{constants.MASKS_SUFFIX}.tif"
    r2_mask_path = seg_dir / f"R2 - {nuc_ch}{constants.MASKS_SUFFIX}.tif"
    r1_mask = tifffile.imread(r1_mask_path) if r1_mask_path.exists() else None
    r2_mask = tifffile.imread(r2_mask_path) if r2_mask_path.exists() else None

    raw_puncta_results = {}  # {("R1", ch): ndarray, ("R2", ch): ndarray}
    channel_jobs = [(rnd, mask) for rnd, mask in [("R1", r1_mask), ("R2", r2_mask)] for _ in puncta_channels]
    job_count = max(len(channel_jobs), 1)
    job_i = 0

    for rnd, rnd_mask, sess_obj in [("R1", r1_mask, r1_sess), ("R2", r2_mask, r2_sess)]:
        for ch in puncta_channels:
            # Build params for this channel
            if puncta_config and ch in puncta_config:
                ch_params = dict(puncta_config[ch])
                nuclei_only = ch_params.pop("nuclei_only", True)
            else:
                ch_params = {
                    'threshold_rel': constants.PUNCTA_THRESHOLD_REL,
                    'min_distance': constants.PUNCTA_MIN_DISTANCE,
                    'method': "Local Maxima",
                }
                nuclei_only = True

            # Only pass nuclear mask if nuclei_only is True
            mask_for_detection = rnd_mask if nuclei_only else None

            ch_idx = list(sess_obj.channels).index(ch)
            ch_data = sess_obj.data[:, ch_idx, :, :]
            job_base = 25 + int((job_i / job_count) * 10)
            _update(job_base, f"Detecting puncta (raw): {rnd} {ch}...")

            result = puncta.process_puncta_detection(
                image_data=ch_data,
                mask_data=mask_for_detection,
                voxels=sess_obj.voxels,
                params=ch_params,
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
            "Check that both rounds have sufficient nuclear signal."
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
        mask_name = f"{prefix} - {nuc_ch}{constants.MASKS_SUFFIX}"
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
    r1_aligned_mask = aligned_dir / f"Aligned_R1_{nuc_ch}{constants.MASKS_SUFFIX}.tif"
    r2_warped_mask = aligned_dir / f"Warped_R2_{nuc_ch}{constants.MASKS_SUFFIX}.tif"
    if not r2_warped_mask.exists():
        r2_warped_mask = aligned_dir / f"Aligned_R2_{nuc_ch}{constants.MASKS_SUFFIX}.tif"

    if not (r1_aligned_mask.exists() and r2_warped_mask.exists()):
        raise RuntimeError(
            "Consensus failed: aligned masks not found. "
            "Canvas generation may have failed."
        )

    merged_mask, _ = segmentation.process_consensus_nuclei(
        mask1=tifffile.imread(r1_aligned_mask),
        mask2=tifffile.imread(r2_warped_mask),
        output_dir=output_dir, threshold=0, method="Intersection",
        progress_callback=lambda p, t: _update(60 + int(p * 0.1), t)
    )

    # --- 8. Transform Puncta to Aligned Space ---
    reports_dir = output_dir / constants.REPORTS_DIR
    reports_dir.mkdir(exist_ok=True, parents=True)

    aligned_puncta_layers = []  # Collect for colocalization analysis
    job_count = max(len(raw_puncta_results), 1)
    job_i = 0
    for (rnd, ch), raw_data in raw_puncta_results.items():
        prefix_str = constants.ALIGNED_PREFIX if rnd == "R1" else constants.WARPED_PREFIX
        job_base = 70 + int((job_i / job_count) * 15)
        job_span = max(int(15 / job_count), 1)
        _update(job_base, f"Transforming puncta: {rnd} {ch}...")

        puncta_layer_name = f"{prefix_str} {rnd} - {ch}{constants.PUNCTA_SUFFIX}"
        csv_out = reports_dir / f"{puncta_layer_name.replace(' ', '_')}.csv"

        transformed = puncta.transform_puncta_to_aligned_space(
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

        # Build layer data dict for colocalization analysis
        if transformed is not None and len(transformed) > 0:
            coords = transformed[:, :3]  # Z, Y, X
            nuc_ids = transformed[:, 3] if transformed.shape[1] > 3 else None
            aligned_puncta_layers.append({
                'name': puncta_layer_name,
                'data': coords,
                'scale': np.array(r1_sess.voxels),
                'translate': np.zeros(3),
                'nucleus_ids': nuc_ids,
            })

        job_i += 1

    # --- 9. Colocalization Analysis ---
    has_coloc = (pairwise_rules and len(pairwise_rules) > 0) or (tri_rules and len(tri_rules) > 0)
    if has_coloc and aligned_puncta_layers:
        _update(85, "Running colocalization analysis...")

        # Map user-friendly rule names to actual aligned layer names.
        # Rules reference channel names (e.g. "Cy5"); resolve to full layer names.
        layer_names = [l['name'] for l in aligned_puncta_layers]

        def _resolve_layer_name(channel_ref):
            """Find the aligned puncta layer matching a channel reference."""
            for ln in layer_names:
                if channel_ref in ln:
                    return ln
            return channel_ref  # Fallback: use as-is

        resolved_pairwise = []
        if pairwise_rules:
            for rule in pairwise_rules:
                resolved_pairwise.append({
                    'source': _resolve_layer_name(rule['source']),
                    'target': _resolve_layer_name(rule['target']),
                    'threshold': rule['threshold'],
                })

        resolved_tri = []
        if tri_rules:
            for rule in tri_rules:
                resolved_tri.append({
                    'anchor': _resolve_layer_name(rule['source']),
                    'channel_a': _resolve_layer_name(rule['target']),
                    'channel_b': _resolve_layer_name(rule['channel_b']),
                    'threshold': rule['threshold'],
                })

        # Count total nuclei for stats
        total_nuclei = int(merged_mask.max()) if merged_mask is not None else None

        report_filename = f"zFISHer_Report{constants.EXCEL_SUFFIX}"
        try:
            analysis.run_colocalization_analysis(
                layers_data=aligned_puncta_layers,
                rules=resolved_pairwise,
                filename=report_filename,
                r1_path=str(r1_path),
                r2_path=str(r2_path),
                output_dir=str(reports_dir),
                tri_rules=resolved_tri if resolved_tri else None,
                total_nuclei=total_nuclei,
            )
            _update(95, "Colocalization report saved.")
        except Exception as exc:
            logger.error("Colocalization analysis failed: %s", exc, exc_info=True)
            _update(95, f"Colocalization failed: {exc}")

    # --- Cleanup ---
    del r1_sess, r2_sess, r1_dapi, r2_dapi, merged_mask, raw_puncta_results
    gc.collect()

    # Ensure the session JSON is fully written before exiting
    session.save_session()

    _update(100, "Done.")
    logger.info("Pipeline complete for %s", output_dir)
