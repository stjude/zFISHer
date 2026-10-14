import numpy as np
import tifffile
from pathlib import Path
import gc
import logging

import pandas as pd

from . import io, registration, segmentation, puncta, analysis, session, stages
from .. import constants
from .timing import timed_stage

logger = logging.getLogger(__name__)


@timed_stage("pipeline_total",
             fields=lambda a: {"dataset": Path(a["output_dir"]).name, "seg_method": a["seg_method"],
                               "apply_warp": a["apply_warp"]})
def run_full_zfisher_pipeline(
    r1_path, r2_path, output_dir,
    seg_method="Classical", merge_splits=True,
    r1_nuclear_channel=None, r2_nuclear_channel=None,
    puncta_config=None, pairwise_rules=None, tri_rules=None,
    apply_warp=True, max_ransac_distance=0,
    overlap_method="Intersection", match_threshold=0,
    remove_extranuclear_puncta=True,
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
    r1_nuclear_channel : str, optional
        Name of the nuclear stain channel in R1 (e.g. "DAPI").  If None,
        auto-detected from channel names or falls back to the first channel.
    r2_nuclear_channel : str, optional
        Name of the nuclear stain channel in R2.  If None, uses the same
        resolution as R1.
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
    try:
        _update(5, "Loading Round 1...")
        r1_sess = io.load_image_session(r1_path)
        _update(8, "Loading Round 2...")
        r2_sess = io.load_image_session(r2_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load image files. Check that the files exist and are valid "
            f"ND2 or TIFF images.\n  R1: {r1_path}\n  R2: {r2_path}\n  Error: {exc}"
        ) from exc

    input_dir = output_dir / constants.INPUT_DIR
    try:
        if str(r1_path).lower().endswith('.nd2'):
            _update(10, "Converting R1 to OME-TIF...")
            io.convert_nd2_to_ome(r1_sess, input_dir, "R1")
        if str(r2_path).lower().endswith('.nd2'):
            _update(12, "Converting R2 to OME-TIF...")
            io.convert_nd2_to_ome(r2_sess, input_dir, "R2")
    except Exception as exc:
        raise RuntimeError(f"Failed to convert ND2 to OME-TIFF: {exc}") from exc

    # --- 2b. Resolve nuclear channels (may differ between rounds) ---
    if r1_nuclear_channel:
        r1_nuc_ch = r1_nuclear_channel
    else:
        r1_nuc_ch = io.find_nuclear_channel(r1_sess.channels)
        if r1_nuc_ch is None:
            r1_nuc_ch = r1_sess.channels[0]
            logger.warning("No known nuclear stain found in R1. Falling back to first channel: %s", r1_nuc_ch)

    if r2_nuclear_channel:
        r2_nuc_ch = r2_nuclear_channel
    else:
        r2_nuc_ch = io.find_nuclear_channel(r2_sess.channels)
        if r2_nuc_ch is None:
            r2_nuc_ch = r2_sess.channels[0]
            logger.warning("No known nuclear stain found in R2. Falling back to first channel: %s", r2_nuc_ch)

    # Store the R1 nuclear channel as the session's primary nuclear channel
    # (used downstream by layer naming, report generation, etc.)
    nuc_ch = r1_nuc_ch
    session.update_data("nuclear_channel", nuc_ch)

    # --- 3. Nuclear Segmentation ---
    r1_dapi = io.get_channel_data(r1_sess, r1_nuc_ch)
    r2_dapi = io.get_channel_data(r2_sess, r2_nuc_ch)

    # Map batch config method names to process_session_dapi convention
    method_key = "cellpose" if seg_method.lower() == "cellpose" else "classical"

    try:
        seg_results = segmentation.process_session_dapi(
            r1_data=r1_dapi, r2_data=r2_dapi, output_dir=output_dir,
            progress_callback=lambda p, t: _update(15 + int(p * 0.10), t),
            method=method_key, merge_splits=merge_splits,
            voxel_spacing=r1_sess.voxels,
            nuclear_channels={"R1": r1_nuc_ch, "R2": r2_nuc_ch},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Nuclear segmentation failed: {exc}\n"
            f"Method: {method_key}, Merge splits: {merge_splits}"
        ) from exc

    # --- 4. Puncta Detection on Raw Images (per-round masks) ---
    seg_dir = output_dir / constants.SEGMENTATION_DIR

    # Determine which channels to detect puncta on and their parameters
    if puncta_config:
        # Use channels and params from batch config
        puncta_channels = list(puncta_config.keys())
    else:
        # Default: all non-nuclear channels with default params
        nuc_names_upper = {r1_nuc_ch.upper(), r2_nuc_ch.upper()}
        puncta_channels = [
            ch for ch in r1_sess.channels
            if ch.upper() not in nuc_names_upper
        ]

    # Load per-round nuclear masks for puncta filtering, as registered by
    # segmentation, once per round rather than once per channel.
    r1_mask_path = stages.round_mask_path("R1")
    r2_mask_path = stages.round_mask_path("R2")
    r1_mask = io.read_label_tif(r1_mask_path) if r1_mask_path else None
    r2_mask = io.read_label_tif(r2_mask_path) if r2_mask_path else None

    detected = []  # (round, channel) of every channel whose raw-space CSV was written
    channel_jobs = [(rnd, mask) for rnd, mask in [("R1", r1_mask), ("R2", r2_mask)] for _ in puncta_channels]
    job_count = max(len(channel_jobs), 1)
    job_i = 0

    for rnd, rnd_mask, sess_obj, raw_path in [("R1", r1_mask, r1_sess, r1_path), ("R2", r2_mask, r2_sess, r2_path)]:
        for ch in puncta_channels:
            # Build params for this channel
            if puncta_config and ch in puncta_config:
                ch_params = dict(puncta_config[ch])
                # Skip this channel if it's restricted to the other round
                ch_round = ch_params.pop("round", "Both")
                if ch_round != "Both" and ch_round != rnd:
                    job_i += 1
                    continue
                nuclei_only = ch_params.pop("nuclei_only", True)
            else:
                ch_params = {
                    'threshold_rel': constants.PUNCTA_THRESHOLD_REL,
                    'min_distance': constants.PUNCTA_MIN_DISTANCE,
                    'method': "Local Maxima",
                }
                nuclei_only = True

            ch_idx = list(sess_obj.channels).index(ch)
            job_base = 25 + int((job_i / job_count) * 10)
            _update(job_base, f"Detecting puncta (raw): {rnd} {ch}...")

            # The same step the GUI re-detects a channel with. It keeps the
            # raw-space puncta, records them, and stores the channel's
            # settings for the report; the aligned rows share their puncta_id.
            try:
                raw_csv = stages.detect_channel(
                    rnd, ch, params=dict(ch_params, nuclei_only=nuclei_only), run_by="batch",
                    image=sess_obj.data[:, ch_idx, :, :], image_path=raw_path, voxels=sess_obj.voxels,
                    mask=rnd_mask,
                    progress_callback=lambda p, t, _b=job_base: _update(_b + int(p / 100 * 2), f"{rnd} {ch}: {t}"),
                )
            except Exception as exc:
                logger.error("Puncta detection failed for %s %s: %s", rnd, ch, exc, exc_info=True)
                _update(job_base, f"Puncta detection failed for {rnd} {ch}: {exc}")
                job_i += 1
                continue
            if raw_csv is not None:
                detected.append((rnd, ch))

            job_i += 1

    # --- 5. Registration (RANSAC) ---
    shift, _ = registration.calculate_session_registration(
        seg_results['R1'][1], seg_results['R2'][1],
        voxels=r1_sess.voxels,
        max_distance=max_ransac_distance,
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

    for prefix, sess_obj, layer_list, round_nuc in [
        ("R1", r1_sess, r1_layers, r1_nuc_ch),
        ("R2", r2_sess, r2_layers, r2_nuc_ch),
    ]:
        mask_name = f"{prefix} - {round_nuc}{constants.MASKS_SUFFIX}"
        mask_path = io.find_mask(seg_dir, mask_name)
        if mask_path:
            layer_list.append({
                'name': mask_name,
                'data': io.read_label_tif(mask_path),
                'scale': sess_obj.voxels,
                'is_label': True
            })

    aligned_dir = output_dir / constants.ALIGNED_DIR
    try:
        _, bspline_transform, canvas_offset = registration.generate_global_canvas(
            r1_layers_data=r1_layers, r2_layers_data=r2_layers,
            shift=shift, output_dir=aligned_dir, apply_warp=apply_warp,
            progress_callback=lambda p, t: _update(45 + int(p * 0.15), t),
            r2_channel_names={r1_nuc_ch: r2_nuc_ch},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Canvas generation (alignment + warping) failed: {exc}"
        ) from exc
    session.update_data("canvas_scale", r1_sess.voxels)

    # --- 7. Consensus Nuclei ---
    r1_aligned_mask = aligned_dir / f"Aligned_R1_{r1_nuc_ch}{constants.MASKS_SUFFIX}.tif"
    r2_warped_mask = aligned_dir / f"Warped_R2_{r2_nuc_ch}{constants.MASKS_SUFFIX}.tif"
    if not r2_warped_mask.exists():
        r2_warped_mask = aligned_dir / f"Aligned_R2_{r2_nuc_ch}{constants.MASKS_SUFFIX}.tif"

    if not (r1_aligned_mask.exists() and r2_warped_mask.exists()):
        raise RuntimeError(
            "Consensus failed: aligned masks not found. "
            "Canvas generation may have failed."
        )

    try:
        merged_mask, _ = segmentation.process_consensus_nuclei(
            mask1=tifffile.imread(r1_aligned_mask),
            mask2=tifffile.imread(r2_warped_mask),
            output_dir=output_dir,
            threshold=match_threshold or None,
            method=overlap_method,
            voxel_size=r1_sess.voxels,
            progress_callback=lambda p, t: _update(60 + int(p * 0.1), t)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Consensus nuclei matching failed: {exc}"
        ) from exc

    # --- 8. Transform Puncta to Aligned Space ---
    reports_dir = output_dir / constants.REPORTS_DIR
    reports_dir.mkdir(exist_ok=True, parents=True)

    aligned_puncta_layers = []  # Collect for colocalization analysis
    job_count = max(len(detected), 1)
    job_i = 0
    for rnd, ch in detected:
        prefix_str = constants.ALIGNED_PREFIX if (rnd == "R1" or bspline_transform is None) else constants.WARPED_PREFIX
        job_base = 70 + int((job_i / job_count) * 15)
        _update(job_base, f"Transforming puncta: {rnd} {ch}...")

        # The same step the GUI rewarps a re-detected channel with: the raw
        # CSV through the registration saved at canvas generation.
        csv_out = stages.transform_channel(rnd, ch, remove_extranuclear=remove_extranuclear_puncta,
                                           run_by="batch", consensus_mask=merged_mask)

        # Build layer data dict for colocalization analysis
        transformed = pd.read_csv(csv_out, float_precision="round_trip") if csv_out is not None else None
        if transformed is not None and len(transformed) > 0:
            aligned_puncta_layers.append({
                'name': f"{prefix_str} {rnd} - {ch}{constants.PUNCTA_SUFFIX}",
                'data': transformed[['Z', 'Y', 'X']].to_numpy(float),
                'scale': np.array(r1_sess.voxels),
                'translate': np.zeros(3),
                'nucleus_ids': transformed['Nucleus_ID'].to_numpy(float),
                'puncta_id': transformed['puncta_id'].to_numpy(float),
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

        # Count total nuclei for stats: distinct nonzero labels in the consensus
        # mask. `.max()` would return the highest label ID, which overcounts when
        # labels are non-contiguous (gaps from remove_small_objects or mask edits).
        if merged_mask is not None:
            unique_ids = np.unique(merged_mask)
            nucleus_ids = unique_ids[unique_ids > 0]
            total_nuclei = int(len(nucleus_ids))
        else:
            nucleus_ids = None
            total_nuclei = None

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
                nucleus_ids=nucleus_ids,
            )
            _update(95, "Colocalization report saved.")
        except Exception as exc:
            logger.error("Colocalization analysis failed: %s", exc, exc_info=True)
            _update(95, f"Colocalization failed: {exc}")

    # --- Cleanup ---
    del r1_sess, r2_sess, r1_dapi, r2_dapi, merged_mask
    gc.collect()

    # Ensure the session JSON is fully written before exiting
    session.save_session()

    _update(100, "Done.")
    logger.info("Pipeline complete for %s", output_dir)
