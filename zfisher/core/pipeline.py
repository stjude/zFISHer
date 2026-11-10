import contextlib
import numpy as np
import tifffile
from pathlib import Path
import gc
import logging

import pandas as pd

from . import io, registration, segmentation, analysis, session, stages, decisions, errors
from .. import constants
from .params import DetectionParams, default
from .timing import timed_stage

logger = logging.getLogger(__name__)


def run_full_zfisher_pipeline(*args, **kwargs):
    """Run the whole pipeline on one R1/R2 pair, on a session of its own.

    Takes the arguments of ``_run_full_zfisher_pipeline``, documented there.
    The run makes and saves its own ``session.Session`` in ``output_dir``; the
    default session, the one a GUI has open, is left as it was, and the
    session log that was attached before the run is attached again after it.
    """
    from .log_config import restoring_session_log
    with session.use(session.Session()), restoring_session_log():
        return _run_full_zfisher_pipeline(*args, **kwargs)


def plan_full_zfisher_pipeline(r1_path, r2_path, output_dir, seg_method="Classical", merge_splits=True,
                               r1_nuclear_channel=None, r2_nuclear_channel=None, puncta_config=None,
                               pairwise_rules=None, tri_rules=None, apply_warp=True, max_ransac_distance=0,
                               overlap_method="Intersection", match_threshold=0, remove_extranuclear_puncta=True,
                               distribution_bins=None, remove_border_nuclei=False, progress_callback=None):
    """The stages a run of the pipeline with these settings would execute in
    ``output_dir``, in order and named as the run lists them (``"segment"``,
    ``"detect R1 - FITC"``, ...), without running any or writing anything.

    A stage runs when its fingerprint (its settings, input files and upstream
    stages) differs from the one recorded, when a file it wrote is gone, or
    when an upstream stage runs and rewrites the files it reads. A detection
    that would fail, or a report whose channels all come out empty, is not
    foreseen. Raises ``LookupError`` when ``output_dir`` holds no pipeline run
    of these images to resume: a run there would do every stage."""
    r1_path, r2_path, output_dir = Path(r1_path), Path(r2_path), Path(output_dir)
    with session.use(session.Session()):
        if not _open_previous_run(output_dir, r1_path, r2_path, read_only=True):
            raise LookupError(f"{output_dir} holds no pipeline run of these images, so a run would do "
                              "every stage.")
        r1_channels, r2_channels = _channels(r1_path), _channels(r2_path)
        r1_nuc = _nuclear_channel(r1_nuclear_channel, r1_channels, "R1")
        r2_nuc = _nuclear_channel(r2_nuclear_channel, r2_channels, "R2")
        planned, changed = [], set()

        def check(stage, key, params, inputs=(), upstream=(), also_fresh=True):
            """Whether ``stage`` would run; notes it, and whether its files would change."""
            fingerprint = stages.stage_fingerprint(stage, key, params, inputs=inputs, upstream=upstream)
            runs = (not also_fresh or not stages.stage_is_fresh(stage, key, fingerprint)
                    or any(stages._stage_id(s, k) in changed for s, k in upstream))
            if runs:
                planned.append(f"{stage} {key}" if key else stage)
                record = stages.stage_record(stage, key)
                # Downstream stages see a rerun through its files; a stage
                # without files (register) changes them only by its fingerprint.
                if record is None or record.get("outputs") or record["fingerprint"] != fingerprint \
                        or any(stages._stage_id(s, k) in changed for s, k in upstream):
                    changed.add(stages._stage_id(stage, key))
            return runs

        check("segment", "", _segment_params(seg_method, merge_splits, r1_nuc, r2_nuc), inputs=(r1_path, r2_path))
        detected = []
        for rnd, ch, raw_path, params, _chosen, upstream in _detection_jobs(
                puncta_config, r1_channels, r1_nuc, r2_nuc, r1_path, r2_path):
            check("detect", f"{rnd} - {ch}", params, inputs=(raw_path,), upstream=upstream)
            detected.append((rnd, ch))
        check("register", "", _register_params(max_ransac_distance), upstream=REGISTER_UPSTREAM,
              also_fresh=session.get_data("shift") is not None)
        check("canvas", "", _canvas_params(apply_warp, r1_nuc, r2_nuc), inputs=(r1_path, r2_path),
              upstream=CANVAS_UPSTREAM)
        check("consensus", "", _consensus_params(overlap_method, match_threshold, remove_border_nuclei),
              upstream=CONSENSUS_UPSTREAM)
        for rnd, ch in detected:
            key = f"{rnd} - {ch}"
            check("transform", key, _transform_params(remove_extranuclear_puncta), upstream=_transform_upstream(key))
        if (pairwise_rules or tri_rules) and detected:
            check("report", "", _report_params(pairwise_rules, tri_rules, distribution_bins),
                  upstream=_report_upstream(detected))
        return planned


@timed_stage("pipeline_total",
             fields=lambda a: {"dataset": Path(a["output_dir"]).name, "seg_method": a["seg_method"],
                               "apply_warp": a["apply_warp"]})
def _run_full_zfisher_pipeline(
    r1_path, r2_path, output_dir,
    seg_method="Classical", merge_splits=True,
    r1_nuclear_channel=None, r2_nuclear_channel=None,
    puncta_config=None, pairwise_rules=None, tri_rules=None,
    apply_warp=True, max_ransac_distance=0,
    overlap_method="Intersection", match_threshold=0,
    remove_extranuclear_puncta=True,
    distribution_bins=None,
    remove_border_nuclei=False,
    progress_callback=None,
    new_session_file=True,
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
    new_session_file : bool
        Resuming a previous run saves the session under the next number,
        as Load Session does (the default); False saves it back into the
        newest session file. Not a setting of the run: not stored.

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

    # --- 1. Session: a new one, or the one a previous run left here -------------
    _update(2, "Initializing session...")
    resumed = _open_previous_run(output_dir, r1_path, r2_path, in_place=not new_session_file)
    if not resumed and not session.initialize_new_session(output_dir, r1_path, r2_path):
        raise errors.SessionError(
            f"Session already exists in {output_dir} and was not made by a pipeline run that can be "
            "resumed. Remove the existing session or choose a different output directory."
        )
    if resumed:
        logger.info("Resuming the pipeline run in %s: stages whose inputs and settings are unchanged "
                    "are skipped.", output_dir)
    session.update_data("pipeline_settings", {
        "seg_method": seg_method, "merge_splits": merge_splits, "r1_nuclear_channel": r1_nuclear_channel,
        "r2_nuclear_channel": r2_nuclear_channel, "puncta_config": puncta_config,
        "pairwise_rules": pairwise_rules, "tri_rules": tri_rules, "apply_warp": apply_warp,
        "max_ransac_distance": max_ransac_distance, "overlap_method": overlap_method,
        "match_threshold": match_threshold, "remove_extranuclear_puncta": remove_extranuclear_puncta,
        "distribution_bins": distribution_bins, "remove_border_nuclei": remove_border_nuclei})
    if distribution_bins:
        session.update_data("distribution_bins", list(distribution_bins))
    ran = []          # the stages this run executed, for the log and the caller

    # Raw images are read only when a stage that runs needs them.
    loaded = {}

    def round_session(rnd):
        if rnd not in loaded:
            path = r1_path if rnd == "R1" else r2_path
            try:
                _update(5 if rnd == "R1" else 8, f"Loading Round {rnd[1]}...")
                loaded[rnd] = io.load_image_session(path)
            except Exception as exc:
                raise errors.InputError(
                    f"Failed to load image files. Check that the files exist and are valid "
                    f"ND2 or TIFF images.\n  R1: {r1_path}\n  R2: {r2_path}\n  Error: {exc}"
                ) from exc
        return loaded[rnd]

    # From here on the session is written at each stage boundary, not after
    # every file a stage registers: one run used to rewrite it some forty times
    # for a single puncta channel.
    with session.current().batch_writes():
        # --- 2. Load the raw images (a new session; a resumed one reads them only if needed) ---
        # A new session's ND2 files were converted to OME-TIFF when it was
        # created (session.initialize_new_session); they used to be converted
        # a second time here.
        if not resumed:
            round_session("R1")
            round_session("R2")

        # --- 2b. Resolve nuclear channels (may differ between rounds) ---
        r1_channels = _channels(r1_path, loaded.get("R1"))
        r2_channels = _channels(r2_path, loaded.get("R2"))
        r1_nuc_ch = _nuclear_channel(r1_nuclear_channel, r1_channels, "R1")
        r2_nuc_ch = _nuclear_channel(r2_nuclear_channel, r2_channels, "R2")

        # Store the R1 nuclear channel as the session's primary nuclear channel
        # (used downstream by layer naming, report generation, etc.)
        nuc_ch = r1_nuc_ch
        session.update_data("nuclear_channel", nuc_ch)

        def r1_voxels():
            """R1's voxel size: the aligned canvas's, which a finished canvas stage stored."""
            stored = session.get_data("canvas_scale")
            return tuple(stored) if stored else tuple(round_session("R1").voxels)

        session.save_session()
        # --- 3. Nuclear Segmentation ---
        seg_params = _segment_params(seg_method, merge_splits, r1_nuc_ch, r2_nuc_ch)
        method_key = seg_params["method"]
        seg_fp = stages.stage_fingerprint("segment", "", seg_params, inputs=(r1_path, r2_path))
        seg_dir = output_dir / constants.SEGMENTATION_DIR
        if stages.stage_is_fresh("segment", "", seg_fp):
            seg_results = {rnd: (None, np.load(seg_dir / f"{rnd} - {ch}{constants.CENTROIDS_SUFFIX}.npy"))
                           for rnd, ch in (("R1", r1_nuc_ch), ("R2", r2_nuc_ch))}
        else:
            r1_dapi = io.get_channel_data(round_session("R1"), r1_nuc_ch)
            r2_dapi = io.get_channel_data(round_session("R2"), r2_nuc_ch)
            try:
                seg_results = segmentation.process_session_dapi(
                    r1_data=r1_dapi, r2_data=r2_dapi, output_dir=output_dir,
                    progress_callback=lambda p, t: _update(15 + int(p * 0.10), t),
                    method=method_key, merge_splits=merge_splits,
                    voxel_spacing=round_session("R1").voxels,
                    nuclear_channels={"R1": r1_nuc_ch, "R2": r2_nuc_ch},
                )
            except Exception as exc:
                raise errors.DetectionError(
                    f"Nuclear segmentation failed: {exc}\n"
                    f"Method: {method_key}, Merge splits: {merge_splits}"
                ) from exc
            del r1_dapi, r2_dapi
            stages.record_stage("segment", "", seg_fp, seg_params, inputs=(r1_path, r2_path), outputs=[
                seg_dir / f"{rnd} - {ch}{suffix}" for rnd, ch in (("R1", r1_nuc_ch), ("R2", r2_nuc_ch))
                for suffix in (constants.MASKS_SUFFIX + io.MASK_EXTENSION, constants.CENTROIDS_SUFFIX + ".npy")])
            ran.append("segment")

        session.save_session()
        # --- 4. Puncta Detection on Raw Images (per-round masks) ---
        jobs = _detection_jobs(puncta_config, r1_channels, r1_nuc_ch, r2_nuc_ch, r1_path, r2_path)

        masks = {}            # per-round nuclear masks, read once per round when a detection runs

        def round_mask(rnd):
            if rnd not in masks:
                path = stages.round_mask_path(rnd)
                masks[rnd] = io.read_label_tif(path) if path else None
            return masks[rnd]

        detected = []  # (round, channel) of every channel whose raw-space CSV was written
        job_count = max(len(jobs), 1)
        job_i = 0
        reports_dir = output_dir / constants.REPORTS_DIR

        for rnd, ch, raw_path, params, chosen, upstream in jobs:
            job_base = 25 + int((job_i / job_count) * 10)
            fp = stages.stage_fingerprint("detect", f"{rnd} - {ch}", params, inputs=(raw_path,),
                                          upstream=upstream)
            raw_csv = constants.puncta_csv_path(reports_dir, stages.raw_puncta_name(rnd, ch))
            if stages.stage_is_fresh("detect", f"{rnd} - {ch}", fp):
                detected.append((rnd, ch))
                job_i += 1
                continue
            _update(job_base, f"Detecting puncta (raw): {rnd} {ch}...")

            # The same step the GUI re-detects a channel with. It keeps the
            # raw-space puncta, records them, and stores the channel's
            # settings for the report; the aligned rows share their puncta_id.
            try:
                sess_obj = round_session(rnd)
                ch_idx = list(sess_obj.channels).index(ch)
                # Settings from a Puncta row were chosen by a person; the
                # built-in ones, for a channel without a row, by nobody.
                with (contextlib.nullcontext() if chosen else decisions.use_actor(decisions.DEFAULT)):
                    raw_csv = stages.detect_channel(
                        rnd, ch, params=params, run_by="batch",
                        image=sess_obj.data[:, ch_idx, :, :], image_path=raw_path, voxels=sess_obj.voxels,
                        mask=round_mask(rnd),
                        progress_callback=lambda p, t, _b=job_base: _update(_b + int(p / 100 * 2),
                                                                            f"{rnd} {ch}: {t}"),
                    )
            except Exception as exc:
                logger.error("Puncta detection failed for %s %s: %s", rnd, ch, exc, exc_info=True)
                _update(job_base, f"Puncta detection failed for {rnd} {ch}: {exc}")
                job_i += 1
                continue
            if raw_csv is not None:
                detected.append((rnd, ch))
                stages.record_stage("detect", f"{rnd} - {ch}", fp, params, inputs=(raw_path,),
                                    upstream=upstream, outputs=[raw_csv])
                ran.append(f"detect {rnd} - {ch}")

            job_i += 1
        masks.clear()

        session.save_session()
        # --- 5. Registration (RANSAC) ---
        reg_params = _register_params(max_ransac_distance)
        reg_fp = stages.stage_fingerprint("register", "", reg_params, upstream=REGISTER_UPSTREAM)
        if stages.stage_is_fresh("register", "", reg_fp) and session.get_data("shift") is not None:
            shift = np.asarray(session.get_data("shift"), dtype=float)
        else:
            shift, _ = registration.calculate_session_registration(
                seg_results['R1'][1], seg_results['R2'][1],
                voxels=r1_voxels(),
                max_distance=max_ransac_distance,
                progress_callback=lambda p, t: _update(35 + int(p * 0.1), t)
            )
            if shift is None:
                raise errors.RegistrationError(
                    "Registration failed: could not calculate a valid shift. "
                    "Check that both rounds have sufficient nuclear signal."
                )
            stages.record_stage("register", "", reg_fp, reg_params, upstream=REGISTER_UPSTREAM)
            ran.append("register")

        session.save_session()
        # --- 6. Canvas Generation (Rigid + Deformable Warping) ---
        aligned_dir = output_dir / constants.ALIGNED_DIR
        canvas_params = _canvas_params(apply_warp, r1_nuc_ch, r2_nuc_ch)
        canvas_up = CANVAS_UPSTREAM
        canvas_fp = stages.stage_fingerprint("canvas", "", canvas_params, inputs=(r1_path, r2_path),
                                             upstream=canvas_up)
        if stages.stage_is_fresh("canvas", "", canvas_fp):
            canvas_record = (session.get_data("registration", default={}) or {}).get("canvas") or {}
            bspline_transform = canvas_record.get("bspline")
        else:
            before = _registered_paths()
            layer_lists = {}
            for prefix, round_nuc in (("R1", r1_nuc_ch), ("R2", r2_nuc_ch)):
                sess_obj = round_session(prefix)
                layers = [{'name': f"{prefix} - {ch}", 'data': sess_obj.data[:, i, :, :],
                           'scale': sess_obj.voxels, 'is_label': False} for i, ch in enumerate(sess_obj.channels)]
                mask_name = f"{prefix} - {round_nuc}{constants.MASKS_SUFFIX}"
                mask_path = io.find_mask(seg_dir, mask_name)
                if mask_path:
                    layers.append({'name': mask_name, 'data': io.read_label_tif(mask_path),
                                   'scale': sess_obj.voxels, 'is_label': True})
                layer_lists[prefix] = layers
            try:
                _, bspline_transform, canvas_offset = registration.generate_global_canvas(
                    r1_layers_data=layer_lists["R1"], r2_layers_data=layer_lists["R2"],
                    shift=shift, output_dir=aligned_dir, apply_warp=apply_warp,
                    progress_callback=lambda p, t: _update(45 + int(p * 0.15), t),
                    r2_channel_names={r1_nuc_ch: r2_nuc_ch},
                )
            except Exception as exc:
                raise errors.RegistrationError(
                    f"Canvas generation (alignment + warping) failed: {exc}"
                ) from exc
            del layer_lists
            session.update_data("canvas_scale", round_session("R1").voxels)
            outputs = sorted(_registered_paths() - before)
            bspline_file = aligned_dir / constants.REGISTRATION_TRANSFORM_FILE
            stages.record_stage("canvas", "", canvas_fp, canvas_params, inputs=(r1_path, r2_path),
                                upstream=canvas_up,
                                outputs=outputs + ([bspline_file] if bspline_file.exists() else []))
            ran.append("canvas")
        # The raw images are not needed past this point.
        loaded.clear()
        gc.collect()

        session.save_session()
        # --- 7. Consensus Nuclei ---
        r1_aligned_mask = aligned_dir / f"Aligned_R1_{r1_nuc_ch}{constants.MASKS_SUFFIX}.tif"
        r2_warped_mask = aligned_dir / f"Warped_R2_{r2_nuc_ch}{constants.MASKS_SUFFIX}.tif"
        if not r2_warped_mask.exists():
            r2_warped_mask = aligned_dir / f"Aligned_R2_{r2_nuc_ch}{constants.MASKS_SUFFIX}.tif"

        consensus_params = _consensus_params(overlap_method, match_threshold, remove_border_nuclei)
        consensus_fp = stages.stage_fingerprint("consensus", "", consensus_params, upstream=CONSENSUS_UPSTREAM)
        consensus_file = io.mask_path(seg_dir, constants.CONSENSUS_MASKS_NAME)
        if stages.stage_is_fresh("consensus", "", consensus_fp):
            merged_mask = io.read_label_tif(consensus_file)
        else:
            if not (r1_aligned_mask.exists() and r2_warped_mask.exists()):
                raise errors.RegistrationError(
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
                    voxel_size=r1_voxels(),
                    remove_border=remove_border_nuclei,
                    progress_callback=lambda p, t: _update(60 + int(p * 0.1), t)
                )
            except Exception as exc:
                raise errors.DetectionError(
                    f"Consensus nuclei matching failed: {exc}"
                ) from exc
            stages.record_stage("consensus", "", consensus_fp, consensus_params, upstream=CONSENSUS_UPSTREAM,
                                outputs=[consensus_file])
            ran.append("consensus")

        session.save_session()
        # --- 8. Transform Puncta to Aligned Space ---
        reports_dir.mkdir(exist_ok=True, parents=True)

        aligned_puncta_layers = []  # Collect for colocalization analysis
        job_count = max(len(detected), 1)
        job_i = 0
        for rnd, ch in detected:
            prefix_str = constants.ALIGNED_PREFIX if (rnd == "R1" or bspline_transform is None) else constants.WARPED_PREFIX
            job_base = 70 + int((job_i / job_count) * 15)
            key = f"{rnd} - {ch}"
            transform_params = _transform_params(remove_extranuclear_puncta)
            transform_up = _transform_upstream(key)
            fp = stages.stage_fingerprint("transform", key, transform_params, upstream=transform_up)
            name = f"{prefix_str} {rnd} - {ch}{constants.PUNCTA_SUFFIX}"
            if stages.stage_is_fresh("transform", key, fp):
                csv_out = constants.puncta_csv_path(reports_dir, name)
            else:
                _update(job_base, f"Transforming puncta: {rnd} {ch}...")
                # The same step the GUI rewarps a re-detected channel with: the raw
                # CSV through the registration saved at canvas generation.
                csv_out = stages.transform_channel(rnd, ch, remove_extranuclear=remove_extranuclear_puncta,
                                                   run_by="batch", consensus_mask=merged_mask)
                stages.record_stage("transform", key, fp, transform_params, upstream=transform_up,
                                    outputs=[csv_out] if csv_out is not None else [])
                ran.append(f"transform {key}")

            # Build layer data dict for colocalization analysis
            transformed = pd.read_csv(csv_out, float_precision="round_trip") if csv_out is not None else None
            if transformed is not None and len(transformed) > 0:
                aligned_puncta_layers.append({
                    'name': name,
                    'data': transformed[['Z', 'Y', 'X']].to_numpy(float),
                    'scale': np.array(r1_voxels()),
                    'translate': np.zeros(3),
                    'nucleus_ids': transformed['Nucleus_ID'].to_numpy(float),
                    'puncta_id': transformed['puncta_id'].to_numpy(float),
                })

            job_i += 1

        session.save_session()
        # --- 9. Colocalization Analysis ---
        has_coloc = (pairwise_rules and len(pairwise_rules) > 0) or (tri_rules and len(tri_rules) > 0)
        report_params = _report_params(pairwise_rules, tri_rules, distribution_bins)
        report_up = _report_upstream(detected)
        report_fp = stages.stage_fingerprint("report", "", report_params, upstream=report_up)
        report_filename = f"zFISHer_Report{constants.EXCEL_SUFFIX}"
        if has_coloc and aligned_puncta_layers and not stages.stage_is_fresh("report", "", report_fp):
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

            try:
                report_path = analysis.run_colocalization_analysis(
                    layers_data=aligned_puncta_layers,
                    rules=resolved_pairwise,
                    filename=report_filename,
                    r1_path=str(r1_path),
                    r2_path=str(r2_path),
                    output_dir=str(reports_dir),
                    tri_rules=resolved_tri if resolved_tri else None,
                    total_nuclei=total_nuclei,
                    nucleus_ids=nucleus_ids,
                    distribution_bins=distribution_bins,
                )
                _update(95, "Colocalization report saved.")
                if report_path is not None:
                    stages.record_stage("report", "", report_fp, report_params, upstream=report_up,
                                        outputs=[report_path])
                    ran.append("report")
            except Exception as exc:
                logger.error("Colocalization analysis failed: %s", exc, exc_info=True)
                _update(95, f"Colocalization failed: {exc}")

    # --- Cleanup ---
    del merged_mask
    gc.collect()

    # Ensure the session JSON is fully written before exiting
    session.save_session()

    _update(100, "Done.")
    logger.info("Pipeline complete for %s; stages run: %s", output_dir, ", ".join(ran) or "none")
    return ran


# --- Each stage's settings and upstream stages ------------------------------------------
# One definition, used by a run and by ``plan_full_zfisher_pipeline``, so the
# plan cannot drift from what a run does. Settings only enter a stage's
# fingerprint when set, where a later version added them, so that runs made
# before still count as up to date.

REGISTER_UPSTREAM = [("segment", "")]
CANVAS_UPSTREAM = [("segment", ""), ("register", "")]
CONSENSUS_UPSTREAM = [("canvas", "")]


def _segment_params(seg_method, merge_splits, r1_nuc, r2_nuc):
    method = "cellpose" if str(seg_method).lower() == "cellpose" else "classical"
    return {"method": method, "merge_splits": bool(merge_splits), "channels": {"R1": r1_nuc, "R2": r2_nuc}}


def _detection_jobs(puncta_config, r1_channels, r1_nuc, r2_nuc, r1_path, r2_path):
    """Every detection a run does, in order: ``(round, channel, raw image,
    settings, chosen, upstream)``. The channels are the template's Puncta rows
    (``puncta_config``), each in both rounds unless its row names one; without
    rows, every channel of R1 that is not a nuclear stain, with the built-in
    settings (``chosen`` False: nobody chose them)."""
    if puncta_config:
        channels = list(puncta_config.keys())
    else:
        nuclear = {r1_nuc.upper(), r2_nuc.upper()}
        channels = [ch for ch in r1_channels if ch.upper() not in nuclear]
    jobs = []
    for rnd, raw_path in (("R1", r1_path), ("R2", r2_path)):
        for ch in channels:
            chosen = bool(puncta_config and ch in puncta_config)
            if chosen:
                ch_params = dict(puncta_config[ch])
                ch_round = ch_params.pop("round", "Both")
                if ch_round != "Both" and ch_round != rnd:
                    continue
                nuclei_only = ch_params.pop("nuclei_only", True)
            else:
                ch_params = {
                    # Sigma and the top-hat are left to detection's own
                    # fallbacks (puncta.DETECTOR_DEFAULTS), as before.
                    'threshold_rel': default(DetectionParams, "threshold_rel"),
                    'min_distance': default(DetectionParams, "min_distance"),
                    'method': "Local Maxima",
                }
                nuclei_only = True
            params = dict(ch_params, nuclei_only=nuclei_only)
            jobs.append((rnd, ch, raw_path, params, chosen, [("segment", "")] if nuclei_only else []))
    return jobs


def effective_puncta_config(r1_path, r2_path, puncta_config=None, r1_nuclear_channel=None,
                            r2_nuclear_channel=None, **_settings):
    """The detection settings per channel a run with these settings uses:
    ``puncta_config`` when given, else the built-in settings for every channel
    of R1 that is not a nuclear stain, in the form ``puncta_config`` takes, so
    that writing it back changes no stage's fingerprint."""
    if puncta_config:
        return {ch: dict(params) for ch, params in puncta_config.items()}
    r1_channels, r2_channels = _channels(Path(r1_path)), _channels(Path(r2_path))
    r1_nuc = _nuclear_channel(r1_nuclear_channel, r1_channels, "R1")
    r2_nuc = _nuclear_channel(r2_nuclear_channel, r2_channels, "R2")
    return {ch: dict(params) for rnd, ch, _, params, _, _ in
            _detection_jobs(None, r1_channels, r1_nuc, r2_nuc, Path(r1_path), Path(r2_path)) if rnd == "R1"}


def _register_params(max_ransac_distance):
    return {"max_distance": max_ransac_distance}


def _canvas_params(apply_warp, r1_nuc, r2_nuc):
    return {"apply_warp": bool(apply_warp), "r2_channel_names": {r1_nuc: r2_nuc}}


def _consensus_params(overlap_method, match_threshold, remove_border_nuclei):
    params = {"method": overlap_method, "match_threshold": match_threshold or 0}
    if remove_border_nuclei:
        params["remove_border_nuclei"] = True
    return params


def _transform_params(remove_extranuclear_puncta):
    return {"remove_extranuclear": bool(remove_extranuclear_puncta)}


def _transform_upstream(key):
    return [("detect", key), ("canvas", ""), ("consensus", "")]


def _report_params(pairwise_rules, tri_rules, distribution_bins):
    params = {"pairwise": pairwise_rules or [], "tri": tri_rules or []}
    if distribution_bins:
        params["distribution_bins"] = list(distribution_bins)
    return params


def _report_upstream(detected):
    return [("transform", f"{rnd} - {ch}") for rnd, ch in detected] + [("consensus", "")]


def _registered_paths():
    """The paths the session's file registry points to now."""
    return {Path(info["path"]) for info in (session.get_data("processed_files", default={}) or {}).values()
            if isinstance(info, dict) and info.get("path")}


def _channels(path, loaded_session=None):
    """The channel names of the image at ``path``, without reading its pixels."""
    if loaded_session is not None:
        return list(loaded_session.channels)
    names = io.peek_channel_names(path)
    if names is None:
        return list(io.load_image_session(path).channels)
    return list(names)


def _nuclear_channel(requested, channels, round_id):
    if requested:
        return requested
    found = io.find_nuclear_channel(channels)
    if found is None:
        found = channels[0]
        logger.warning("No known nuclear stain found in %s. Falling back to first channel: %s", round_id, found)
    return found


def _open_previous_run(output_dir, r1_path, r2_path, read_only=False, in_place=False):
    """Open the session a previous pipeline run left in ``output_dir``, for
    the same two images, so stages that are up to date are not run again.
    False when there is no session there, or one that no pipeline run with
    stage records made (a GUI session, a batch run of zFISHer before 3.0),
    which is not touched. ``read_only`` opens it without writing anything;
    ``in_place`` saves it back into its newest file instead of a new one."""
    folder = output_dir / constants.SESSIONS_DIR
    files = sorted(folder.glob("zfisher_session_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[-1])
                   if p.stem.rsplit("_", 1)[-1].isdigit() else -1) if folder.exists() else []
    if not files:
        return False
    import json
    data = json.loads(files[-1].read_text())
    same_inputs = all(Path(data.get(k) or "").name == Path(p).name for k, p in (("r1_path", r1_path),
                                                                                ("r2_path", r2_path)))
    if not data.get("stages") or "pipeline_settings" not in data or not same_inputs:
        return False
    if read_only:
        session.current().load_into(files[-1], read_only=True)
    else:
        session.load_session_file(files[-1], in_place=in_place)
    return True
