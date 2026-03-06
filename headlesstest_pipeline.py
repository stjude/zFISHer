import numpy as np
from pathlib import Path
import gc
import tifffile
from zfisher.core import session, io, segmentation, registration, puncta, analysis
from zfisher import constants

def run_headless_full_pipeline():
    """
    Validates the complete zFISHer automated backbone.
    Conversion -> Segmentation -> Puncta (raw) -> Registration -> Warping -> Consensus -> Transform Puncta -> Analysis.
    """
    # --- STEP 1: SETUP & SESSION INITIALIZATION ---
    input_r1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
    input_r2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")
    output_dir = Path.home() / "zFISHer_Headless_FullTest"

    print(f"--- STEP 1: Starting Headless Session ---")
    success = session.initialize_new_session(
        output_dir=output_dir,
        r1_path=input_r1,
        r2_path=input_r2,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    if not success:
        print("Session already exists. Continuing with existing data...")

    # --- STEP 2: DAPI MAPPING (SEGMENTATION) ---
    print(f"\n--- STEP 2: DAPI Mapping (Segmentation) ---")
    r1_path = output_dir / "input" / "R1_converted.ome.tif"
    r2_path = output_dir / "input" / "R2_converted.ome.tif"

    r1_session = io.load_image_session(r1_path)
    r2_session = io.load_image_session(r2_path)

    r1_dapi = io.get_channel_data(r1_session, constants.DAPI_CHANNEL_NAME)
    r2_dapi = io.get_channel_data(r2_session, constants.DAPI_CHANNEL_NAME)

    seg_results = segmentation.process_session_dapi(
        r1_data=r1_dapi,
        r2_data=r2_dapi,
        output_dir=output_dir,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    # --- STEP 3: PUNCTA DETECTION ON RAW IMAGES ---
    print(f"\n--- STEP 3: Puncta Detection on Raw Images ---")
    seg_dir = output_dir / constants.SEGMENTATION_DIR
    puncta_channels = [ch for ch in r1_session.channels if ch.upper() != constants.DAPI_CHANNEL_NAME.upper()]
    print(f"Found puncta channels for analysis: {puncta_channels}")

    r1_mask_path = seg_dir / f"R1 - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    r2_mask_path = seg_dir / f"R2 - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
    r1_mask = tifffile.imread(r1_mask_path) if r1_mask_path.exists() else None
    r2_mask = tifffile.imread(r2_mask_path) if r2_mask_path.exists() else None

    puncta_params = {'threshold_rel': 0.15, 'min_distance': 3, 'method': "Local Maxima", 'nuclei_only': True}
    raw_puncta_results = {}

    for rnd, rnd_mask, sess_obj in [("R1", r1_mask, r1_session), ("R2", r2_mask, r2_session)]:
        for ch in puncta_channels:
            ch_idx = list(sess_obj.channels).index(ch)
            ch_data = sess_obj.data[:, ch_idx, :, :]
            print(f"Detecting spots (raw) in {rnd} - {ch}...")
            result = puncta.process_puncta_detection(
                image_data=ch_data,
                mask_data=rnd_mask,
                voxels=sess_obj.voxels,
                params=puncta_params,
            )
            raw_puncta_results[(rnd, ch)] = result
            print(f"  Found {len(result)} raw spots.")

    # --- STEP 4: REGISTRATION (RANSAC) ---
    print(f"\n--- STEP 4: Registration (RANSAC) ---")
    r1_centroids = seg_results['R1'][1]
    r2_centroids = seg_results['R2'][1]

    print(f"DIAGNOSTIC (pipeline): R1 Centroids shape: {r1_centroids.shape if r1_centroids is not None else 'None'}")
    if r1_centroids is not None and r1_centroids.size > 0:
        print(f"DIAGNOSTIC (pipeline): R1 Centroids mean (Z,Y,X): {np.mean(r1_centroids, axis=0)}")
    print(f"DIAGNOSTIC (pipeline): R2 Centroids shape: {r2_centroids.shape if r2_centroids is not None else 'None'}")
    if r2_centroids is not None and r2_centroids.size > 0:
        print(f"DIAGNOSTIC (pipeline): R2 Centroids mean (Z,Y,X): {np.mean(r2_centroids, axis=0)}")

    shift, rmsd = registration.calculate_session_registration(
        r1_centroids,
        r2_centroids,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )

    print(f"DIAGNOSTIC (pipeline): Calculated shift from registration: {shift}")

    if shift is None:
        print("FAILED: Registration failed. Aborting.")
        return

    # --- STEP 5: GLOBAL CANVAS (WARPING) ---
    print(f"\n--- STEP 5: Global Canvas (Headless Warping) ---")
    r1_layers_data = []
    r2_layers_data = []

    for i, ch_name in enumerate(r1_session.channels):
        r1_layers_data.append({'name': f"R1 - {ch_name}", 'data': r1_session.data[:, i, :, :], 'scale': r1_session.voxels, 'is_label': False})
    for i, ch_name in enumerate(r2_session.channels):
        r2_layers_data.append({'name': f"R2 - {ch_name}", 'data': r2_session.data[:, i, :, :], 'scale': r2_session.voxels, 'is_label': False})

    # Add the newly created masks to the layer data for warping
    for prefix, session_obj, layer_list in [("R1", r1_session, r1_layers_data), ("R2", r2_session, r2_layers_data)]:
        dapi_layer_name = f"{prefix} - {constants.DAPI_CHANNEL_NAME}"
        mask_layer_name = f"{dapi_layer_name}{constants.MASKS_SUFFIX}"
        mask_path = seg_dir / f"{mask_layer_name}.tif"

        if mask_path.exists():
            layer_list.append({
                'name': mask_layer_name,
                'data': tifffile.imread(mask_path),
                'scale': session_obj.voxels,
                'is_label': True
            })

    aligned_dir = output_dir / constants.ALIGNED_DIR
    aligned_dir.mkdir(exist_ok=True, parents=True)

    _, bspline_transform, canvas_offset = registration.generate_global_canvas(
        r1_layers_data=r1_layers_data,
        r2_layers_data=r2_layers_data,
        shift=shift,
        output_dir=aligned_dir,
        apply_warp=True,
        progress_callback=lambda p, t: print(f"[{p}%] {t}")
    )
    session.update_data("canvas_scale", r1_session.voxels)

    # --- STEP 6: MATCH NUCLEI (CONSENSUS) ---
    print(f"\n--- STEP 6: Match Nuclei (Headless Consensus) ---")
    r1_aligned_mask_path = aligned_dir / "Aligned_R1_DAPI_masks.tif"
    r2_warped_mask_path = aligned_dir / "Warped_R2_DAPI_masks.tif"

    if not r2_warped_mask_path.exists():
        r2_warped_mask_path = aligned_dir / "Aligned_R2_DAPI_masks.tif"

    if r1_aligned_mask_path.exists() and r2_warped_mask_path.exists():
        merged_mask, pts1 = segmentation.process_consensus_nuclei(
            mask1=tifffile.imread(r1_aligned_mask_path),
            mask2=tifffile.imread(r2_warped_mask_path),
            output_dir=output_dir,
            threshold=20.0,
            method="Intersection",
            progress_callback=lambda p, t: print(f"[{p}%] {t}")
        )
        print(f"SUCCESS: Consensus generated with {len(pts1)} matched nuclei.")
    else:
        print("FAILED: Aligned masks not found.")
        return

    # --- STEP 7: TRANSFORM PUNCTA TO ALIGNED SPACE ---
    print(f"\n--- STEP 7: Transform Puncta to Aligned Space ---")
    puncta_layers_for_analysis = []

    for (rnd, ch), raw_data in raw_puncta_results.items():
        prefix = constants.ALIGNED_PREFIX if rnd == "R1" else constants.WARPED_PREFIX
        csv_out = seg_dir / f"{prefix}_{rnd}_{ch}{constants.PUNCTA_SUFFIX}.csv"
        puncta_layer_name = f"{prefix} {rnd} - {ch}{constants.PUNCTA_SUFFIX}"
        print(f"Transforming {rnd} - {ch} ({len(raw_data)} raw spots)...")

        results = puncta.transform_puncta_to_aligned_space(
            raw_puncta=raw_data,
            round_id=rnd,
            shift=shift,
            canvas_offset=canvas_offset,
            bspline_transform=bspline_transform if rnd == "R2" else None,
            consensus_mask=merged_mask,
            output_path=csv_out,
            layer_name=puncta_layer_name,
        )

        puncta_layers_for_analysis.append({
            'name': f"{rnd}_{ch}",
            'data': results[:, :3],  # Just ZYX coords
            'scale': r1_session.voxels,
            'nucleus_ids': results[:, 3] if len(results) > 0 else np.array([])
        })
        print(f"  {len(results)} spots in consensus nuclei.")

    # --- STEP 8: ANALYSIS & EXPORT (HEADLESS) ---
    print(f"\n--- STEP 8: Analysis & Export (Headless) ---")

    rules = []
    for ch in puncta_channels:
        rules.append({'source': f'R1_{ch}', 'target': f'R2_{ch}', 'threshold': 1.0})
    print(f"Applying colocalization rules: {rules}")

    final_report = analysis.run_colocalization_analysis(
        layers_data=puncta_layers_for_analysis,
        rules=rules,
        filename="Headless_Final_Analysis.xlsx",
        r1_path=input_r1,
        r2_path=input_r2,
        output_dir=output_dir / constants.REPORTS_DIR
    )

    if final_report:
        print(f"SUCCESS: Master Report generated: {final_report.name}")

    print(f"\n--- ALL STEPS COMPLETE ---")
    print(f"Final results in: {output_dir}")
    gc.collect()

if __name__ == "__main__":
    try:
        run_headless_full_pipeline()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
