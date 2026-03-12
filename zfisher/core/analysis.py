# zfisher/core/analysis.py
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree

from .report import export_report
from . import session
from .. import constants

logger = logging.getLogger(__name__)


# =====================================================================
# Computation helpers
# =====================================================================

def _to_world(layer):
    """Convert a layer dict's pixel data to world coordinates (microns)."""
    return layer['data'] * layer['scale'] + layer.get('translate', 0)


def calculate_distances(points_layers_data):
    """
    Calculates nearest neighbor distances between all pairs of points layers.

    Parameters
    ----------
    points_layers_data : list[dict]
        A list of dictionaries, where each dictionary represents a points layer
        and contains 'name', 'data' (in pixels), 'scale', and 'translate' keys.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the nearest neighbor analysis, with columns
        for source/target layers, IDs, distance, and coordinates.
    """
    results = []

    # Iterate all pairs (A -> B)
    for source in points_layers_data:
        for target in points_layers_data:
            if source['name'] == target['name']:
                continue

            # Convert to world coordinates (microns) using scale + translate
            # Napari world = data * scale + translate
            src_data = source['data'] * source['scale'] + source.get('translate', 0)
            tgt_data = target['data'] * target['scale'] + target.get('translate', 0)

            if len(src_data) == 0 or len(tgt_data) == 0:
                continue

            # Build KDTree for target
            tree = cKDTree(tgt_data)

            # Query nearest neighbors
            dists, idxs = tree.query(src_data)

            # Collect results
            src_nuc_ids = source.get('nucleus_ids')
            for i, (d, idx) in enumerate(zip(dists, idxs)):
                nuc_id = int(src_nuc_ids[i]) if src_nuc_ids is not None and i < len(src_nuc_ids) else None
                results.append({
                    "Source_Layer": source['name'],
                    "Source_ID": i,
                    "Nucleus_ID": nuc_id,
                    "Target_Layer": target['name'],
                    "Target_ID": idx,
                    "Distance_um": d,
                    "Z": src_data[i][0],
                    "Y": src_data[i][1],
                    "X": src_data[i][2]
                })

    return pd.DataFrame(results)


def calculate_pairwise_colocalization(df, rules):
    """
    Filters the pairwise distances DataFrame by colocalization rules.

    Parameters
    ----------
    df : pd.DataFrame
        The distances DataFrame from `calculate_distances`.
    rules : list[dict]
        Each dict has 'source', 'target', 'threshold'.

    Returns
    -------
    pd.DataFrame
        Filtered rows that satisfy the colocalization rules.
    list[dict]
        Metadata entries for each rule.
    """
    coloc_rows = []
    meta_entries = []

    for rule in rules:
        src = rule['source']
        tgt = rule['target']
        thresh = rule['threshold']

        meta_entries.append({"Key": f"Coloc Rule: {src} -> {tgt}", "Value": f"<= {thresh} um"})

        mask = (df['Source_Layer'] == src) & \
               (df['Target_Layer'] == tgt) & \
               (df['Distance_um'] <= thresh)

        subset = df[mask].copy()
        subset['Coloc_Threshold_um'] = thresh
        coloc_rows.append(subset)

    df_coloc = pd.concat(coloc_rows) if coloc_rows else pd.DataFrame()
    return df_coloc, meta_entries


def calculate_tri_colocalization(points_layers_data, tri_rules):
    """
    Tri-colocalization with exclusive greedy matching.

    For each rule (anchor, channel_a, channel_b, threshold):
    1. Each ChA punctum is assigned to its closest anchor (KDTree query).
    2. Each ChB punctum is assigned to its closest anchor.
    3. Anchors with at least one ChA AND one ChB within cutoff form candidates.
    4. Greedy loop: pick the best triplet (lowest chA_dist + chB_dist),
       consume the ChA and ChB puncta (anchor stays), repeat.

    Parameters
    ----------
    points_layers_data : list[dict]
        Layer dicts with 'name', 'data', 'scale', and optionally 'translate'.
    tri_rules : list[dict]
        Each dict has 'anchor', 'channel_a', 'channel_b', 'threshold'.

    Returns
    -------
    pd.DataFrame
        Hit table with columns: Anchor_Layer, Anchor_ID, Channel_A, ChA_ID,
        ChA_Distance_um, Channel_B, ChB_ID, ChB_Distance_um, Coloc_Threshold_um.
    list[dict]
        Metadata entries for each rule.
    """
    layers_by_name = {l['name']: l for l in points_layers_data}
    all_hits = []
    meta_entries = []

    for rule in tri_rules:
        anchor_name = rule['anchor']
        ch_a_name = rule['channel_a']
        ch_b_name = rule['channel_b']
        thresh = rule['threshold']

        meta_entries.append({
            "Key": f"Tri-Coloc Rule: {anchor_name} <-> {ch_a_name} + {ch_b_name}",
            "Value": f"<= {thresh} um"
        })

        # Validate layers exist
        if anchor_name not in layers_by_name or ch_a_name not in layers_by_name or ch_b_name not in layers_by_name:
            logger.warning("Tri-coloc rule references missing layer(s). Skipping.")
            continue

        anchor_world = _to_world(layers_by_name[anchor_name])
        ch_a_world = _to_world(layers_by_name[ch_a_name])
        ch_b_world = _to_world(layers_by_name[ch_b_name])

        if len(anchor_world) == 0 or len(ch_a_world) == 0 or len(ch_b_world) == 0:
            continue

        # Build KDTree of anchor points
        anchor_tree = cKDTree(anchor_world)

        # Assign each ChA punctum to its closest anchor
        ch_a_dists, ch_a_anchor_ids = anchor_tree.query(ch_a_world)
        # Assign each ChB punctum to its closest anchor
        ch_b_dists, ch_b_anchor_ids = anchor_tree.query(ch_b_world)

        # Build lookup: anchor_id -> list of (chA_id, distance)
        anchor_to_ch_a = defaultdict(list)
        for ch_a_id in range(len(ch_a_world)):
            anchor_id = int(ch_a_anchor_ids[ch_a_id])
            dist = float(ch_a_dists[ch_a_id])
            if dist <= thresh:
                anchor_to_ch_a[anchor_id].append((ch_a_id, dist))

        anchor_to_ch_b = defaultdict(list)
        for ch_b_id in range(len(ch_b_world)):
            anchor_id = int(ch_b_anchor_ids[ch_b_id])
            dist = float(ch_b_dists[ch_b_id])
            if dist <= thresh:
                anchor_to_ch_b[anchor_id].append((ch_b_id, dist))

        # Build all candidate triplets from anchors that have both
        used_ch_a = set()
        used_ch_b = set()

        while True:
            # Rebuild candidates each round (after removals)
            candidates = []
            for anchor_id in anchor_to_ch_a:
                if anchor_id not in anchor_to_ch_b:
                    continue
                for ch_a_id, ch_a_dist in anchor_to_ch_a[anchor_id]:
                    if ch_a_id in used_ch_a:
                        continue
                    for ch_b_id, ch_b_dist in anchor_to_ch_b[anchor_id]:
                        if ch_b_id in used_ch_b:
                            continue
                        candidates.append((
                            ch_a_dist + ch_b_dist,
                            anchor_id, ch_a_id, ch_a_dist, ch_b_id, ch_b_dist
                        ))

            if not candidates:
                break

            # Pick the best triplet (lowest total distance)
            candidates.sort(key=lambda x: x[0])
            _, best_anchor, best_ch_a, best_ch_a_dist, best_ch_b, best_ch_b_dist = candidates[0]

            # Record hit with coordinates and nucleus ID
            a_coord = anchor_world[best_anchor]
            ca_coord = ch_a_world[best_ch_a]
            cb_coord = ch_b_world[best_ch_b]

            anchor_layer = layers_by_name[anchor_name]
            nuc_ids = anchor_layer.get('nucleus_ids')
            nuc_id = int(nuc_ids[best_anchor]) if nuc_ids is not None and best_anchor < len(nuc_ids) else None

            all_hits.append({
                'Anchor_Layer': anchor_name,
                'Anchor_ID': best_anchor,
                'Nucleus_ID': nuc_id,
                'Anchor_Z': a_coord[0],
                'Anchor_Y': a_coord[1],
                'Anchor_X': a_coord[2],
                'Channel_A': ch_a_name,
                'ChA_ID': best_ch_a,
                'ChA_Z': ca_coord[0],
                'ChA_Y': ca_coord[1],
                'ChA_X': ca_coord[2],
                'ChA_Distance_um': best_ch_a_dist,
                'Channel_B': ch_b_name,
                'ChB_ID': best_ch_b,
                'ChB_Z': cb_coord[0],
                'ChB_Y': cb_coord[1],
                'ChB_X': cb_coord[2],
                'ChB_Distance_um': best_ch_b_dist,
                'Coloc_Threshold_um': thresh
            })

            # Consume ChA and ChB (anchor stays)
            used_ch_a.add(best_ch_a)
            used_ch_b.add(best_ch_b)

    df_tri = pd.DataFrame(all_hits)
    return df_tri, meta_entries


def calculate_per_nucleus_counts(points_layers_data):
    """
    Counts the number of puncta per nucleus for each channel.

    Filters out nuclear stain, centroid, and consensus layers. Only layers that
    carry a 'nucleus_ids' array are included.

    Parameters
    ----------
    points_layers_data : list[dict]
        Layer dicts with 'name', 'data', and optionally 'nucleus_ids'.

    Returns
    -------
    pd.DataFrame
        Pivot table with Nuclei_ID as rows and channel names as columns.
        Nucleus ID 0 (background) is excluded.
    """
    SKIP_PATTERNS = [
        session.get_nuclear_channel().upper(),
        constants.CENTROIDS_SUFFIX.upper(),
        constants.CONSENSUS_MASKS_NAME.upper(),
    ]

    records = []
    for layer in points_layers_data:
        name = layer['name']
        nucleus_ids = layer.get('nucleus_ids')
        if nucleus_ids is None:
            continue

        # Skip non-puncta layers
        name_upper = name.upper()
        if any(pat in name_upper for pat in SKIP_PATTERNS):
            continue

        for nid in nucleus_ids:
            nid_int = int(nid)
            if nid_int == 0:
                continue
            records.append({'Nuclei_ID': nid_int, 'Channel': name})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    pivot = df.groupby(['Nuclei_ID', 'Channel']).size().unstack(fill_value=0)
    pivot = pivot.reset_index().sort_values('Nuclei_ID')
    return pivot


def calculate_stats(per_nucleus_df, total_nuclei=None):
    """
    Computes per-channel summary statistics from the per-nucleus counts table.

    Parameters
    ----------
    per_nucleus_df : pd.DataFrame
        Pivot table from `calculate_per_nucleus_counts` with Nuclei_ID as first
        column and channel counts as remaining columns.
    total_nuclei : int, optional
        Actual number of nuclei in the consensus mask. If None, falls back to
        counting unique IDs in the dataframe.

    Returns
    -------
    pd.DataFrame
        Rows: one per channel. Columns: Channel, Total_Nuclei, Raw_Sum,
        Mean_per_Nucleus, StdDev_per_Nucleus, CV_pct.
    """
    if per_nucleus_df.empty:
        return pd.DataFrame()

    channels = [c for c in per_nucleus_df.columns if c != 'Nuclei_ID']
    if total_nuclei is None:
        total_nuclei = int(per_nucleus_df['Nuclei_ID'].nunique())

    rows = []
    for ch in channels:
        # Extend counts to include nuclei with 0 puncta in this channel
        present = per_nucleus_df[ch].values
        n_missing = max(total_nuclei - len(present), 0)
        zeros = np.zeros(n_missing)
        all_counts = np.concatenate([present, zeros])

        raw_sum = int(all_counts.sum())
        mean = float(all_counts.mean())
        std = float(all_counts.std(ddof=0))
        cv = (std / mean * 100) if mean > 0 else 0.0

        rows.append({
            'Channel': ch,
            'Total_Nuclei': total_nuclei,
            'Raw_Sum': raw_sum,
            'Mean_per_Nucleus': round(mean, 4),
            'StdDev_per_Nucleus': round(std, 4),
            'CV_pct': round(cv, 2),
        })

    return pd.DataFrame(rows)


def calculate_distribution(per_nucleus_df, total_nuclei=None):
    """
    Bins nuclei by puncta count for each channel.

    Parameters
    ----------
    per_nucleus_df : pd.DataFrame
        Pivot table from `calculate_per_nucleus_counts`.
    total_nuclei : int, optional
        Actual number of nuclei in the consensus mask. If None, falls back to
        counting unique IDs in the dataframe.

    Returns
    -------
    pd.DataFrame
        Rows: one per channel. Columns: Channel, Nuclei_0_puncta,
        Nuclei_1_puncta, Nuclei_2_puncta, Nuclei_3to10_puncta,
        Nuclei_gt10_puncta.
    """
    if per_nucleus_df.empty:
        return pd.DataFrame()

    channels = [c for c in per_nucleus_df.columns if c != 'Nuclei_ID']
    if total_nuclei is None:
        total_nuclei = int(per_nucleus_df['Nuclei_ID'].nunique())

    rows = []
    for ch in channels:
        present = per_nucleus_df[ch].values
        zeros_count = max(total_nuclei - len(present), 0)

        n0 = int((present == 0).sum()) + zeros_count
        n1 = int((present == 1).sum())
        n2 = int((present == 2).sum())
        n3_10 = int(((present >= 3) & (present <= 10)).sum())
        ngt10 = int((present > 10).sum())

        rows.append({
            'Channel': ch,
            'Nuclei_0_puncta': n0,
            'Nuclei_1_puncta': n1,
            'Nuclei_2_puncta': n2,
            'Nuclei_3to10_puncta': n3_10,
            'Nuclei_gt10_puncta': ngt10,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Orchestrator
# =====================================================================

def run_colocalization_analysis(layers_data, rules, filename, r1_path, r2_path, output_dir, tri_rules=None, total_nuclei=None):
    """
    Core Orchestrator for Step 7 & 8.
    Processes puncta distances and exports the master report.
    """
    # 1. Pairwise nearest-neighbor distances
    df = calculate_distances(layers_data)

    if df.empty:
        logger.warning("No puncta found to analyze.")
        return None

    # 2. Pairwise colocalization filtering
    df_coloc = pd.DataFrame()
    coloc_meta = []
    if rules:
        df_coloc, coloc_meta = calculate_pairwise_colocalization(df, rules)

    # 3. Tri-colocalization (needs raw layer coordinates, not the pairwise df)
    tri_coloc_df = pd.DataFrame()
    tri_coloc_meta = []
    if tri_rules:
        tri_coloc_df, tri_coloc_meta = calculate_tri_colocalization(layers_data, tri_rules)

    # 4. Per-nucleus puncta counts + derived stats
    per_nucleus_df = calculate_per_nucleus_counts(layers_data)
    stats_df = calculate_stats(per_nucleus_df, total_nuclei=total_nuclei)
    distribution_df = calculate_distribution(per_nucleus_df, total_nuclei=total_nuclei)

    # 5. Export multi-sheet Excel report
    save_path = Path(output_dir) / filename
    final_path = export_report(
        df,
        save_path,
        r1_path=r1_path,
        r2_path=r2_path,
        output_dir=output_dir,
        coloc_df=df_coloc,
        coloc_meta=coloc_meta,
        tri_coloc_df=tri_coloc_df,
        tri_coloc_meta=tri_coloc_meta,
        per_nucleus_df=per_nucleus_df,
        stats_df=stats_df,
        distribution_df=distribution_df
    )

    # 6. Session Tracking
    if final_path.exists():
        session.set_processed_file(
            layer_name="Master_Analysis_Report",
            path=str(final_path),
            layer_type="report"
        )

    return final_path
