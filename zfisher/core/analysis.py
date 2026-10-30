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
from .timing import timed_stage

logger = logging.getLogger(__name__)

# One-shot guard so the positional-fallback warning is emitted at most once.
_warned_positional_ids = False


def _stable_id(id_array, idx):
    """Return the persisted ``puncta_id`` at ``idx``, or the positional index.

    Analysis reports identify puncta by their stable ``puncta_id`` so the same
    physical punctum carries the same id across the Distances, Colocalization,
    and Tri-Colocalization sheets (and across re-runs). When a layer has no
    ``puncta_id`` (legacy in-memory layers / old CSVs), it falls back to the
    positional index and warns once so the degraded case is visible.
    """
    if id_array is not None and idx < len(id_array):
        return int(id_array[idx])
    global _warned_positional_ids
    if not _warned_positional_ids:
        logger.warning(
            "A puncta layer has no puncta_id; analysis is falling back to "
            "positional indices for reported IDs. IDs may not be stable across "
            "sheets or re-runs for this layer."
        )
        _warned_positional_ids = True
    return int(idx)


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
            src_pids = source.get('puncta_id')
            tgt_pids = target.get('puncta_id')
            for i, (d, idx) in enumerate(zip(dists, idxs)):
                nuc_id = int(src_nuc_ids[i]) if src_nuc_ids is not None and i < len(src_nuc_ids) else None
                results.append({
                    "Source_Layer": source['name'],
                    "Source_ID": _stable_id(src_pids, i),
                    "Nucleus_ID": nuc_id,
                    "Target_Layer": target['name'],
                    "Target_ID": _stable_id(tgt_pids, idx),
                    "Distance_um": d,
                    "Z": src_data[i][0],
                    "Y": src_data[i][1],
                    "X": src_data[i][2]
                })

    return pd.DataFrame(results)


def calculate_pairwise_colocalization(points_layers_data, rules):
    """
    Symmetric pairwise colocalization using radius search.

    For each rule, finds ALL unique (A, B) pairs where A is from one layer
    and B is from the other and their distance is within the cutoff.  The
    result is identical regardless of which layer is called source vs target.

    Parameters
    ----------
    points_layers_data : list[dict]
        Layer dicts with 'name', 'data', 'scale', 'translate', and
        optionally 'nucleus_ids'.
    rules : list[dict]
        Each dict has 'source', 'target', 'threshold'.

    Returns
    -------
    pd.DataFrame
        All unique colocalization pairs with distance and coordinates.
    list[dict]
        Metadata entries for each rule.
    """
    layers_by_name = {l['name']: l for l in points_layers_data}
    coloc_rows = []
    meta_entries = []

    for rule in rules:
        src_name = rule['source']
        tgt_name = rule['target']
        thresh = rule['threshold']

        meta_entries.append({
            "Key": f"Coloc Rule: {src_name} <-> {tgt_name}",
            "Value": f"<= {thresh} um"
        })

        if src_name not in layers_by_name or tgt_name not in layers_by_name:
            logger.warning("Pairwise rule references missing layer(s): %s, %s. Skipping.",
                           src_name, tgt_name)
            continue

        src_layer = layers_by_name[src_name]
        tgt_layer = layers_by_name[tgt_name]

        src_world = _to_world(src_layer)
        tgt_world = _to_world(tgt_layer)

        if len(src_world) == 0 or len(tgt_world) == 0:
            continue

        # Radius search: for each source point, find all target points within cutoff
        src_tree = cKDTree(src_world)
        tgt_tree = cKDTree(tgt_world)
        pairs = src_tree.query_ball_tree(tgt_tree, r=thresh)

        # Collect unique pairs (deduplicate A-B / B-A by always storing
        # the pair with the smaller layer name first, then by index)
        seen = set()
        src_nuc_ids = src_layer.get('nucleus_ids')
        tgt_nuc_ids = tgt_layer.get('nucleus_ids')
        src_pids = src_layer.get('puncta_id')
        tgt_pids = tgt_layer.get('puncta_id')

        for src_idx, tgt_indices in enumerate(pairs):
            for tgt_idx in tgt_indices:
                # Canonical key: alphabetically smaller layer name first
                if src_name < tgt_name:
                    pair_key = (src_name, src_idx, tgt_name, tgt_idx)
                else:
                    pair_key = (tgt_name, tgt_idx, src_name, src_idx)

                if pair_key in seen:
                    continue
                seen.add(pair_key)

                dist = float(np.linalg.norm(src_world[src_idx] - tgt_world[tgt_idx]))
                src_nid = int(src_nuc_ids[src_idx]) if src_nuc_ids is not None and src_idx < len(src_nuc_ids) else None
                tgt_nid = int(tgt_nuc_ids[tgt_idx]) if tgt_nuc_ids is not None and tgt_idx < len(tgt_nuc_ids) else None

                coloc_rows.append({
                    "Layer_A": src_name,
                    "ID_A": _stable_id(src_pids, src_idx),
                    "Nucleus_ID_A": src_nid,
                    "Z_A": src_world[src_idx][0],
                    "Y_A": src_world[src_idx][1],
                    "X_A": src_world[src_idx][2],
                    "Layer_B": tgt_name,
                    "ID_B": _stable_id(tgt_pids, tgt_idx),
                    "Nucleus_ID_B": tgt_nid,
                    "Z_B": tgt_world[tgt_idx][0],
                    "Y_B": tgt_world[tgt_idx][1],
                    "X_B": tgt_world[tgt_idx][2],
                    "Distance_um": dist,
                    "Coloc_Threshold_um": thresh,
                })

    df_coloc = pd.DataFrame(coloc_rows)
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

        # Greedy triplet selection:
        # 1. For each anchor, find all ChA and ChB puncta within the distance threshold
        # 2. Build candidate triplets (anchor, chA, chB) sorted by total distance
        # 3. Iteratively pick the best (lowest sum distance), mark ChA and ChB as consumed
        # This ensures no punctum participates in more than one triplet, preventing over-counting.
        # Trade-off: May miss globally-optimal matching in favor of greedy speed.
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
            anchor_pids = anchor_layer.get('puncta_id')
            cha_pids = layers_by_name[ch_a_name].get('puncta_id')
            chb_pids = layers_by_name[ch_b_name].get('puncta_id')

            all_hits.append({
                'Anchor_Layer': anchor_name,
                'Anchor_ID': _stable_id(anchor_pids, best_anchor),
                'Nucleus_ID': nuc_id,
                'Anchor_Z': a_coord[0],
                'Anchor_Y': a_coord[1],
                'Anchor_X': a_coord[2],
                'Channel_A': ch_a_name,
                'ChA_ID': _stable_id(cha_pids, best_ch_a),
                'ChA_Z': ca_coord[0],
                'ChA_Y': ca_coord[1],
                'ChA_X': ca_coord[2],
                'ChA_Distance_um': best_ch_a_dist,
                'Channel_B': ch_b_name,
                'ChB_ID': _stable_id(chb_pids, best_ch_b),
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


def _puncta_layer_skip_patterns():
    """Name fragments identifying layers that are not puncta (nuclear stain,
    centroid and consensus layers)."""
    return [
        session.get_nuclear_channel().upper(),
        constants.CENTROIDS_SUFFIX.upper(),
        constants.CONSENSUS_MASKS_NAME.upper(),
    ]


def _iter_puncta_layers_with_ids(points_layers_data):
    """Yield ``(name, nucleus_ids)`` for every puncta layer carrying nucleus IDs."""
    skip = _puncta_layer_skip_patterns()
    for layer in points_layers_data:
        nucleus_ids = layer.get('nucleus_ids')
        if nucleus_ids is None:
            continue
        name = layer['name']
        if any(pat in name.upper() for pat in skip):
            continue
        yield name, np.asarray(nucleus_ids)


def extranuclear_counts(points_layers_data):
    """
    Counts, per channel, the puncta that carry ``Nucleus_ID`` 0.

    These are puncta outside every nucleus in the consensus mask (kept when
    "remove extranuclear" is off) or whose nucleus was deleted in the mask
    editor. They are excluded from the per-nucleus table but stay in the
    Distances and Colocalization sheets, so this count is what makes the
    sheet totals reconcile.

    Returns
    -------
    dict[str, int]
        ``{channel_name: count}`` for every puncta layer with nucleus IDs.
    """
    counts = {}
    for name, nucleus_ids in _iter_puncta_layers_with_ids(points_layers_data):
        counts[name] = int((nucleus_ids.astype(int) == 0).sum())
    return counts


def calculate_per_nucleus_counts(points_layers_data, nucleus_ids=None):
    """
    Counts the number of puncta per nucleus for each channel.

    Filters out nuclear stain, centroid, and consensus layers. Only layers that
    carry a 'nucleus_ids' array are included.

    Parameters
    ----------
    points_layers_data : list[dict]
        Layer dicts with 'name', 'data', and optionally 'nucleus_ids'.
    nucleus_ids : array-like of int, optional
        The nonzero label values present in the consensus mask. When given,
        the table has exactly one row per mask nucleus (zero-count nuclei
        included), and puncta whose ``Nucleus_ID`` is not in the mask are
        dropped with a warning: they carry an assignment from a mask that has
        since been edited. When omitted, rows exist only for nuclei that have
        at least one punctum.

    Returns
    -------
    pd.DataFrame
        Pivot table with Nuclei_ID as rows and channel names as columns.
        Nucleus ID 0 (background) is excluded; see ``extranuclear_counts``.
    """
    records = []
    stale = {}
    valid = None
    if nucleus_ids is not None:
        valid = set(int(n) for n in np.asarray(nucleus_ids).ravel() if int(n) != 0)

    for name, ids in _iter_puncta_layers_with_ids(points_layers_data):
        for nid in ids:
            nid_int = int(nid)
            if nid_int == 0:
                continue
            if valid is not None and nid_int not in valid:
                stale[name] = stale.get(name, 0) + 1
                continue
            records.append({'Nuclei_ID': nid_int, 'Channel': name})

    for name, n in stale.items():
        logger.warning(
            "calculate_per_nucleus_counts[%s]: %d puncta carry a Nucleus_ID that "
            "is not in the consensus mask (mask edited after assignment?); "
            "they are left out of the per-nucleus table.", name, n,
        )

    if not records and valid is None:
        return pd.DataFrame()

    if records:
        df = pd.DataFrame(records)
        pivot = df.groupby(['Nuclei_ID', 'Channel']).size().unstack(fill_value=0)
    else:
        pivot = pd.DataFrame(index=pd.Index([], name='Nuclei_ID'))

    if valid is not None:
        pivot = pivot.reindex(sorted(valid), fill_value=0)
        pivot.index.name = 'Nuclei_ID'
        if pivot.shape[1] == 0:
            return pd.DataFrame()

    pivot.columns.name = None
    pivot = pivot.reset_index().sort_values('Nuclei_ID').reset_index(drop=True)
    return pivot


def calculate_stats(per_nucleus_df, total_nuclei=None, extranuclear=None):
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
    extranuclear : dict[str, int], optional
        Per-channel count of ``Nucleus_ID`` 0 puncta from ``extranuclear_counts``.
        Missing channels count as 0.

    Returns
    -------
    pd.DataFrame
        Rows: one per channel. Columns: Channel, Total_Nuclei, Raw_Sum,
        Extranuclear_Puncta, Total_Puncta, Mean_per_Nucleus,
        StdDev_per_Nucleus, CV_pct. ``Raw_Sum`` counts puncta inside nuclei;
        ``Total_Puncta`` adds the extranuclear ones and equals the number of
        rows that channel contributes as a source in the Distances sheet.
    """
    if per_nucleus_df.empty:
        return pd.DataFrame()

    extranuclear = extranuclear or {}
    channels = [c for c in per_nucleus_df.columns if c != 'Nuclei_ID']
    if total_nuclei is None:
        total_nuclei = int(per_nucleus_df['Nuclei_ID'].nunique())

    if total_nuclei <= 0:
        return pd.DataFrame()

    rows = []
    for ch in channels:
        present = per_nucleus_df[ch].values
        if len(present) > total_nuclei:
            logger.warning(
                "calculate_stats[%s]: per-nucleus dataframe has %d nuclei but "
                "consensus mask only has %d (likely mask edited after puncta "
                "assignment); dividing by mask count.",
                ch, len(present), total_nuclei,
            )

        # Mean and std are reported against the consensus mask's nucleus count.
        # Raw_Sum is the in-nucleus total; Nucleus_ID 0 puncta are reported
        # separately so Total_Puncta accounts for every punctum in the layer.
        raw_sum = int(present.sum())
        extra = int(extranuclear.get(ch, 0))
        if len(present) < total_nuclei:
            all_counts = np.concatenate([present, np.zeros(total_nuclei - len(present))])
        else:
            all_counts = present.astype(float)
        mean = raw_sum / total_nuclei
        # Population std around the reported mean, divisor = total_nuclei
        std = float(np.sqrt(((all_counts - mean) ** 2).sum() / total_nuclei))
        cv = (std / mean * 100) if mean > 0 else 0.0

        rows.append({
            'Channel': ch,
            'Total_Nuclei': total_nuclei,
            'Raw_Sum': raw_sum,
            'Extranuclear_Puncta': extra,
            'Total_Puncta': raw_sum + extra,
            'Mean_per_Nucleus': round(mean, 4),
            'StdDev_per_Nucleus': round(std, 4),
            'CV_pct': round(cv, 2),
        })

    return pd.DataFrame(rows)


DEFAULT_DISTRIBUTION_BINS = (0, 1, 2, 3, 11)
"""Lower edges of the Distribution sheet's bins: 0, 1, 2, 3 to 10, and more than 10 puncta."""


def distribution_bins(edges=None):
    """``[(label, low, high)]`` for the lower ``edges`` of the puncta-count
    bins (``DEFAULT_DISTRIBUTION_BINS`` when None); ``high`` is None for the
    last, open bin. Labels name the sheet's columns: ``Nuclei_0_puncta``,
    ``Nuclei_3to10_puncta``, ``Nuclei_gt10_puncta``."""
    edges = sorted({int(e) for e in (edges if edges is not None else DEFAULT_DISTRIBUTION_BINS)})
    if not edges or edges[0] != 0:
        raise ValueError(f"Distribution bins must start at 0 puncta; got {edges}.")
    bins = []
    for low, nxt in zip(edges, edges[1:]):
        high = nxt - 1
        bins.append((f"Nuclei_{low}_puncta" if low == high else f"Nuclei_{low}to{high}_puncta", low, high))
    bins.append((f"Nuclei_gt{edges[-1] - 1}_puncta", edges[-1], None))
    return bins


def calculate_distribution(per_nucleus_df, total_nuclei=None, bins=None):
    """
    Bins nuclei by puncta count for each channel.

    Parameters
    ----------
    per_nucleus_df : pd.DataFrame
        Pivot table from `calculate_per_nucleus_counts`.
    total_nuclei : int, optional
        Actual number of nuclei in the consensus mask. If None, falls back to
        counting unique IDs in the dataframe.
    bins : sequence of int, optional
        Lower edges of the bins (see ``distribution_bins``); the default gives
        0, 1, 2, 3 to 10 and more than 10 puncta.

    Returns
    -------
    pd.DataFrame
        Rows: one per channel. Columns: Channel, Total_Nuclei, then one per
        bin (``Nuclei_0_puncta`` ... ``Nuclei_gt10_puncta`` by default). The
        bins sum to Total_Nuclei.
    """
    if per_nucleus_df.empty:
        return pd.DataFrame()

    channels = [c for c in per_nucleus_df.columns if c != 'Nuclei_ID']
    if total_nuclei is None:
        total_nuclei = int(per_nucleus_df['Nuclei_ID'].nunique())

    rows = []
    for ch in channels:
        present = per_nucleus_df[ch].values
        if len(present) > total_nuclei:
            logger.warning(
                "calculate_distribution[%s]: per-nucleus dataframe has %d nuclei "
                "but consensus mask only has %d (likely mask edited after puncta "
                "assignment); bins will sum to more than Total_Nuclei.",
                ch, len(present), total_nuclei,
            )
        zeros_count = max(total_nuclei - len(present), 0)

        row = {'Channel': ch, 'Total_Nuclei': int(total_nuclei)}
        for label, low, high in distribution_bins(bins):
            inside = (present >= low) if high is None else ((present >= low) & (present <= high))
            row[label] = int(inside.sum()) + (zeros_count if low == 0 else 0)
        rows.append(row)

    return pd.DataFrame(rows)


# =====================================================================
# Sidecars: the report's tables as CSV
# =====================================================================

DISTANCES_SIDECAR = "distances.csv"
SHEETS_DIR = "report_sheets"


def sheet_file_name(sheet):
    return sheet.replace(" ", "_").replace("/", "_") + ".csv"


def write_report_sidecars(reports_dir, tables):
    """Write each table of the report (``{sheet: DataFrame}``) as CSV: the
    Distances sheet as ``reports/distances.csv``, every sheet also under
    ``reports/report_sheets/``. Queries read these, not the workbook; they
    are rewritten at every export, so they hold the latest report's tables."""
    from . import outputs
    reports_dir = Path(reports_dir)
    for sheet, table in tables.items():
        if table is None or getattr(table, "empty", True):
            continue
        outputs.write_csv(table, reports_dir / SHEETS_DIR / sheet_file_name(sheet))
        if sheet == constants.DISTANCES_SHEET:
            outputs.write_csv(table, reports_dir / DISTANCES_SIDECAR)


DISTANCE_SEMANTICS = (
    "For each punctum of the source channel, the distance to the nearest punctum of the target channel "
    "(nearest neighbour, in µm, in aligned space); a source punctum counts as within d when that nearest "
    "distance is at most d. Pairs are not counted twice, and a target punctum may be the nearest of "
    "several source puncta.")


def query_distances(distances, source, target, max_um, bin_um=None):
    """How many ``source`` puncta have a ``target`` punctum within ``max_um``
    µm, from the Distances table (``distances.csv``), with a histogram of the
    nearest distances up to ``max_um`` in bins of ``bin_um`` µm (one bin when
    None). See ``DISTANCE_SEMANTICS``."""
    rows = distances[(distances["Source_Layer"] == source) & (distances["Target_Layer"] == target)]
    nearest = rows["Distance_um"].to_numpy(float)
    within = nearest <= max_um
    width = bin_um if bin_um and bin_um > 0 else max_um
    n_bins = max(1, int(np.ceil(max_um / width - 1e-9))) if max_um > 0 else 1
    edges = (np.round(np.minimum(np.arange(n_bins + 1) * width, max_um), 9) if max_um > 0
             else np.array([0.0, 0.0]))
    counts = np.histogram(nearest[within], bins=edges)[0] if max_um > 0 else np.array([int(within.sum())])
    return {"source": source, "target": target, "max_um": float(max_um), "n_source": int(len(nearest)),
            "n_within": int(within.sum()), "fraction_within": float(within.mean()) if len(nearest) else 0.0,
            "bins": [[float(lo), float(hi), int(c)] for lo, hi, c in zip(edges[:-1], edges[1:], counts)],
            "semantics": DISTANCE_SEMANTICS}


# =====================================================================
# Orchestrator
# =====================================================================

@timed_stage("colocalization",
             fields=lambda a: {"n_layers": len(a["layers_data"]), "n_rules": len(a["rules"] or []),
                               "n_tri_rules": len(a["tri_rules"] or [])})
def run_colocalization_analysis(layers_data, rules, filename, r1_path, r2_path, output_dir,
                                tri_rules=None, total_nuclei=None, nucleus_ids=None, distribution_bins=None):
    """
    Core Orchestrator for Step 7 & 8.
    Processes puncta distances and exports the master report.

    ``nucleus_ids`` is the list of nonzero label values in the consensus mask.
    When given, the per-nucleus table has a row for every mask nucleus and
    ``total_nuclei`` defaults to its length.
    """
    if nucleus_ids is not None:
        nucleus_ids = np.asarray(nucleus_ids).ravel()
        nucleus_ids = nucleus_ids[nucleus_ids != 0]
        if total_nuclei is None:
            total_nuclei = int(len(nucleus_ids))

    logger.info("Analysis: %d layers, %d rules, %d tri-rules, filename=%s, total_nuclei=%s",
                len(layers_data), len(rules), len(tri_rules or []), filename, total_nuclei)

    # 1. Pairwise nearest-neighbor distances
    df = calculate_distances(layers_data)

    if df.empty:
        logger.warning("No puncta found to analyze.")
        return None

    # 2. Pairwise colocalization (symmetric radius search)
    df_coloc = pd.DataFrame()
    coloc_meta = []
    if rules:
        df_coloc, coloc_meta = calculate_pairwise_colocalization(layers_data, rules)

    # 3. Tri-colocalization (needs raw layer coordinates, not the pairwise df)
    tri_coloc_df = pd.DataFrame()
    tri_coloc_meta = []
    if tri_rules:
        tri_coloc_df, tri_coloc_meta = calculate_tri_colocalization(layers_data, tri_rules)

    # 4. Per-nucleus puncta counts + derived stats
    per_nucleus_df = calculate_per_nucleus_counts(layers_data, nucleus_ids=nucleus_ids)
    extra = extranuclear_counts(layers_data)
    stats_df = calculate_stats(per_nucleus_df, total_nuclei=total_nuclei, extranuclear=extra)
    if distribution_bins is None:
        distribution_bins = session.get_data("distribution_bins")
    distribution_df = calculate_distribution(per_nucleus_df, total_nuclei=total_nuclei, bins=distribution_bins)

    # 5. Build parameters sheet from session puncta_params
    from .puncta import detection_settings
    puncta_params = detection_settings()
    if puncta_params:
        from .stages import detection_method
        params_rows = []
        for channel, params in puncta_params.items():
            row = {'Channel': channel}
            if " - " in channel:
                row['Detected on'] = detection_method(*channel.split(" - ", 1))
            row.update({k: v for k, v in params.items() if k != 'detected_in'})
            params_rows.append(row)
        params_df = pd.DataFrame(params_rows)
    else:
        params_df = pd.DataFrame()

    # 6. Export multi-sheet Excel report
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
        distribution_df=distribution_df,
        params_df=params_df
    )

    # 7. The tables as CSV beside the report, which queries read instead of the workbook.
    if final_path.exists():
        write_report_sidecars(output_dir, {
            constants.DISTANCES_SHEET: df, constants.COLOCALIZATION_SHEET: df_coloc,
            constants.TRI_COLOCALIZATION_SHEET: tri_coloc_df, constants.PER_NUCLEI_SHEET: per_nucleus_df,
            constants.STATS_SHEET: stats_df, constants.DISTRIBUTION_SHEET: distribution_df,
            constants.PARAMETERS_SHEET: params_df})

    # 8. Session Tracking
    if final_path.exists():
        session.set_processed_file(
            layer_name="Master_Analysis_Report",
            path=str(final_path),
            layer_type="report"
        )
        # What the report was made from, so it can be told when it is out of date.
        from .stages import record_report
        record_report(final_path, [layer['name'] for layer in layers_data], output_dir)

    return final_path
