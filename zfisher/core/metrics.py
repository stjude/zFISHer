"""Quality-control numbers for one session, computed from its files.

``compute_metrics`` reads the session's registration record, its aligned
puncta CSVs, its consensus mask and, if asked, its aligned images, and
returns the numbers a person (or ``flag_batch``) looks at to decide whether
a dataset needs attention: how well the rounds registered, how strong and how
many the puncta are per channel, how many fall outside nuclei or were placed
by hand, how the nuclei are sized, how much of each image is saturated, and
how each channel was detected.
"""
import logging
from pathlib import Path

import numpy as np

from .. import constants
from . import io, puncta, session

logger = logging.getLogger(__name__)

LOW_SNR = 2.0
"""A punctum with an SNR below this counts towards ``low_snr_fraction``."""

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}
    return {f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in QUANTILES}


def registration_metrics():
    """Rigid fit of the rounds: method, RMSD (µm when recorded), pairs, inliers."""
    rigid = (session.get_data("registration", default={}) or {}).get("rigid") or {}
    return {"method": rigid.get("method"), "fallback_used": rigid.get("fallback_used"),
            "fallback_reason": rigid.get("fallback_reason"), "rmsd_um": rigid.get("rmsd_um"),
            "rmsd_px": rigid.get("rmsd"), "n_pairs": rigid.get("n_pairs"), "n_inliers": rigid.get("n_inliers"),
            "inlier_fraction": rigid.get("inlier_fraction"), "shift": rigid.get("shift")}


def channel_metrics(mask=None):
    """Per aligned puncta channel: counts, SNR quantiles and low tail, the
    fractions outside nuclei (from the consensus ``mask`` when given, else the
    CSV's ``Nucleus_ID``) and placed by hand, and how it was detected."""
    from .stages import detected_in
    out = {}
    reports = Path(session.get_data("output_dir")) / constants.REPORTS_DIR
    for csv in puncta.aligned_puncta_csvs(reports):
        import pandas as pd
        table = pd.read_csv(csv, float_precision="round_trip")
        name = None
        for layer in (session.get_data("processed_files", default={}) or {}):
            if constants.puncta_csv_path(reports, layer).name == csv.name:
                name = layer
        key = puncta.channel_key(name or csv.stem) or csv.stem
        n = len(table)
        if mask is not None and n:
            nuclei = puncta.nucleus_ids_from_mask(table[["Z", "Y", "X"]].to_numpy(float), mask)
        else:
            nuclei = pd.to_numeric(table.get("Nucleus_ID", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
        snr = pd.to_numeric(table.get("SNR", pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        source = table["Source"].astype(str) if "Source" in table else pd.Series(["unknown"] * n)
        round_id, _, channel = key.partition(" - ")
        out[key] = {
            "n_puncta": int(n),
            "snr": _quantiles(snr),
            "low_snr_fraction": float(np.mean(snr < LOW_SNR)) if n else 0.0,
            "extranuclear_fraction": float(np.mean(np.asarray(nuclei) <= 0)) if n else 0.0,
            "manual_fraction": float(np.mean(source == constants.PUNCTA_SOURCE_MANUAL)) if n else 0.0,
            "puncta_per_nucleus": None,
            "detected_on": detected_in(round_id, channel)[0] if channel else None,
        }
    return out


def nucleus_metrics(mask):
    """Count and volume quantiles (µm³) of the consensus nuclei."""
    scale = session.get_data("canvas_scale") or (1.0, 1.0, 1.0)
    voxel_um3 = float(np.prod(scale))
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    return {"n_nuclei": int(len(labels)), "volume_um3": _quantiles(counts * voxel_um3)}


def saturation_metrics():
    """Per aligned channel image, the fraction of voxels at the type's maximum."""
    import tifffile
    from .cells import aligned_channels
    out = {}
    for name, path in aligned_channels():
        data = tifffile.imread(path)
        if np.issubdtype(data.dtype, np.integer):
            out[name] = float(np.mean(data == np.iinfo(data.dtype).max))
        del data
    return out


def compute_metrics(include_saturation=False):
    """Every metric of the open session (see the module docstring). Reading the
    aligned images for ``include_saturation`` is the slow part."""
    from .stages import _consensus_path
    consensus = _consensus_path()
    mask = io.read_label_tif(consensus) if consensus is not None else None
    channels = channel_metrics(mask)
    nuclei = nucleus_metrics(mask) if mask is not None else {"n_nuclei": 0, "volume_um3": {}}
    for values in channels.values():
        if nuclei["n_nuclei"]:
            values["puncta_per_nucleus"] = values["n_puncta"] * (1 - values["extranuclear_fraction"]) / nuclei["n_nuclei"]
    detected = [v["detected_on"] for v in channels.values() if v.get("detected_on")]
    return {
        "dataset": Path(session.get_data("output_dir")).name,
        "registration": registration_metrics(),
        "channels": channels,
        "nuclei": nuclei,
        "saturation": saturation_metrics() if include_saturation else None,
        "detection_mix": {d: detected.count(d) for d in sorted(set(detected))},
    }
