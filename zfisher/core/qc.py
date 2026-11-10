"""Which datasets of a batch need a look, and why (``flag_batch``).

For every dataset folder under a batch root, the quality-control metrics are
read (``metrics.compute_metrics``); the cohort's typical value of each metric
is its median across the datasets, and its spread the median absolute
deviation (scaled to a standard deviation), so one bad dataset does not move
the reference. Deterministic rules then flag a dataset when a metric is far
from the cohort (``robust_z`` beyond a threshold) or past an absolute limit
that holds without a cohort (a registration that fell back to the rough
shift, a channel with no puncta). Each flag says the dataset, what to do and
why, with the value and what was expected; ``write_qc_report`` writes them
as ``qc_report.md`` in the batch root.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .. import constants
from .params import QCThresholds


@dataclass
class Flag:
    dataset: str
    action: str
    reason: str
    metric: str
    value: object = None
    expected: object = None


QC_REPORT = "qc_report.md"


def dataset_folders(batch_root):
    """The dataset folders under ``batch_root``: those with a session file."""
    root = Path(batch_root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / constants.SESSIONS_DIR).exists())


def cohort_metrics(batch_root, include_saturation=False):
    """``{dataset: metrics}`` for every dataset folder under ``batch_root``,
    read without writing anything; a folder whose session cannot be read is
    reported with ``{"error": ...}``."""
    from . import api, metrics, session
    out = {}
    for folder in dataset_folders(batch_root):
        try:
            with session.use(api.open_session(folder)):
                out[folder.name] = metrics.compute_metrics(include_saturation=include_saturation)
        except Exception as exc:
            out[folder.name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _flat(m):
    """The scalar metrics of one dataset that the cohort rules compare."""
    values = {"registration.rmsd_um": m["registration"].get("rmsd_um"),
              "registration.inlier_fraction": m["registration"].get("inlier_fraction"),
              "nuclei.n_nuclei": m["nuclei"].get("n_nuclei"),
              "nuclei.median_volume_um3": (m["nuclei"].get("volume_um3") or {}).get("p50")}
    for key, c in m["channels"].items():
        values[f"{key}.n_puncta"] = c["n_puncta"]
        values[f"{key}.median_snr"] = (c.get("snr") or {}).get("p50")
        values[f"{key}.extranuclear_fraction"] = c["extranuclear_fraction"]
        values[f"{key}.puncta_per_nucleus"] = c.get("puncta_per_nucleus")
    return {k: float(v) for k, v in values.items() if v is not None}


def cohort_priors(cohort):
    """``{metric: (median, robust_sd, n)}`` across the datasets of ``cohort``."""
    columns = {}
    for m in cohort.values():
        if "error" in m:
            continue
        for key, value in _flat(m).items():
            columns.setdefault(key, []).append(value)
    priors = {}
    for key, values in columns.items():
        values = np.asarray(values, dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median))) * 1.4826
        priors[key] = (median, mad, int(len(values)))
    return priors


def readable(cohort):
    """How many datasets of ``cohort`` could be read."""
    return sum(1 for m in cohort.values() if "error" not in m)


def compared_with_batch(cohort, thresholds=None):
    """Whether ``cohort`` has enough readable datasets (``min_cohort``) for
    their metrics to be compared with each other; with fewer, only the fixed
    limits apply."""
    return readable(cohort) >= (thresholds or QCThresholds()).min_cohort


def small_batch_note(cohort, thresholds=None):
    """Why the datasets of ``cohort`` were not compared with each other, or
    None when they were: an empty list of flags then does not mean all is
    well."""
    if compared_with_batch(cohort, thresholds):
        return None
    t = thresholds or QCThresholds()
    return (f"Only {readable(cohort)} dataset(s) could be read; at least {t.min_cohort} are needed to compare "
            "their numbers with each other, so only the fixed limits were checked (a registration that fell "
            "back to the rough shift, few agreeing centroid pairs, a channel with no puncta, saturation). "
            "A dataset unlike the others is not flagged.")


def robust_z(value, prior):
    median, sd, _ = prior
    if sd > 0:
        return (value - median) / sd
    return 0.0 if np.isclose(value, median) else float(np.sign(value - median) * np.inf)


_ACTIONS = {
    "registration": "Check the registration (Alignment & Consensus)",
    "nuclei": "Check the nuclei segmentation",
    "n_puncta": "Check the detection settings of this channel",
    "median_snr": "Check the image quality or detection of this channel",
    "extranuclear_fraction": "Check the consensus mask against this channel's puncta",
    "puncta_per_nucleus": "Check the detection settings of this channel",
}


def flag_batch(cohort, thresholds=None):
    """The flags of every dataset of ``cohort`` (``cohort_metrics``) under
    ``thresholds`` (``params.QCThresholds``); see the module docstring."""
    t = thresholds or QCThresholds()
    priors = cohort_priors(cohort)
    compare = compared_with_batch(cohort, t)
    flags = []
    for name, m in cohort.items():
        if "error" in m:
            flags.append(Flag(name, "Open the dataset and check it", m["error"], "session"))
            continue
        reg = m["registration"]
        if reg.get("fallback_used"):
            flags.append(Flag(name, _ACTIONS["registration"],
                              f"RANSAC was not used ({reg.get('fallback_reason')}); the rough shift was kept",
                              "registration.method", reg.get("method"), "ransac"))
        if reg.get("inlier_fraction") is not None and reg["inlier_fraction"] < t.min_inlier_fraction:
            flags.append(Flag(name, _ACTIONS["registration"], "few centroid pairs agree with the shift",
                              "registration.inlier_fraction", reg["inlier_fraction"], f">= {t.min_inlier_fraction}"))
        for key, c in m["channels"].items():
            if c["n_puncta"] == 0:
                flags.append(Flag(name, _ACTIONS["n_puncta"], f"{key} has no puncta", f"{key}.n_puncta", 0, "> 0"))
            if key.startswith("R2 - ") and c.get("detected_on") == "aligned":
                flags.append(Flag(name, "Re-detect on the original image to match batch",
                                  f"{key} was detected on the resampled R2 image", f"{key}.detected_on",
                                  "aligned", "raw"))
        for image, fraction in (m.get("saturation") or {}).items():
            if fraction > t.max_saturated_fraction:
                flags.append(Flag(name, "Check the acquisition settings", f"{image} is saturated in "
                                  f"{100 * fraction:.2f}% of voxels", f"{image}.saturation", fraction,
                                  f"<= {t.max_saturated_fraction}"))
        if not compare:
            continue
        for key, value in _flat(m).items():
            prior = priors.get(key)
            if prior is None or prior[2] < t.min_cohort:
                continue
            z = robust_z(value, prior)
            if abs(z) >= t.z_threshold:
                kind = key.rsplit(".", 1)[-1] if not key.startswith(("registration", "nuclei")) else key.split(".")[0]
                direction = "high" if z > 0 else "low"
                flags.append(Flag(name, _ACTIONS.get(kind, "Check this dataset"),
                                  f"{key} is {direction} for this batch ({value:.3g}; batch median {prior[0]:.3g})",
                                  key, value, prior[0]))
    return flags


def write_qc_report(batch_root, cohort, flags, thresholds=None):
    """``qc_report.md`` in ``batch_root``: the flags, then each metric's batch median and spread."""
    from . import outputs
    t = thresholds or QCThresholds()
    priors = cohort_priors(cohort)
    lines = ["# zFISHer batch QC", "", f"{len(cohort)} datasets, {len({f.dataset for f in flags})} flagged. "
             f"Flagged when a metric is {t.z_threshold:g} robust standard deviations from the batch median "
             f"(with at least {t.min_cohort} datasets), or past an absolute limit.", ""]
    note = small_batch_note(cohort, t)
    if note:
        lines += [f"**The datasets were not compared with each other.** {note}", ""]
    if flags:
        lines += ["## Flags", "", "| dataset | action | reason |", "|---|---|---|"]
        lines += [f"| {f.dataset} | {f.action} | {f.reason} |" for f in flags]
    else:
        lines += ["No dataset was flagged."]
    lines += ["", "## Batch medians", "", "| metric | median | robust SD | datasets |", "|---|---|---|---|"]
    lines += [f"| {k} | {m:.4g} | {sd:.3g} | {n} |" for k, (m, sd, n) in sorted(priors.items())]
    return outputs.write_text(Path(batch_root) / QC_REPORT, "\n".join(lines) + "\n")


def flags_as_dicts(flags):
    return [asdict(f) for f in flags]
