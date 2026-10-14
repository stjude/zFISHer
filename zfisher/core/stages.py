"""The steps one channel of a finished session can be run through again.

``detect_channel`` detects a channel's puncta on its raw image, read from disk,
and writes the channel's raw-space CSV; ``transform_channel`` takes that CSV
through the registration saved by canvas generation into aligned space and
writes the aligned CSV. Together they re-detect one channel of a reopened
session, batch or GUI, without registering again: detect on the original image,
then warp, the method batch uses.

Both work on the session's files, not on viewer layers, and take no UI; the GUI
and batch call the same functions. Detection settings, the mask, the image and
the registration used are recorded in ``provenance.jsonl`` like every other
puncta file.
"""
import logging
from pathlib import Path

import numpy as np

from .. import constants
from . import io, provenance, puncta, registration, session

logger = logging.getLogger(__name__)

# How the puncta already in a channel's CSV meet a new detection; the same
# choices as the GUI's replace-or-merge question.
EXISTING_ACTIONS = ("replace", "replace_all", "merge")


def raw_puncta_name(round_id, channel):
    """``'R2 - FITC_puncta'``: the session name of a channel's raw puncta."""
    return f"{round_id} - {channel}{constants.PUNCTA_SUFFIX}"


def aligned_puncta_name(round_id, channel, saved):
    """The aligned puncta layer of a channel under the registration ``saved``:
    ``'Warped R2 - FITC_puncta'`` when R2 is warped by a B-spline, otherwise
    ``'Aligned <round> - FITC_puncta'``, as canvas generation names them."""
    warped = round_id == "R2" and saved.bspline is not None
    prefix = constants.WARPED_PREFIX if warped else constants.ALIGNED_PREFIX
    return f"{prefix} {round_id} - {channel}{constants.PUNCTA_SUFFIX}"


def _reports_dir():
    out = session.get_data("output_dir")
    if not out:
        raise RuntimeError("No session is open.")
    path = Path(out) / constants.REPORTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def round_mask_path(round_id):
    """The raw-space nuclei mask of ``round_id`` registered in the session
    (``'R2 - DAPI_masks'``), or None. Looked up by registration rather than by
    name, because the mask carries the nuclear channel of the round it was
    saved under."""
    for name, info in (session.get_data("processed_files", default={}) or {}).items():
        if (isinstance(info, dict) and info.get("subtype") == "mask" and name.startswith(f"{round_id} - ")
                and name.endswith(constants.MASKS_SUFFIX) and Path(info.get("path", "")).exists()):
            return Path(info["path"])
    return None


def recorded_detection_parameters(round_id, channel):
    """The full detection parameters last recorded for the channel's raw CSV,
    or None."""
    csv = constants.puncta_csv_path(_reports_dir(), raw_puncta_name(round_id, channel))
    record = provenance.latest(csv) if csv.exists() else None
    return dict(record["detection_parameters"]) if record and record.get("detection_parameters") else None


def _combine(existing, coords, rows, action, voxels, name):
    """The rows of a channel's CSV after a new detection: ``existing`` (the
    current table, or None) and the new ``coords`` and ``rows`` (Nucleus_ID,
    Intensity, SNR) met as ``action`` says. New rows within
    ``PUNCTA_DUPLICATE_DISTANCE_UM`` of a kept punctum are dropped; new rows,
    and kept ones without an id, get ids the channel has never had."""
    import pandas as pd
    from scipy.spatial import cKDTree

    new = pd.DataFrame({"Z": coords[:, 0], "Y": coords[:, 1], "X": coords[:, 2],
                        "Nucleus_ID": rows[:, 0], "Intensity": rows[:, 1], "SNR": rows[:, 2],
                        "Source": constants.PUNCTA_SOURCE_AUTO})
    old_ids = existing["puncta_id"] if existing is not None and "puncta_id" in existing.columns else ()
    if existing is None or action == "replace_all":
        kept = new.iloc[:0]
    elif action == "replace":
        kept = existing[existing["Source"] == constants.PUNCTA_SOURCE_MANUAL] if "Source" in existing.columns \
            else existing.iloc[:0]
    else:
        kept = existing
    kept = kept.reset_index(drop=True)
    if len(kept) and len(new):
        scale = np.asarray(voxels, dtype=float)
        dists, _ = cKDTree(kept[["Z", "Y", "X"]].to_numpy(float) * scale).query(coords * scale)
        new = new[dists > constants.PUNCTA_DUPLICATE_DISTANCE_UM].reset_index(drop=True)
    kept_ids = (pd.to_numeric(kept["puncta_id"], errors="coerce") if "puncta_id" in kept.columns
                else pd.Series(np.nan, index=kept.index))
    missing = kept_ids.isna().to_numpy()
    ids = puncta.allocate_puncta_ids(name, int(missing.sum()) + len(new), old_ids)
    kept_ids[missing] = ids[:missing.sum()]
    kept = kept.assign(puncta_id=kept_ids.astype(np.int64))
    new["puncta_id"] = ids[missing.sum():]
    return pd.concat([kept, new], ignore_index=True)


def detect_channel(round_id, channel, params=None, existing="replace", chooser=None, confirm=None,
                   run_by="gui", progress_callback=None):
    """Detect ``channel``'s puncta on the raw image of ``round_id`` and write
    the channel's raw-space CSV (``reports/R2_-_FITC_puncta.csv``). Returns its
    path.

    ``params`` are ``puncta.process_puncta_detection``'s parameters; None means
    the ones recorded for this CSV last time. With ``nuclei_only`` (the
    default) puncta outside the round's own nuclei mask are dropped.
    ``existing`` says what happens to the puncta already in the CSV:
    ``'replace'`` keeps those placed by hand, ``'replace_all'`` keeps none,
    ``'merge'`` keeps all. The raw image is found with ``io.resolve_raw_path``
    (``chooser`` and ``confirm`` as there; without a chooser a missing image
    raises ``io.RawImageNotFound``) and only this channel is read.
    """
    import pandas as pd

    if existing not in EXISTING_ACTIONS:
        raise ValueError(f"existing must be one of {EXISTING_ACTIONS}, not {existing!r}")
    if params is None:
        params = recorded_detection_parameters(round_id, channel)
        if params is None:
            raise LookupError(f"No detection settings are recorded for {round_id} {channel}; give them.")
    params = dict(params)
    nuclei_only = params.get("nuclei_only", True)
    name = raw_puncta_name(round_id, channel)
    csv = constants.puncta_csv_path(_reports_dir(), name)

    image_path = io.resolve_raw_path(round_id, chooser=chooser, confirm=confirm)
    if progress_callback: progress_callback(5, f"Reading {round_id} {channel}...")
    image = io.read_channel(image_path, channel)
    voxels = io.image_identity(image_path)["voxels"]
    mask_path = None
    mask = None
    if nuclei_only:
        mask_path = round_mask_path(round_id)
        if mask_path is None:
            raise FileNotFoundError(f"{round_id} has no nuclei mask, which detecting only inside nuclei "
                                    f"needs. Segment {round_id}'s nuclei, or detect everywhere.")
        mask = io.read_label_tif(mask_path)

    def detection_progress(p, text):
        if progress_callback and p >= 0:
            progress_callback(10 + int(p * 0.8), text)
    found = puncta.process_puncta_detection(image, mask_data=mask, voxels=voxels, params=params,
                                            layer_name=name, progress_callback=detection_progress)
    del image, mask

    current = pd.read_csv(csv, float_precision="round_trip") if csv.exists() else None
    table = _combine(current, found[:, :3], found[:, 3:6], existing, voxels, name)
    puncta.write_puncta_csv(csv, table[["Z", "Y", "X"]].to_numpy(float), table.drop(columns=["Z", "Y", "X"]))
    session.set_processed_file(name, str(csv), "points", metadata={
        "format": "csv", "subtype": "puncta_csv", "space": "raw", "round": round_id})
    puncta.record_puncta(
        csv, "detected", round=round_id, channel=channel, space=f"raw:{round_id}", detected_in="raw",
        source_image=provenance.file_identity(image_path),
        mask=provenance.file_identity(mask_path) if mask_path is not None else "none",
        detection_parameters=params, voxel_size=[float(v) for v in voxels],
        existing_action=existing if current is not None else None, run_by=run_by)
    puncta.record_detection_settings(name, puncta.detection_settings_row(params, len(found), "raw"))
    logger.info("Detected %d puncta in %s %s; %d rows in %s after %s", len(found), round_id, channel,
                len(table), csv.name, existing)
    if progress_callback: progress_callback(100, f"{len(table)} puncta in {round_id} {channel}.")
    return csv


def _numeric(table, column, default):
    """``table[column]`` as floats, ``default`` where missing or not a number."""
    import pandas as pd
    if column not in table.columns:
        return np.full(len(table), default, dtype=float)
    return pd.to_numeric(table[column], errors="coerce").fillna(default).to_numpy(float)


def _aligned_only_manual(aligned_csv, raw_csv):
    """The rows of ``aligned_csv`` placed by hand on the aligned layer: manual,
    and not the transform of a manual row of ``raw_csv`` (same ``puncta_id``).
    None if there is no aligned CSV yet."""
    import pandas as pd
    if not aligned_csv.exists():
        return None
    old = pd.read_csv(aligned_csv, float_precision="round_trip")
    if "Source" not in old.columns:
        return old.iloc[:0]
    raw = pd.read_csv(raw_csv)
    raw_manual = (_numeric(raw, "puncta_id", np.nan)[(raw["Source"] == constants.PUNCTA_SOURCE_MANUAL).to_numpy()]
                  if "Source" in raw.columns else np.empty(0))
    manual = (old["Source"] == constants.PUNCTA_SOURCE_MANUAL).to_numpy()
    return old[manual & ~np.isin(_numeric(old, "puncta_id", np.nan), raw_manual)].reset_index(drop=True)


def transform_channel(round_id, channel, remove_extranuclear=None, keep_aligned_manual=True, run_by="gui"):
    """Take the channel's raw-space CSV into aligned space with the
    registration saved by canvas generation, assign nuclei from the consensus
    mask, and write the channel's aligned CSV. Returns its path.

    ``remove_extranuclear`` drops puncta outside every consensus nucleus; None
    means as the channel's previous aligned CSV was made (batch records it; the
    GUI canvas step kept them). With ``keep_aligned_manual``, puncta placed by
    hand on the aligned layer, which have no raw-space row, are carried over
    unchanged, at the same coordinates. Raises ``LookupError`` for a session
    with no saved registration (made before 2.1.0) and ``FileNotFoundError``
    without a raw CSV.
    """
    import pandas as pd

    saved = registration.load_registration()
    reports = _reports_dir()
    raw_csv = constants.puncta_csv_path(reports, raw_puncta_name(round_id, channel))
    if not raw_csv.exists():
        raise FileNotFoundError(f"{round_id} {channel} has no raw-space puncta ({raw_csv.name}); "
                                f"detect them on the raw image first.")
    name = aligned_puncta_name(round_id, channel, saved)
    aligned_csv = constants.puncta_csv_path(reports, name)
    previous = provenance.latest(aligned_csv) if aligned_csv.exists() else None
    if remove_extranuclear is None:
        remove_extranuclear = bool((previous or {}).get("remove_extranuclear", False))
    consensus_path = io.find_mask(Path(session.get_data("output_dir")) / constants.SEGMENTATION_DIR,
                                  constants.CONSENSUS_MASKS_NAME)
    consensus = io.read_label_tif(consensus_path) if consensus_path is not None else None

    data, sources = puncta.transform_puncta_csv(raw_csv, round_id, saved, consensus_mask=consensus,
                                                remove_extranuclear=remove_extranuclear)
    del consensus
    carried = _aligned_only_manual(aligned_csv, raw_csv) if keep_aligned_manual else None
    if carried is not None and len(carried):
        # An id the new rows also use (or none) is replaced, so ids stay unique.
        ids = _numeric(carried, "puncta_id", np.nan)
        clash = np.isnan(ids) | np.isin(ids, data[:, 6]) | pd.Series(ids).duplicated().to_numpy()
        if clash.any():
            ids[clash] = puncta.allocate_puncta_ids(name, int(clash.sum()), list(data[:, 6]) + list(ids[~clash]))
        rows = np.column_stack([carried[["Z", "Y", "X"]].to_numpy(float), _numeric(carried, "Nucleus_ID", 0),
                                _numeric(carried, "Intensity", 0), _numeric(carried, "SNR", 0), ids])
        data = np.vstack([data.reshape(-1, 7), rows])
        sources = np.concatenate([sources, np.full(len(carried), constants.PUNCTA_SOURCE_MANUAL, dtype=object)])
    n_carried = 0 if carried is None else len(carried)
    puncta._save_puncta_csv(data.reshape(-1, 7), aligned_csv, name, sources=sources)
    puncta.record_puncta(
        aligned_csv, "transformed", round=round_id, channel=channel, space="aligned", detected_in="raw",
        derived_from=raw_csv.name, registration=registration.registration_fingerprint(),
        consensus_mask=provenance.file_identity(consensus_path) if consensus_path is not None else "none",
        remove_extranuclear=remove_extranuclear, n_carried_manual=n_carried, run_by=run_by)
    logger.info("Transformed %s %s into %s: %d puncta, %d placed by hand on the aligned layer kept",
                round_id, channel, aligned_csv.name, len(data), n_carried)
    return aligned_csv
