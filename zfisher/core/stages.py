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

``status`` reads those records back and says which aligned puncta CSVs and
which report are out of date, missing, or edited by hand since they were made.
"""
import logging
from dataclasses import dataclass, field
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


def _in_output_folder(subdir, stored):
    """``stored`` (a registered path) as the file of that name in the open
    session's ``subdir``, when there is one. A session opened from a copied
    output folder can still have paths into the folder it was copied from."""
    out = session.get_data("output_dir")
    if out and stored:
        local = Path(out) / subdir / Path(stored).name
        if local.exists():
            return local
    return Path(stored) if stored and Path(stored).exists() else None


def round_mask_path(round_id):
    """The raw-space nuclei mask of ``round_id`` registered in the session
    (``'R2 - DAPI_masks'``), or None. Looked up by registration rather than by
    name, because the mask carries the nuclear channel of the round it was
    saved under."""
    for name, info in (session.get_data("processed_files", default={}) or {}).items():
        if (isinstance(info, dict) and info.get("subtype") == "mask" and name.startswith(f"{round_id} - ")
                and name.endswith(constants.MASKS_SUFFIX)):
            path = _in_output_folder(constants.SEGMENTATION_DIR, info.get("path"))
            if path is not None:
                return path
    return None


def recorded_detection_parameters(round_id, channel):
    """The full detection parameters last recorded for the channel's raw CSV,
    or None."""
    csv = constants.puncta_csv_path(_reports_dir(), raw_puncta_name(round_id, channel))
    record = provenance.latest(csv) if csv.exists() else None
    return dict(record["detection_parameters"]) if record and record.get("detection_parameters") else None


def _combine(existing, found, action, voxels, name):
    """The rows of a channel's CSV after a new detection: ``existing`` (the
    current table, or None) and ``found`` (``process_puncta_detection``'s
    rows) met as ``action`` says. New rows within
    ``PUNCTA_DUPLICATE_DISTANCE_UM`` of a kept punctum are dropped.

    New rows keep the ids detection gave them, shifted above every id the
    channel has had, so a first detection is numbered as detection numbers it
    and a later one reuses none; kept rows without an id get new ones."""
    import pandas as pd
    from scipy.spatial import cKDTree

    found = np.asarray(found, dtype=float).reshape(-1, 7)
    coords = found[:, :3]
    new = pd.DataFrame({"Z": coords[:, 0], "Y": coords[:, 1], "X": coords[:, 2],
                        "Nucleus_ID": found[:, 3], "Intensity": found[:, 4], "SNR": found[:, 5],
                        "Source": constants.PUNCTA_SOURCE_AUTO, "_detected": found[:, 6]})
    old_ids = existing["puncta_id"] if existing is not None and "puncta_id" in existing.columns else ()
    if existing is None or action == "replace_all":
        kept = new.iloc[:0]
    elif action == "replace":
        kept = existing[existing["Source"] == constants.PUNCTA_SOURCE_MANUAL] if "Source" in existing.columns \
            else existing.iloc[:0]
    else:
        kept = existing
    kept = kept.drop(columns="_detected", errors="ignore").reset_index(drop=True)
    if len(kept) and len(new):
        scale = np.asarray(voxels, dtype=float)
        dists, _ = cKDTree(kept[["Z", "Y", "X"]].to_numpy(float) * scale).query(coords * scale)
        new = new[dists > constants.PUNCTA_DUPLICATE_DISTANCE_UM].reset_index(drop=True)
    kept_ids = (pd.to_numeric(kept["puncta_id"], errors="coerce") if "puncta_id" in kept.columns
                else pd.Series(np.nan, index=kept.index))
    missing = kept_ids.isna().to_numpy()
    span = int(found[:, 6].max()) + 1 if len(found) else 0
    ids = puncta.allocate_puncta_ids(name, int(missing.sum()) + span, old_ids)
    kept_ids[missing] = ids[:missing.sum()]
    kept = kept.assign(puncta_id=kept_ids.astype(np.int64))
    start = int(ids[missing.sum()]) if span else 0
    new["puncta_id"] = (start + new.pop("_detected")).astype(np.int64)
    return pd.concat([kept, new], ignore_index=True)


def detect_channel(round_id, channel, params=None, existing="replace", chooser=None, confirm=None,
                   run_by="gui", progress_callback=None, image=None, image_path=None, voxels=None, mask=None):
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

    Batch, which holds the round already, passes the channel as ``image``,
    with ``image_path`` and ``voxels`` and the round's nuclei ``mask``, so
    nothing is read twice. Returns None, writing nothing, when nothing is
    found and the channel has no CSV to replace.
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

    if image_path is None:
        image_path = io.resolve_raw_path(round_id, chooser=chooser, confirm=confirm)
    if image is None:
        if progress_callback: progress_callback(5, f"Reading {round_id} {channel}...")
        image = io.read_channel(image_path, channel)
    if voxels is None:
        voxels = io.image_identity(image_path)["voxels"]
    mask_path = None
    if nuclei_only:
        mask_path = round_mask_path(round_id)
        if mask_path is None:
            raise FileNotFoundError(f"{round_id} has no nuclei mask, which detecting only inside nuclei "
                                    f"needs. Segment {round_id}'s nuclei, or detect everywhere.")
        if mask is None:
            mask = io.read_label_tif(mask_path)
    else:
        mask = None

    def detection_progress(p, text):
        if progress_callback and p >= 0:
            progress_callback(10 + int(p * 0.8), text)
    found = puncta.process_puncta_detection(image, mask_data=mask, voxels=voxels, params=params,
                                            layer_name=name, progress_callback=detection_progress)
    del image, mask

    current = pd.read_csv(csv, float_precision="round_trip") if csv.exists() else None
    if len(found) == 0 and current is None:
        puncta.record_detection_settings(name, puncta.detection_settings_row(params, 0, "raw"))
        logger.info("No puncta found in %s %s; nothing written", round_id, channel)
        if progress_callback: progress_callback(100, f"No puncta in {round_id} {channel}.")
        return None
    table = _combine(current, found, existing, voxels, name)
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


def transform_channel(round_id, channel, remove_extranuclear=None, keep_aligned_manual=True, run_by="gui",
                      consensus_mask=None):
    """Take the channel's raw-space CSV into aligned space with the
    registration saved by canvas generation, assign nuclei from the consensus
    mask, and write the channel's aligned CSV. Returns its path.

    ``remove_extranuclear`` drops puncta outside every consensus nucleus; None
    means as the channel's previous aligned CSV was made (batch records it; the
    GUI canvas step kept them). With ``keep_aligned_manual``, puncta placed by
    hand on the aligned layer, which have no raw-space row, are carried over
    unchanged, at the same coordinates. Raises ``LookupError`` for a session
    with no saved registration (made before 2.1.0) and ``FileNotFoundError``
    without a raw CSV. Batch passes the ``consensus_mask`` it holds. Returns
    None, writing nothing, when no punctum reaches aligned space and the
    channel has no aligned CSV to replace.
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
    consensus = consensus_mask
    if consensus is None and consensus_path is not None:
        consensus = io.read_label_tif(consensus_path)

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
    if len(data) == 0 and not aligned_csv.exists():
        logger.info("No %s %s puncta reach aligned space; nothing written", round_id, channel)
        return None
    puncta._save_puncta_csv(data.reshape(-1, 7), aligned_csv, name, sources=sources)
    puncta.record_puncta(
        aligned_csv, "transformed", round=round_id, channel=channel, space="aligned", detected_in="raw",
        derived_from=raw_csv.name, derived_from_sha256=provenance.file_sha256(raw_csv),
        registration=registration.registration_fingerprint(),
        consensus_mask=provenance.file_identity(consensus_path) if consensus_path is not None else "none",
        remove_extranuclear=remove_extranuclear, n_carried_manual=n_carried,
        points_sha256=puncta.points_fingerprint(aligned_csv), run_by=run_by)
    logger.info("Transformed %s %s into %s: %d puncta, %d placed by hand on the aligned layer kept",
                round_id, channel, aligned_csv.name, len(data), n_carried)
    return aligned_csv


# --- What is out of date ------------------------------------------------------

REPORT_NAME = "Master_Analysis_Report"
FRESH, STALE, MISSING, UNKNOWN = "fresh", "stale", "missing", "unknown"
DETECTED_ON_ALIGNED = "detected_on_aligned"


@dataclass
class ArtifactStatus:
    """Whether one output of the session is up to date.

    ``state`` is ``fresh``; ``stale`` (something it was made from changed since);
    ``missing`` (it should exist and does not); ``detected_on_aligned`` (an
    aligned puncta CSV last made by detecting on the aligned image, not from
    the raw CSV, so warping the raw CSV again would replace it); or ``unknown``
    (made before zFISHer recorded what it was made from). ``edited`` says the
    puncta in an aligned CSV were changed by hand since it was made, which a
    re-warp would discard. ``reasons`` explain the state in words."""
    kind: str                 # "aligned_puncta" or "report"
    name: str                 # layer name, or the report's file name
    path: object              # Path, or None when missing
    state: str
    edited: bool = False
    reasons: list = field(default_factory=list)
    round: str = None
    channel: str = None


def record_report(path, layer_names, reports_dir):
    """Record the report just written at ``path`` from the puncta layers
    ``layer_names``: each layer's CSV with ``puncta.points_fingerprint``, and
    the consensus mask."""
    inputs = {}
    for name in layer_names:
        csv = constants.puncta_csv_path(reports_dir, name)
        if csv.exists():
            inputs[csv.name] = puncta.points_fingerprint(csv, with_source=False)
    consensus = _consensus_path()
    provenance.record_artifact(path, "exported", kind="report", inputs=inputs,
                               consensus_mask=provenance.file_identity(consensus) if consensus else "none")


def _consensus_path():
    out = session.get_data("output_dir")
    return io.find_mask(Path(out) / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME) if out else None


def _sha(path):
    return provenance.file_sha256(path) if path is not None and Path(path).exists() else None


def _aligned_status(round_id, channel, raw_csv, saved, consensus_sha):
    name = aligned_puncta_name(round_id, channel, saved)
    path = constants.puncta_csv_path(raw_csv.parent, name)
    out = ArtifactStatus("aligned_puncta", name, path, FRESH, round=round_id, channel=channel)
    if not path.exists():
        out.path, out.state = None, MISSING
        out.reasons.append(f"{raw_csv.name} has not been warped into aligned space")
        return out
    history = provenance.records(path)
    made = [r for r in history if r.get("event") in ("transformed", "detected")]
    if not made:
        out.state = UNKNOWN
        out.reasons.append("no record of how it was made")
        return out
    last = made[-1]
    if last["event"] == "detected":
        out.state = DETECTED_ON_ALIGNED
        out.reasons.append("detected on the aligned image, not warped from the raw puncta")
        return out
    # Edited by hand since it was made: its puncta differ from what was written.
    if last.get("points_sha256"):
        out.edited = puncta.points_fingerprint(path) != last["points_sha256"]
    else:                     # records from before fingerprints: compare counts by Source
        out.edited = (history[-1].get("n_by_source") or {}) != (last.get("n_by_source") or {})
    if out.edited:
        out.reasons.append("puncta were added, moved or deleted by hand since it was made")
    # Stale: what it was made from has changed.
    stale = []
    if last.get("derived_from_sha256"):
        if _sha(raw_csv) != last["derived_from_sha256"]:
            stale.append(f"{raw_csv.name} changed since it was warped")
    else:                     # records from before the raw CSV's hash was kept
        raw_made = [r for r in provenance.records(raw_csv) if r.get("event") == "detected"]
        if raw_made and raw_made[-1].get("time", "") > last.get("time", ""):
            stale.append(f"{raw_csv.name} was detected again since it was warped")
    if last.get("registration") and last["registration"] != registration.registration_fingerprint():
        stale.append("the registration changed since it was warped")
    made_with = last.get("consensus_mask")
    if last.get("remove_extranuclear") and isinstance(made_with, dict) and made_with.get("sha256") != consensus_sha:
        stale.append("the consensus mask changed, and puncta outside nuclei were removed with it")
    out.reasons += stale
    if stale:
        out.state = STALE
    return out


def _report_status(aligned, consensus_sha):
    info = (session.get_data("processed_files", default={}) or {}).get(REPORT_NAME)
    path = _in_output_folder(constants.REPORTS_DIR, info.get("path")) if isinstance(info, dict) else None
    if path is None:
        return ArtifactStatus("report", REPORT_NAME, None, MISSING, reasons=["no report has been exported"])
    out = ArtifactStatus("report", path.name, path, FRESH)
    record = provenance.latest(path)
    if not record or record.get("kind") != "report":
        out.state = UNKNOWN
        out.reasons.append("exported before zFISHer recorded what a report was made from")
        return out
    inputs = record.get("inputs") or {}
    for csv_name, fingerprint in inputs.items():
        csv = path.parent / csv_name
        if not csv.exists():
            out.reasons.append(f"{csv_name}, which it used, is gone")
        elif puncta.points_fingerprint(csv, with_source=False) != fingerprint:
            out.reasons.append(f"the puncta in {csv_name} changed since it was exported")
    for item in aligned:
        if item.path is not None and item.path.name not in inputs:
            out.reasons.append(f"{item.path.name} is not in it")
        if item.state in (STALE, MISSING):
            out.reasons.append(f"{item.name} is {item.state}")
    made_with = record.get("consensus_mask")
    if isinstance(made_with, dict) and made_with.get("sha256") != consensus_sha:
        out.reasons.append("the consensus mask changed since it was exported")
    if out.reasons:
        out.state = STALE
    return out


def status():
    """What is out of date in the open session: one ``ArtifactStatus`` per
    channel with raw-space puncta (its aligned CSV), then one for the report.

    Channels are listed only for sessions with a saved registration; sessions
    made before 2.1.0 cannot be warped again, so their aligned CSVs are not
    judged against their raw ones."""
    consensus_sha = _sha(_consensus_path())
    aligned = []
    try:
        saved = registration.load_registration()
    except (LookupError, FileNotFoundError):
        saved = None
    if saved is not None:
        for name, info in sorted((session.get_data("processed_files", default={}) or {}).items()):
            if not (isinstance(info, dict) and info.get("space") == "raw" and info.get("subtype") == "puncta_csv"):
                continue
            key = puncta.channel_key(name)
            raw_csv = constants.puncta_csv_path(_reports_dir(), name)
            if key is None or not raw_csv.exists():
                continue
            round_id, channel = key.split(" - ", 1)
            aligned.append(_aligned_status(round_id, channel, raw_csv, saved, consensus_sha))
    return aligned + [_report_status(aligned, consensus_sha)]


# --- Re-detecting one channel of a reopened session ------------------------------

def can_rewarp():
    """``(True, None)`` when puncta detected on a raw image can be warped into
    this session's aligned space, else ``(False, reason)``: sessions made
    before 2.1.0 have no saved registration."""
    try:
        registration.load_registration()
    except (LookupError, FileNotFoundError) as exc:
        return False, str(exc)
    return True, None


def redetect_channel(round_id, channel, params, existing="replace", image_path=None, consensus_mask=None,
                     run_by="gui", progress_callback=None):
    """Re-detect ``channel`` of ``round_id`` on its raw image and warp it into
    aligned space again: ``detect_channel`` then ``transform_channel``, the
    method batch uses. Returns ``(raw_csv, aligned_csv)``; either is None when
    there was nothing to write.

    ``existing`` is as for ``detect_channel``. Puncta placed by hand on the
    aligned layer are kept unless it is ``'replace_all'``. The GUI resolves
    ``image_path`` first, where it can ask for a moved file, and passes the
    consensus mask it shows."""
    def step(start, span):
        if progress_callback is None:
            return None
        return lambda p, text: progress_callback(start + int(p * span / 100), text)
    raw_csv = detect_channel(round_id, channel, params=params, existing=existing, image_path=image_path,
                             run_by=run_by, progress_callback=step(0, 85))
    if raw_csv is None:
        return None, None
    if progress_callback: progress_callback(85, f"Warping {round_id} {channel} into aligned space...")
    aligned_csv = transform_channel(round_id, channel, keep_aligned_manual=existing != "replace_all",
                                    run_by=run_by, consensus_mask=consensus_mask)
    if progress_callback: progress_callback(100, f"{round_id} {channel} re-detected and warped.")
    return raw_csv, aligned_csv


def report_note():
    """A sentence for the GUI when the session's report is out of date, else
    None (also when there is no report yet, or it is too old to tell)."""
    report = status()[-1]
    if report.state != STALE:
        return None
    more = len(report.reasons) - 1
    return (f"The report {report.name} is out of date: {report.reasons[0]}"
            + (f" (and {more} more change{'s' if more > 1 else ''})" if more else "")
            + ". Export it again to update it.")
