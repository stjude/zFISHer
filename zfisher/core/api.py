"""The operations every front end calls: the CLI now, other front ends later.

Each operation takes a ``session.Session`` (from ``open_session`` or
``create_session``), runs the core function on it (``session.use``), and
returns a typed result whose JSON schema other front ends can generate tool
signatures from. Nothing here holds logic of its own: it is the one surface
over ``stages``, ``pipeline``, ``puncta``, ``params`` and ``session``, and it
never touches the GUI's session unless given it.

Read-only use: ``open_session(path)`` reads a session without writing anything
(``read_only=True`` is the default), and every query below works on it.
Operations that change files need a session opened with ``read_only=False``,
which, as Load Session does, writes the next numbered session file.
"""
import copy
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from .. import constants
from . import session as _session
from .params import DetectionParams


class _Result(BaseModel):
    model_config = ConfigDict(frozen=True)


class SessionInfo(_Result):
    output_dir: str
    session_file: str
    read_only: bool
    r1_path: Optional[str] = None
    r2_path: Optional[str] = None
    nuclear_channel: str
    schema_version: int
    zfisher_version: Optional[str] = None
    created_with_zfisher: Optional[str] = None
    channels: list[str]              # "R1 - FITC", ... : channels with puncta, raw or aligned
    has_consensus_mask: bool
    has_saved_registration: bool
    has_report: bool


class ArtifactState(_Result):
    kind: str
    name: str
    path: Optional[str] = None
    state: str
    edited: bool = False
    reasons: list[str] = []
    round: Optional[str] = None
    channel: Optional[str] = None


class StageState(_Result):
    stage: str
    key: str = ""
    state: str                       # fresh, stale, edited or missing
    time: Optional[str] = None


class Parameters(_Result):
    detection: dict[str, dict]       # per channel key, as the report's Parameters sheet lists them
    colocalization_rules: list[dict]
    tri_colocalization_rules: list[dict]
    registration: dict               # the saved registration record (shift, rmsd, method, ...)


class ReportPage(_Result):
    sheet: str
    columns: list[str]
    total_rows: int                  # rows matching ``where``, before paging
    offset: int
    rows: list[dict]


class StagePlan(_Result):
    operation: str
    writes: list[str]                # files the operation writes
    makes_stale: list[str]           # outputs out of date afterwards, until made again


class ParamChange(_Result):
    scope: str                       # segmentation, registration, consensus, detection or report
    name: str
    channel: Optional[str] = None
    old: Any = None                  # None: not set, the built-in value applies
    new: Any = None
    applied: bool                    # False for a dry run or a value that was already set
    would_run: list[str]             # the stages ``update`` runs after the change


class BatchPlan(_Result):
    output_base: str
    datasets: list[dict]             # name, output_dir, r1, r2, new, would_run


class BatchRun(_Result):
    output_base: str
    datasets: list[dict]             # name, output_dir, ok, error, seconds, stages_run
    qc_report: Optional[str] = None


class ReviewItem(_Result):
    id: str
    time: str
    name: str                        # the setting
    value: Any = None                # the value proposed
    old: Any = None
    scope: dict = {}
    justification: Optional[str] = None
    actor: str                       # who proposed it (agent:<model>)
    via: Optional[str] = None        # the client it came through


class NucleusIdCheck(_Result):
    file: str
    rows: int
    agree: int
    fraction: float
    zero_but_inside: int
    repeated_ids: list[int]


class DetectionResult(_Result):
    csv: str
    n_puncta: int
    n_by_source: dict[str, int]
    parameters: dict


class TransformResult(_Result):
    csv: Optional[str]
    n_puncta: int


class ReportResult(_Result):
    path: str
    sheets: dict[str, int]           # sheet name -> rows


class DistanceQuery(_Result):
    source: str
    target: str
    max_um: float
    n_source: int                    # source puncta in the latest report
    n_within: int                    # of them, with a target punctum within max_um
    fraction_within: float
    bins: list[list[float]]          # [low_um, high_um, count] up to max_um
    semantics: str


class Metrics(_Result):
    dataset: str
    registration: dict               # method, fallback, rmsd_um, pairs, inliers, shift
    channels: dict[str, dict]        # per "R1 - FITC": counts, SNR quantiles, fractions, detected_on
    nuclei: dict                     # n_nuclei, volume_um3 quantiles
    saturation: Optional[dict] = None
    detection_mix: dict[str, int]


class BatchQC(_Result):
    batch_root: str
    n_datasets: int
    flags: list[dict]                # dataset, action, reason, metric, value, expected
    compared_with_batch: bool        # False: too few datasets, only the fixed limits were checked
    note: Optional[str] = None       # why the datasets were not compared with each other
    report: Optional[str] = None     # qc_report.md


class ThresholdSweep(_Result):
    channel: str
    thresholds: list[float]
    counts: list[int]
    plateau: Optional[dict] = None   # start, end, width, suggested, count
    params: dict


class CellsResult(_Result):
    folder: str
    n_cells: int
    paths: list[str]


class PipelineResult(_Result):
    output_dir: str
    session_file: str
    report: Optional[str] = None


# --- Sessions -------------------------------------------------------------------------

def open_session(path, read_only=True):
    """The session in the session file ``path`` (``zfisher_session_<n>.json``)
    or in the output folder ``path`` (its newest session file)."""
    path = Path(path)
    if path.is_dir():
        files = session_files(path)
        if not files:
            raise FileNotFoundError(f"No zfisher_session_*.json in {path} or its sessions/ folder.")
        path = files[-1]
    s = _session.Session()
    s.load_into(path, read_only=read_only)
    return s


def session_files(folder):
    """The session files of the output folder ``folder``, oldest number first:
    those in ``sessions/``, or beside it for folders from before that."""
    folder = Path(folder)
    return sorted((folder / constants.SESSIONS_DIR).glob("zfisher_session_*.json"), key=_session_number) \
        or sorted(folder.glob("zfisher_session_*.json"), key=_session_number)


def _session_number(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def find_sessions(root, max_depth=3):
    """The output folders under ``root``, ``root`` included, at most
    ``max_depth`` folders down: those with a session file in ``sessions/`` or,
    from before that folder, beside them. The search does not go below an
    output folder, into hidden folders or through symbolic links."""
    found = []

    def walk(folder, depth):
        if session_files(folder):
            found.append(folder)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in folder.iterdir()
                              if p.is_dir() and not p.is_symlink() and not p.name.startswith("."))
        except OSError:
            return
        for child in children:
            walk(child, depth + 1)

    walk(Path(root), 0)
    return found


def create_session(output_dir, r1_path, r2_path, progress_callback=None):
    """A new session in ``output_dir`` for the two round images. Raises
    ``FileExistsError`` if the folder already has one."""
    s = _session.Session()
    with _session.use(s):
        if not _session.initialize_new_session(output_dir, r1_path, r2_path, progress_callback=progress_callback):
            raise FileExistsError(f"{output_dir} already has a session; open it instead.")
    return s


def session_info(s):
    """What ``s`` is: its folder, inputs, versions, channels and what it has made."""
    from . import registration, stages
    with _session.use(s):
        data = s.snapshot()
        files = data.get("processed_files", {}) or {}
        from .puncta import channel_key
        channels = sorted({channel_key(n) for n, i in files.items()
                           if isinstance(i, dict) and i.get("subtype") == "puncta_csv" and channel_key(n)})
        try:
            registration.load_registration()
            saved = True
        except (LookupError, FileNotFoundError):
            saved = False
        return SessionInfo(
            output_dir=str(data.get("output_dir")), session_file=str(data.get("session_filename")),
            read_only=s.read_only, r1_path=data.get("r1_path"), r2_path=data.get("r2_path"),
            nuclear_channel=s.nuclear_channel(), schema_version=data.get("schema_version", 1),
            zfisher_version=data.get("zfisher_version"), created_with_zfisher=data.get("created_with_zfisher"),
            channels=channels, has_consensus_mask=stages._consensus_path() is not None,
            has_saved_registration=saved, has_report=stages.REPORT_NAME in files)


def get_status(s):
    """What is out of date in ``s``: each channel's aligned puncta and the report."""
    from . import stages
    with _session.use(s):
        return [ArtifactState(kind=a.kind, name=a.name, path=str(a.path) if a.path else None, state=a.state,
                              edited=a.edited, reasons=list(a.reasons), round=a.round, channel=a.channel)
                for a in stages.status()]


def get_stage_status(s):
    """Every pipeline stage recorded in ``s``: fresh, stale, edited or missing."""
    from . import stages
    with _session.use(s):
        return [StageState(**st) for st in stages.stage_status()]


def update(s, progress_callback=None):
    """Run again, on ``s``'s folder and with the settings its pipeline run
    stored, the stages that are not up to date (and what depends on them).
    Returns the stages that ran; none when everything was fresh."""
    from . import pipeline
    settings = s.get("pipeline_settings")
    if not settings:
        raise LookupError("This session was not made by a pipeline run, so there is nothing to update from.")
    return pipeline.run_full_zfisher_pipeline(s.get("r1_path"), s.get("r2_path"), s.get("output_dir"),
                                              progress_callback=progress_callback, **settings)


def plan_update(s, settings=None):
    """The stages ``update`` would run on ``s`` (``pipeline.plan_full_zfisher_pipeline``),
    with its stored pipeline settings or with ``settings`` instead. Runs and
    writes nothing."""
    from . import pipeline
    stored = s.get("pipeline_settings")
    if not stored:
        raise LookupError("This session was not made by a pipeline run, so there is nothing to update from.")
    return pipeline.plan_full_zfisher_pipeline(s.get("r1_path"), s.get("r2_path"), s.get("output_dir"),
                                               **(settings if settings is not None else stored))


def _setting_scopes():
    """``{scope: (model, {model field: pipeline setting})}`` of the settings
    ``set_param`` changes, other than detection and report ones."""
    from .params import ConsensusParams, RegistrationParams, SegmentationParams
    return {
        "segmentation": (SegmentationParams, {"method": "seg_method", "merge_splits": "merge_splits"}),
        "registration": (RegistrationParams, {"max_ransac_distance": "max_ransac_distance",
                                              "apply_warp": "apply_warp"}),
        "consensus": (ConsensusParams, {"overlap_method": "overlap_method", "match_threshold": "match_threshold",
                                        "remove_extranuclear_puncta": "remove_extranuclear_puncta",
                                        "remove_border_nuclei": "remove_border_nuclei"}),
    }


SETTING_SCOPES = ("segmentation", "registration", "consensus", "detection", "report")


def set_param(s, scope, name, value, channel=None, justification=None, dry_run=False):
    """Change one pipeline setting of ``s`` for its next ``update``, after
    checking ``value`` against the setting's allowed range or choices (the
    models of ``params``). ``scope`` is segmentation, registration, consensus,
    detection (with ``channel``) or report (``distribution_bins``). The change
    is stored in the session and logged in ``decisions.jsonl`` with
    ``justification``, unless ``dry_run``; either way the result says which
    stages ``update`` would then run. Raises ``ValueError`` for a setting that
    does not exist or a value outside its range, and ``LookupError`` for a
    session no pipeline run made."""
    from . import analysis, decisions, pipeline
    settings = s.get("pipeline_settings")
    if not settings:
        raise LookupError("This session was not made by a pipeline run, so it has no settings to change.")
    new_settings = copy.deepcopy(settings)
    key = name
    if scope == "detection":
        if not channel:
            raise ValueError("A detection setting needs the channel it applies to.")
        fields = DetectionParams.model_fields
        key = "threshold_rel" if name == "sensitivity" else name
        if key not in fields:
            raise ValueError(f"No detection setting '{name}'; they are {', '.join(fields)}.")
        config = pipeline.effective_puncta_config(**{"r1_path": s.get("r1_path"), "r2_path": s.get("r2_path"),
                                                     **settings})
        if channel not in config:
            raise ValueError(f"No detection settings for channel '{channel}'; the channels are "
                             f"{', '.join(config) or 'none'}.")
        current = dict(config[channel])
        old = current.get(key)
        checked = DetectionParams.model_validate({**{k: v for k, v in current.items() if k in fields}, key: value})
        new = checked.model_dump(mode="json")[key]
        if (key == "z_scale" and new is None) or (key == "z_edge_exclude" and not new):
            current.pop(key, None)                  # as detection_kwargs leaves them out when off
        else:
            current[key] = new
        config[channel] = current
        new_settings["puncta_config"] = config
    elif scope == "report":
        if name != "distribution_bins":
            raise ValueError(f"No report setting '{name}'; the one there is is distribution_bins.")
        old = settings.get("distribution_bins")
        if not isinstance(value, (list, tuple)) or not all(isinstance(v, int) and not isinstance(v, bool)
                                                            for v in value):
            raise ValueError("distribution_bins is a list of whole numbers, the lower edges of the bins.")
        new = list(value)
        analysis.distribution_bins(new)             # raises unless they start at 0 and rise
        new_settings["distribution_bins"] = new
    elif scope in _setting_scopes():
        model, keys = _setting_scopes()[scope]
        if name not in keys:
            raise ValueError(f"No {scope} setting '{name}'; they are {', '.join(keys)}.")
        current = {f: settings[k] for f, k in keys.items() if settings.get(k) is not None}
        if scope == "segmentation" and "method" in current:
            from .params import SegMethod
            current["method"] = SegMethod.parse(current["method"])
        old = settings.get(keys[name])
        new = model.model_validate({**current, name: value}).model_dump(mode="json")[name]
        new_settings[keys[name]] = new
    else:
        raise ValueError(f"No setting scope '{scope}'; the scopes are {', '.join(SETTING_SCOPES)}.")
    would_run = plan_update(s, new_settings)
    unchanged = new_settings == settings or (scope == "detection" and new_settings["puncta_config"] == (
        pipeline.effective_puncta_config(**{"r1_path": s.get("r1_path"), "r2_path": s.get("r2_path"), **settings})))
    applied = not dry_run and not unchanged
    if applied:
        with _session.use(s):
            s.set("pipeline_settings", new_settings)
            decisions.record(key, new, old=old, scope={"stage": scope, **({"channel": channel} if channel else {})},
                             justification=justification)
    return ParamChange(scope=scope, name=key, channel=channel, old=old, new=new, applied=applied,
                       would_run=would_run)


def plan_operation(s, operation, round_id=None, channel=None):
    """What an operation on ``s`` would write and leave out of date, without
    running it: ``detect`` a channel (its raw-space CSV; its aligned CSV and
    the report go stale), ``transform`` it (its aligned CSV; the report goes
    stale), ``report`` (the report and its tables) or ``cells`` (the per-nucleus
    folder)."""
    from . import analysis, cells, stages
    with _session.use(s):
        artifacts = stages.status()
        out = Path(s.get("output_dir"))
        reports = out / constants.REPORTS_DIR
        report = next((a for a in artifacts if a.kind == "report"), None)
        existing_report = [str(report.path)] if report is not None and report.path else []

        def aligned():
            found = next((a for a in artifacts if a.kind == "aligned_puncta" and a.round == round_id
                          and a.channel == channel), None)
            return found.path if found is not None and found.path else None

        if operation in ("detect", "transform"):
            if not (round_id and channel):
                raise ValueError(f"Say which round and channel to {operation}.")
        if operation == "detect":
            raw = constants.puncta_csv_path(reports, stages.raw_puncta_name(round_id, channel))
            writes, stale = [str(raw)], ([str(aligned())] if aligned() else []) + existing_report
        elif operation == "transform":
            target = aligned()
            if target is None:
                from . import registration
                try:
                    name = stages.aligned_puncta_name(round_id, channel, registration.load_registration())
                except (LookupError, FileNotFoundError):
                    name = f"{constants.ALIGNED_PREFIX} {round_id} - {channel}{constants.PUNCTA_SUFFIX}"
                target = constants.puncta_csv_path(reports, name)
            writes, stale = [str(target)], existing_report
        elif operation == "report":
            from .report import next_free_path             # an export never overwrites a report
            writes = [str(next_free_path(reports / f"{stages.REPORT_NAME}{constants.EXCEL_SUFFIX}")),
                      str(reports / analysis.DISTANCES_SIDECAR), str(reports / analysis.SHEETS_DIR)]
            stale = []
        elif operation == "cells":
            writes, stale = [str(out / cells.CELLS_DIR)], []
        else:
            raise ValueError(f"No operation '{operation}'; they are detect, transform, report and cells.")
        return StagePlan(operation=operation, writes=writes, makes_stale=stale)


def propose(s, name, value, scope=None, justification=None):
    """Log a suggested change of setting ``name`` to ``value`` in ``s``'s
    ``decisions.jsonl`` with ``status: "proposed"``, for a person to accept or
    dismiss; nothing else changes. Works on a read-only session. Returns the
    record."""
    from . import decisions
    with _session.use(s):
        entry = decisions.record(name, value, scope=scope or {}, justification=justification, status="proposed",
                                 even_read_only=True)
    if entry is None:
        raise LookupError("The session has no folder to log the proposal in.")
    return entry


def review_queue(s):
    """The proposals in ``s``'s decisions log that nobody has accepted or
    dismissed yet, oldest first."""
    records = get_decisions(s)
    reviewed = {r["new"].get("id") for r in records
                if r.get("name") == "proposal_review" and isinstance(r.get("new"), dict)}
    return [ReviewItem(id=r["id"], time=r.get("time", ""), name=r.get("name", ""), value=r.get("new"),
                       old=r.get("old"), scope=r.get("scope") or {}, justification=r.get("justification"),
                       actor=r.get("actor", ""), via=r.get("via"))
            for r in records if r.get("status") == "proposed" and r.get("id") and r["id"] not in reviewed]


def review_proposal(s, proposal_id, accept, note=None):
    """Accept or dismiss the open proposal ``proposal_id`` of ``s``: logged in
    its decisions log, by whoever the front end stamps (the GUI: ``human``),
    with ``note``. Accepting records agreement; the change itself is made
    separately. Returns the record."""
    from . import decisions
    proposal = next((p for p in review_queue(s) if p.id == proposal_id), None)
    if proposal is None:
        raise LookupError(f"No open proposal '{proposal_id}'; it may have been reviewed already.")
    with _session.use(s):
        return decisions.record("proposal_review", {"id": proposal.id, "name": proposal.name, "value": proposal.value},
                                scope=proposal.scope, justification=note,
                                status="accepted" if accept else "dismissed", even_read_only=True)


def get_parameters(s, channel=None):
    """The settings ``s`` was made with; the detection settings of one
    channel only if ``channel`` names one (``'R1 - FITC'``, or a channel only
    one round has)."""
    from . import puncta, stages
    with _session.use(s):
        detection = puncta.detection_settings()
        if channel is not None:
            key = stages.resolve_layer(channel, sorted(detection))
            detection = {key: detection[key]}
        return Parameters(detection=detection,
                          colocalization_rules=list(s.get("colocalization_rules", []) or []),
                          tri_colocalization_rules=list(s.get("tri_colocalization_rules", []) or []),
                          registration=dict(s.get("registration", {}) or {}))


def query_report(s, sheet, limit=None):
    """The rows of one sheet of ``s``'s latest report, as dicts (at most
    ``limit``), read from the CSV sidecar written beside the report; reports
    exported before sidecars existed are read from the workbook."""
    table = _report_table(s, sheet)
    if limit is not None:
        table = table.head(int(limit))
    return _records(table)


def report_page(s, sheet, where=None, limit=200, offset=0):
    """One page of a sheet of ``s``'s latest report: the rows whose columns
    equal the values in ``where`` (a list matches any of its values), from
    row ``offset``, at most ``limit`` of them, and how many matched."""
    import pandas as pd
    table = _report_table(s, sheet)
    for column, wanted in (where or {}).items():
        if column not in table.columns:
            raise ValueError(f"The {sheet} sheet has no column '{column}'; it has "
                             f"{', '.join(map(str, table.columns))}.")
        values = wanted if isinstance(wanted, (list, tuple)) else [wanted]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            match = pd.to_numeric(table[column], errors="coerce").isin(values)
        else:
            match = table[column].astype(str).isin([str(v) for v in values])
        table = table[match]
    offset, limit = max(0, int(offset)), max(0, int(limit))
    return ReportPage(sheet=sheet, columns=[str(c) for c in table.columns], total_rows=len(table),
                      offset=offset, rows=_records(table.iloc[offset:offset + limit]))


def _records(table):
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def _report_table(s, sheet):
    """The sheet ``sheet`` of ``s``'s latest report as a table."""
    import pandas as pd
    from . import analysis, stages
    sidecars = Path(s.get("output_dir")) / constants.REPORTS_DIR / analysis.SHEETS_DIR
    if sidecars.exists():
        available = {p.stem: p for p in sidecars.glob("*.csv")}
        wanted = analysis.sheet_file_name(sheet)[:-4]
        if wanted not in available:
            raise ValueError(f"The report has no sheet '{sheet}'; it has "
                             f"{', '.join(sorted(k.replace('_', ' ') for k in available))}.")
        table = pd.read_csv(available[wanted])
    else:
        info = (s.get("processed_files", {}) or {}).get(stages.REPORT_NAME)
        path = Path(info["path"]) if isinstance(info, dict) and info.get("path") else None
        if path is None or not path.exists():
            raise FileNotFoundError("The session has no exported report.")
        sheets = pd.ExcelFile(path).sheet_names
        if sheet not in sheets:
            raise ValueError(f"The report has no sheet '{sheet}'; it has {', '.join(sheets)}.")
        table = pd.read_excel(path, sheet_name=sheet)
    return table


def query_distances(s, source, target, max_um, bin_um=None):
    """How many ``source`` puncta have a ``target`` punctum within ``max_um``
    µm in ``s``'s latest report (``analysis.query_distances``), read from
    ``reports/distances.csv``. ``source`` and ``target`` may name a layer, a
    round and channel, or a channel only one round has."""
    import pandas as pd
    from . import analysis, stages
    path = Path(s.get("output_dir")) / constants.REPORTS_DIR / analysis.DISTANCES_SIDECAR
    if not path.exists():
        raise FileNotFoundError("The session has no distances table yet; export the report first.")
    distances = pd.read_csv(path)
    names = sorted(set(distances["Source_Layer"]) | set(distances["Target_Layer"]))
    return DistanceQuery(**analysis.query_distances(distances, stages.resolve_layer(source, names),
                                                    stages.resolve_layer(target, names), max_um, bin_um))


def get_metrics(s, include_saturation=False):
    """Quality-control numbers of ``s`` from its files (``metrics.compute_metrics``)."""
    from . import metrics
    with _session.use(s):
        return Metrics(**metrics.compute_metrics(include_saturation=include_saturation))


def flag_batch(batch_root, thresholds=None, write_report=True, include_saturation=False):
    """Which datasets under ``batch_root`` need a look (``qc.flag_batch``);
    writes ``qc_report.md`` there unless told not to. Reads the datasets only."""
    from . import qc
    from .params import QCThresholds
    if isinstance(thresholds, dict):
        thresholds = QCThresholds.model_validate(thresholds)
    cohort = qc.cohort_metrics(batch_root, include_saturation=include_saturation)
    flags = qc.flag_batch(cohort, thresholds)
    report = qc.write_qc_report(batch_root, cohort, flags, thresholds) if write_report and cohort else None
    return BatchQC(batch_root=str(batch_root), n_datasets=len(cohort), flags=qc.flags_as_dicts(flags),
                   compared_with_batch=qc.compared_with_batch(cohort, thresholds),
                   note=qc.small_batch_note(cohort, thresholds), report=str(report) if report else None)


def threshold_sweep(s, round_id, channel, params=None, thresholds=None, progress_callback=None):
    """Puncta counts over a range of sensitivities for ``channel`` and where they
    level off (``stages.sweep_channel``); stored in ``s``. ``params`` (a
    ``DetectionParams`` or dict) sets the other settings, else the recorded ones."""
    from . import stages
    if isinstance(params, dict):
        params = DetectionParams.model_validate(params)
    kwargs = params.detection_kwargs() if params is not None else None
    with _session.use(s):
        result = stages.sweep_channel(round_id, channel, kwargs, thresholds, progress_callback=progress_callback)
    return ThresholdSweep(channel=f"{round_id} - {channel}", **result)


def check_nucleus_ids(s):
    """How well each aligned puncta CSV's ``Nucleus_ID`` agrees with the saved
    consensus mask (``puncta.check_nucleus_ids``); reads only."""
    from . import io, puncta, stages
    with _session.use(s):
        mask_path = stages._consensus_path()
        if mask_path is None:
            raise FileNotFoundError("The session has no consensus nuclei mask.")
        mask = io.read_label_tif(mask_path)
        return [NucleusIdCheck(**puncta.check_nucleus_ids(csv, mask))
                for csv in puncta.aligned_puncta_csvs(Path(s.get("output_dir")) / constants.REPORTS_DIR)]


# --- Operations that write -------------------------------------------------------------

def detect_channel(s, round_id, channel, params=None, existing="replace"):
    """Detect ``channel`` on round ``round_id``'s raw image (``stages.detect_channel``).
    ``params`` is a ``DetectionParams``, a dict of its fields, or None for the
    settings recorded last time."""
    import pandas as pd
    from . import stages
    if isinstance(params, dict):
        params = DetectionParams.model_validate(params)
    kwargs = params.detection_kwargs() if params is not None else None
    with _session.use(s):
        csv = stages.detect_channel(round_id, channel, kwargs, existing=existing, run_by="api")
        if csv is None:
            return DetectionResult(csv="", n_puncta=0, n_by_source={}, parameters=kwargs or {})
        table = pd.read_csv(csv)
        counts = table["Source"].value_counts().to_dict() if "Source" in table else {}
        recorded = stages.recorded_detection_parameters(round_id, channel) or {}
        return DetectionResult(csv=str(csv), n_puncta=len(table), n_by_source={str(k): int(v) for k, v in counts.items()},
                               parameters=recorded)


def transform_channel(s, round_id, channel, remove_extranuclear=None):
    """Warp ``channel``'s raw puncta into aligned space with the saved registration."""
    import pandas as pd
    from . import stages
    with _session.use(s):
        csv = stages.transform_channel(round_id, channel, remove_extranuclear=remove_extranuclear, run_by="api")
        return TransformResult(csv=str(csv) if csv else None, n_puncta=len(pd.read_csv(csv)) if csv else 0)


def export_report(s, rules=None, tri_rules=None, filename=None):
    """Write ``s``'s colocalization report from its files (``stages.export_report``)."""
    import pandas as pd
    from . import stages
    with _session.use(s):
        path = stages.export_report(rules, tri_rules, filename)
    rows = {name: len(frame) for name, frame in pd.read_excel(path, sheet_name=None).items()}
    return ReportResult(path=str(path), sheets=rows)


def export_cells(s, ids=None, padding_um=2.0, with_puncta=True, progress_callback=None):
    """One OME-TIFF per nucleus of ``s`` (``cells.export_cells``)."""
    from . import cells
    with _session.use(s):
        paths = cells.export_cells(ids=ids, padding_um=padding_um, with_puncta=with_puncta,
                                   progress_callback=progress_callback)
        folder = Path(s.get("output_dir")) / cells.CELLS_DIR
    return CellsResult(folder=str(folder), n_cells=len(paths), paths=[str(p) for p in paths])


def run_pipeline(dataset, output_dir, puncta_config=None, pairwise_rules=None, tri_rules=None,
                 progress_callback=None):
    """Run the whole pipeline for one ``params.DatasetSpec`` into ``output_dir``,
    on a session of its own (``pipeline.run_full_zfisher_pipeline``)."""
    from . import pipeline
    from .generate_batch_template import pipeline_kwargs
    pipeline.run_full_zfisher_pipeline(output_dir=output_dir, puncta_config=puncta_config,
                                       pairwise_rules=pairwise_rules, tri_rules=tri_rules,
                                       progress_callback=progress_callback, **pipeline_kwargs(dataset))
    s = open_session(output_dir)
    report = (s.get("processed_files", {}) or {}).get("Master_Analysis_Report")
    return PipelineResult(output_dir=str(output_dir), session_file=str(s.get("session_filename")),
                          report=report.get("path") if isinstance(report, dict) else None)


def read_batch_template(path):
    """The ``params.BatchConfig`` of the batch template (``.xlsx``) at ``path``.
    Raises ``SchemaError`` saying what is wrong with it."""
    from .errors import SchemaError
    from .generate_batch_template import parse_batch_config
    config, error = parse_batch_config(path)
    if error:
        raise SchemaError(f"The batch template cannot be used: {error}")
    return config


def plan_batch(config, output_base):
    """For each dataset of ``config`` (a ``BatchConfig``, or a template's
    path), its input images, output folder, and the stages a batch run into
    ``output_base`` would run there: every stage for a folder with no run yet
    (``new``), else those not up to date. Runs and writes nothing."""
    from . import batch, pipeline
    from .generate_batch_template import pipeline_kwargs
    if not isinstance(config, BaseModel):
        config = read_batch_template(config)
    datasets = []
    for name, ds, out, puncta_config, pairwise, tri in batch.dataset_jobs(config, output_base):
        entry = {"name": name, "output_dir": str(out), "r1": str(ds.r1), "r2": str(ds.r2), "new": True,
                 "would_run": ["every stage"]}
        try:
            entry["would_run"] = pipeline.plan_full_zfisher_pipeline(
                output_dir=out, puncta_config=puncta_config, pairwise_rules=pairwise, tri_rules=tri,
                **pipeline_kwargs(ds))
            entry["new"] = False
        except LookupError:
            pass
        datasets.append(entry)
    return BatchPlan(output_base=str(output_base), datasets=datasets)


def run_batch(config, output_base, workers=1):
    """Run every dataset of ``config`` (a ``BatchConfig``, or a template's
    path) into ``output_base`` (``batch.run_batch``), then write the batch's
    ``qc_report.md``. A dataset that fails is reported, not raised."""
    from . import batch
    from .qc import QC_REPORT
    if not isinstance(config, BaseModel):
        config = read_batch_template(config)
    results = batch.run_batch(config, output_base, workers=workers)
    report = Path(output_base) / QC_REPORT
    return BatchRun(output_base=str(output_base),
                    datasets=[{"name": r.name, "output_dir": r.output_dir, "ok": r.ok, "error": r.error,
                               "seconds": round(r.seconds, 1), "stages_run": list(r.stages_run)} for r in results],
                    qc_report=str(report) if report.exists() else None)


def get_decisions(s, name=None):
    """Who set which parameter of ``s``, oldest first (``decisions.jsonl``);
    only the decisions about ``name`` if given."""
    from . import decisions
    with _session.use(s):
        return decisions.records(name)


def get_provenance(s, path=None):
    """How the files of ``s`` were made (``provenance.jsonl``), oldest first;
    only the records of ``path`` if given, relative to the session's folder
    or absolute."""
    from . import provenance
    with _session.use(s):
        if path is not None and not Path(path).is_absolute():
            path = Path(s.get("output_dir")) / path
        return provenance.records(path)


def read_log(s, kind=None):
    """The timing lines of ``s``'s session logs (``logs/zfisher_*.log``),
    oldest log first: each a dict of its fields with ``kind`` (STAGE, ACTION,
    STALL or ENV), ``time`` and ``log``, the log file's name. Only lines of
    ``kind`` if given."""
    from . import log_profile
    logs = Path(s.get("output_dir")) / constants.LOGS_DIR
    return log_profile.timing_records(log_profile.find_logs([logs]) if logs.is_dir() else [], kind)


def schema():
    """The JSON schema of every result model, by name."""
    models = (SessionInfo, ArtifactState, StageState, DistanceQuery, Metrics, BatchQC, ThresholdSweep, Parameters, ReportPage, StagePlan, ParamChange, BatchPlan, BatchRun, ReviewItem, NucleusIdCheck, DetectionResult, TransformResult,
              ReportResult, CellsResult, PipelineResult)
    return {m.__name__: m.model_json_schema() for m in models}
