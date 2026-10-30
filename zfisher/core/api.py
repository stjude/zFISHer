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
from pathlib import Path
from typing import Optional

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
        files = sorted((path / constants.SESSIONS_DIR).glob("zfisher_session_*.json"), key=_session_number) \
            or sorted(path.glob("zfisher_session_*.json"), key=_session_number)
        if not files:
            raise FileNotFoundError(f"No zfisher_session_*.json in {path} or its sessions/ folder.")
        path = files[-1]
    s = _session.Session()
    s.load_into(path, read_only=read_only)
    return s


def _session_number(path):
    try:
        return int(Path(path).stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


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


def get_parameters(s):
    """The settings ``s`` was made with."""
    from . import puncta
    with _session.use(s):
        return Parameters(detection=puncta.detection_settings(),
                          colocalization_rules=list(s.get("colocalization_rules", []) or []),
                          tri_colocalization_rules=list(s.get("tri_colocalization_rules", []) or []),
                          registration=dict(s.get("registration", {}) or {}))


def query_report(s, sheet, limit=None):
    """The rows of one sheet of ``s``'s report, as dicts (at most ``limit``)."""
    import pandas as pd
    from . import stages
    info = (s.get("processed_files", {}) or {}).get(stages.REPORT_NAME)
    path = Path(info["path"]) if isinstance(info, dict) and info.get("path") else None
    if path is None or not path.exists():
        raise FileNotFoundError("The session has no exported report.")
    sheets = pd.ExcelFile(path).sheet_names
    if sheet not in sheets:
        raise ValueError(f"The report has no sheet '{sheet}'; it has {', '.join(sheets)}.")
    table = pd.read_excel(path, sheet_name=sheet)
    if limit is not None:
        table = table.head(int(limit))
    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


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


def get_decisions(s, name=None):
    """Who set which parameter of ``s``, oldest first (``decisions.jsonl``);
    only the decisions about ``name`` if given."""
    from . import decisions
    with _session.use(s):
        return decisions.records(name)


def schema():
    """The JSON schema of every result model, by name."""
    models = (SessionInfo, ArtifactState, StageState, Parameters, NucleusIdCheck, DetectionResult, TransformResult,
              ReportResult, PipelineResult)
    return {m.__name__: m.model_json_schema() for m in models}
