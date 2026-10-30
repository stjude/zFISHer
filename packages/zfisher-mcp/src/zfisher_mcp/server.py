"""The server: tools, resources and prompts over ``zfisher.core.api``.

Every tool names its session by the session's path relative to a root (see
``paths``), opens it read-only, calls one ``api`` function and returns that
function's typed result with paths made relative. The server keeps no state
between calls, so any number of clients can share it.

The server starts in a mode, and only that mode's tools exist: a client
cannot call what it is not shown. ``query`` has the read-only tools.
"""
import contextlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceError, ToolError
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from zfisher.core import api

from . import __version__
from .paths import Roots

logger = logging.getLogger(__name__)

SHEETS = Literal["Distances", "Colocalization", "Tri-Colocalization", "ROI per Nuclei", "Stats",
                 "Distribution", "Parameters", "Metadata"]
"""The report's sheets (``zfisher.constants``; a test keeps the two in step)."""

LOG_KINDS = Literal["STAGE", "ACTION", "STALL", "ENV"]

Limit = Annotated[int, Field(ge=1, description="Most items to return; the server caps it at its page size.")]
Offset = Annotated[int, Field(ge=0, description="How many items to skip.")]

MODES = {"query": ("read",)}
"""The tool groups each mode exposes."""

INSTRUCTIONS = """\
This server answers questions about zFISHer analyses: multi-round FISH image sets that zFISHer
has segmented, registered and measured. Each analysis is a session folder.

- Start with list_sessions; pass the "session" value it returns to every other tool.
- Distances are in micrometres, from the images' voxel sizes.
- query_distances counts, for each punctum of the source layer, whether its nearest punctum of
  the target layer lies within the distance. It is nearest-neighbour-within-d, not all pairs
  within d.
- get_status says which outputs are out of date (stale) relative to their inputs or settings.
- get_metrics gives the quality-control numbers (registration fit, puncta counts and SNR,
  extranuclear and hand-placed fractions, nucleus volumes).
- Paths in results are relative to the configured root, except inside provenance records,
  which name files relative to their session folder.
- Results come from the files on disk; nothing here changes them.
"""

_READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True,
                             open_world_hint=False)

_EXPECTED = (OSError, ValueError, LookupError, TypeError)
"""Errors a tool reports to the client as a message; anything else is a bug,
logged with its traceback, and the client sees only that the tool failed."""


@dataclass(frozen=True)
class ServerConfig:
    roots: Roots
    mode: str = "query"
    page_size: int = 200          # the most rows or records one call returns
    max_depth: int = 3            # how far below a root list_sessions looks
    log_level: str = "WARNING"    # of the server's own messages, to stderr


# --- Result models of the server's own (the rest are zfisher.core.api's) --------------

class SessionEntry(BaseModel):
    session: str                  # pass this to the other tools
    session_file: str             # the newest session file
    modified: str                 # its modification time, ISO 8601, UTC


class SessionList(BaseModel):
    total: int
    offset: int
    sessions: list[SessionEntry]


class SessionStatus(BaseModel):
    artifacts: list[api.ArtifactState]    # aligned puncta per channel, the report
    stages: list[api.StageState]          # every recorded pipeline stage


class RecordPage(BaseModel):
    total: int
    offset: int
    records: list[dict[str, Any]]


def build_server(config):
    """The ``MCPServer`` for ``config``."""
    if config.mode not in MODES:
        raise ValueError(f"No mode '{config.mode}'; the modes are {', '.join(MODES)}.")
    server = MCPServer("zfisher", title="zFISHer", version=__version__, instructions=INSTRUCTIONS,
                       log_level=config.log_level)
    tools = _Tools(config)
    for group in MODES[config.mode]:
        for fn in tools.group(group):
            server.add_tool(fn, annotations=_READ_ONLY if group == "read" else None)
    _add_resources(server, tools)
    _add_prompts(server)
    return server


class _Tools:
    """The tool functions, bound to one configuration."""

    def __init__(self, config):
        self.config = config
        self.roots = config.roots

    # --- plumbing ---------------------------------------------------------------------

    @contextlib.contextmanager
    def answer(self):
        """Turn an expected failure into a message for the client, with paths made relative."""
        try:
            yield
        except ToolError:
            raise
        except _EXPECTED as exc:
            raise ToolError(self.roots.scrub_text(str(exc)) or type(exc).__name__) from exc

    def open(self, session):
        path = self.roots.resolve(session)
        if not path.exists():
            raise FileNotFoundError(f"No session at '{session}'; list_sessions gives the sessions there are.")
        return api.open_session(path)                 # read-only: writes nothing

    def out(self, model_cls, value):
        """``value`` (a model or plain data) as ``model_cls`` with every path made relative."""
        data = value.model_dump() if isinstance(value, BaseModel) else value
        return model_cls.model_validate(self.roots.scrub(data))

    def page(self, records, limit, offset):
        records = list(records)
        return RecordPage(total=len(records), offset=offset,
                          records=self.roots.scrub(records[offset:offset + limit]))

    def relative_file(self, file):
        """A file named relative to a session folder: no absolute paths, no '..'."""
        text = file.replace("\\", "/").strip()
        if text.startswith("/") or ":" in text.split("/")[0] or ".." in text.split("/"):
            raise ValueError("Name the file relative to the session folder, as provenance records do.")
        return text

    def group(self, name):
        return {"read": [self.list_sessions, self.get_session_info, self.get_status, self.get_parameters,
                         self.get_metrics, self.query_distances, self.query_report, self.get_provenance,
                         self.get_decisions, self.read_log]}[name]

    # --- tier 1: read-only -------------------------------------------------------------
    # The annotations in the signatures are what the client sees as each tool's input schema.

    def list_sessions(self, under: Annotated[str, Field(
            description="A folder below the root to search; empty for the whole root.")] = "",
            limit: Limit = 100, offset: Offset = 0) -> SessionList:
        """The zFISHer sessions (analysis folders) under the root, newest session file of each."""
        with self.answer():
            limit = self._clamp(limit)
            folders = [self.roots.resolve(under)] if under else [f for _, f in self.roots.folders()]
            found = [f for folder in folders for f in api.find_sessions(folder, self.config.max_depth)]
            entries = []
            for folder in found[offset:offset + limit]:
                newest = api.session_files(folder)[-1]
                modified = datetime.fromtimestamp(newest.stat().st_mtime, timezone.utc)
                entries.append(SessionEntry(session=self.roots.relative(folder), session_file=newest.name,
                                            modified=modified.isoformat(timespec="seconds")))
            return SessionList(total=len(found), offset=offset, sessions=entries)

    def get_session_info(self, session: str) -> api.SessionInfo:
        """What a session is: its inputs, zFISHer version, nuclear channel, channels with
        puncta, and whether it has a consensus mask, a saved registration and a report."""
        with self.answer():
            return self.out(api.SessionInfo, api.session_info(self.open(session)))

    def get_status(self, session: str) -> SessionStatus:
        """Which outputs of a session are fresh, stale, edited or missing, with the reasons,
        and the state of every recorded pipeline stage."""
        with self.answer():
            s = self.open(session)
            return self.out(SessionStatus, {"artifacts": [a.model_dump() for a in api.get_status(s)],
                                            "stages": [st.model_dump() for st in api.get_stage_status(s)]})

    def get_parameters(self, session: str, channel: Annotated[Optional[str], Field(
            description="'R1 - FITC', or a channel only one round has; omit for every channel.")] = None
            ) -> api.Parameters:
        """The settings a session was made with: puncta detection per channel, the
        colocalization rules, and the saved registration."""
        with self.answer():
            return self.out(api.Parameters, api.get_parameters(self.open(session), channel))

    def get_metrics(self, session: str, include_saturation: Annotated[bool, Field(
            description="Also measure saturated voxels; reads the aligned images, so slower.")] = False
            ) -> api.Metrics:
        """Quality-control numbers of a session, computed from its files."""
        with self.answer():
            return self.out(api.Metrics, api.get_metrics(self.open(session), include_saturation))

    def query_distances(self, session: str,
                        source: Annotated[str, Field(description="Source layer: 'R1 - FITC', or a channel only one round has.")],
                        target: Annotated[str, Field(description="Target layer, named the same way.")],
                        max_um: Annotated[float, Field(gt=0, le=100, description="Distance in micrometres.")],
                        bin_um: Annotated[float, Field(ge=0, le=100, description=(
                            "Width of the histogram bins up to max_um, in micrometres; 0 for one bin."))] = 0.05,
                        ) -> api.DistanceQuery:
        """How many source puncta have their nearest target punctum within max_um, binned
        by distance, from the session's latest report."""
        with self.answer():
            return self.out(api.DistanceQuery, api.query_distances(self.open(session), source, target, max_um,
                                                                   bin_um or None))

    def query_report(self, session: str, sheet: SHEETS,
                     where: Annotated[Optional[dict[str, Any]], Field(description=(
                         "Column values rows must have, e.g. {\"Nucleus_ID\": 4}; a list matches any of its values."
                     ))] = None,
                     limit: Limit = 50, offset: Offset = 0) -> api.ReportPage:
        """A page of rows of one sheet of the session's latest report, and how many rows match."""
        with self.answer():
            return self.out(api.ReportPage, api.report_page(self.open(session), sheet, where,
                                                            self._clamp(limit), offset))

    def get_provenance(self, session: str, file: Annotated[Optional[str], Field(description=(
            "A file relative to the session folder, e.g. 'reports/Aligned_R1_-_FITC_puncta.csv'; "
            "omit for every file."))] = None,
            limit: Limit = 50, offset: Offset = 0) -> RecordPage:
        """How the session's files were made (inputs, settings, software version), oldest first."""
        with self.answer():
            s = self.open(session)
            return self.page(api.get_provenance(s, self.relative_file(file) if file else None),
                             self._clamp(limit), offset)

    def get_decisions(self, session: str, name: Annotated[Optional[str], Field(
            description="Only the decisions about this parameter.")] = None,
            limit: Limit = 50, offset: Offset = 0) -> RecordPage:
        """Who set which parameter of the session, and when, oldest first."""
        with self.answer():
            return self.page(api.get_decisions(self.open(session), name), self._clamp(limit), offset)

    def read_log(self, session: str, kind: LOG_KINDS, limit: Limit = 100,
                 offset: Offset = 0) -> RecordPage:
        """Timing lines of the session's logs, oldest first: STAGE (each pipeline step's
        time and memory), ACTION (GUI actions), STALL (GUI freezes) or ENV (the machine)."""
        with self.answer():
            return self.page(api.read_log(self.open(session), kind), self._clamp(limit), offset)

    def _clamp(self, limit):
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        return min(int(limit), self.config.page_size)


def _add_resources(server, tools):
    """Read-only views of a session's files, for clients that browse resources."""

    def read(fn):
        try:
            return fn()
        except ToolError as exc:
            raise ResourceError(str(exc)) from exc
        except _EXPECTED as exc:
            raise ResourceError(tools.roots.scrub_text(str(exc))) from exc

    @server.resource("zfisher://session/{+session}", mime_type="application/json",
                     description="The session file, paths made relative.")
    def session_file(session: str) -> str:
        return read(lambda: json.dumps(tools.roots.scrub(tools.open(session).snapshot()), indent=1, default=str))

    @server.resource("zfisher://provenance/{+session}", mime_type="application/json",
                     description="How each of the session's files was made.")
    def provenance(session: str) -> str:
        return read(lambda: json.dumps(tools.roots.scrub(api.get_provenance(tools.open(session))), indent=1))

    @server.resource("zfisher://decisions/{+session}", mime_type="application/json",
                     description="Who set which parameter, and when.")
    def decisions(session: str) -> str:
        return read(lambda: json.dumps(tools.roots.scrub(api.get_decisions(tools.open(session))), indent=1))


EXPLAIN_STATUS = """\
Explain the state of the zFISHer session "{session}" to the biologist who ran it.

1. Call get_session_info and get_status for it.
2. Say in one or two sentences what the session is (rounds, channels, whether it has a report).
3. List every output that is not fresh, each with its reason in plain words, and what would
   bring it up to date: re-running the stage named, or exporting the report again.
4. If everything is fresh, say so and stop.

Name channels as the session does ("R1 - FITC"). Do not guess at causes the tools do not give."""


def _add_prompts(server):
    @server.prompt(description="Explain what is out of date in a session, and why.")
    def explain_status(session: str) -> str:
        return EXPLAIN_STATUS.format(session=session)
