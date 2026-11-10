"""Tools that act. ``triage`` mode adds flagging a batch and proposing a
change; ``operate`` mode adds changing settings, running stages and running a
batch.

Every tool that changes files takes ``dry_run``, true unless the client says
otherwise. A dry run says what the call would write, what it would leave out
of date and which stages would run, and changes nothing. A real run opens the
session for writing (zFISHer saves it under the next session number, as Load
Session does), runs one ``zfisher.core.api`` operation, and stamps every
decision it logs with the server's actor and the client the call came from.
Calls that write to one folder run one at a time. Sessions are reached, and
changes run, through the server's host (``host``), so a GUI that runs the
server can give it its own open session.
"""
import contextvars
import logging
import threading
from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from mcp.server.mcpserver import Context
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from zfisher import constants
from zfisher.core import api, decisions
from zfisher.core.params import DetectionParams
from zfisher.core.qc import QC_REPORT

logger = logging.getLogger(__name__)


class Operation(BaseModel):
    dry_run: bool
    writes: list[str] = []          # files or folders the call writes (would write)
    makes_stale: list[str] = []     # outputs out of date afterwards
    would_run: list[str] = []       # pipeline stages the change makes due (set_param, update)
    result: Optional[dict[str, Any]] = None


class Proposal(BaseModel):
    record: dict[str, Any]          # the decisions-log record, status "proposed"


Round = Literal["R1", "R2"]
Scope = Literal["segmentation", "registration", "consensus", "detection", "report"]
DryRun = Annotated[bool, Field(description="True (the default): say what the call would do, and change nothing.")]
Justification = Annotated[str, Field(min_length=1, description=(
    "Why: logged with the change for the person who reviews it."))]
Channel = Annotated[str, Field(description=(
    "The channel as the image names it ('FITC'), or with its round as the read tools name it ('R1 - FITC')."))]

TRIAGE = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
SETS = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
WRITES = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)


def client_name(ctx):
    """The name the client gave in its handshake, or ``unknown``."""
    try:
        params = ctx.session.client_params
        info = params.client_info if params is not None else None
        return info.name if info is not None and info.name else "unknown"
    except Exception:
        return "unknown"


class ActingTools:
    """The triage and operate tools; mixed into the server's tool set."""

    def triage_tools(self):
        return [(self.flag_batch, TRIAGE), (self.propose, TRIAGE)]

    def operate_tools(self):
        return [(self.set_param, SETS), (self.update, WRITES), (self.detect_channel, WRITES),
                (self.transform_channel, WRITES), (self.export_report, WRITES), (self.export_cells, WRITES),
                (self.threshold_sweep, SETS), (self.run_batch, WRITES)]

    # --- plumbing ---------------------------------------------------------------------

    def act(self, ctx, folder, what, fn, session="write"):
        """Run the change ``fn`` to ``folder`` through the host, holding the
        folder's writer lock, with the server's actor and the client stamped on
        every decision it logs. ``fn`` is given the folder's session, opened
        for writing (``session="write"``) or not (``"read"``), or nothing
        (``None``)."""
        with self._locks_lock:
            lock = self._locks.setdefault(Path(folder), threading.Lock())
        client = client_name(ctx)
        with lock, decisions.use_actor(self.config.actor, via=f"mcp:{client}"):
            logger.info("%s on %s by %s via %s", what, self.roots.relative(folder), self.config.actor, client)
            context = contextvars.copy_context()   # the actor, for a host that runs fn on another thread

            def change():
                if session is None:
                    return fn()
                return fn(self.host.session(folder, writable=session == "write"))
            return self.host.run_write(what, folder, lambda: context.run(change))

    def folder(self, session):
        path = self.roots.resolve(session)
        if not path.exists():
            raise FileNotFoundError(f"No session at '{session}'; list_sessions gives the sessions there are.")
        return path

    def operation(self, dry_run, plan=None, would_run=(), result=None, writes=None):
        data = {"dry_run": dry_run, "would_run": list(would_run),
                "writes": list(writes if writes is not None else (plan.writes if plan else [])),
                "makes_stale": list(plan.makes_stale) if plan else [],
                "result": result.model_dump() if isinstance(result, BaseModel) else result}
        return Operation.model_validate(self.roots.scrub(data))

    # --- triage -----------------------------------------------------------------------

    def flag_batch(self, under: Annotated[str, Field(description=(
            "The batch folder below the root (the folder holding the datasets); empty for the root."))] = "",
            write_report: Annotated[bool, Field(description=(
                "Also write qc_report.md in the batch folder."))] = False) -> api.BatchQC:
        """Which datasets of a batch need a look, and why: each dataset's quality-control
        numbers compared with the rest of the batch and with fixed floors. A batch of fewer
        than four datasets is checked against the fixed floors only (compared_with_batch
        false, with a note), so no flags then does not mean every dataset is alike."""
        with self.answer():
            if under:
                folder = self.roots.resolve(under)
            elif self.roots.single:
                folder = self.roots.folders()[0][1]
            else:
                raise ValueError(f"Name the batch folder, starting with a root: {', '.join(self.roots.names)}.")
            return self.out(api.BatchQC, api.flag_batch(folder, write_report=write_report))

    def propose(self, ctx: Context, session: str, name: Annotated[str, Field(description=(
                "The setting, as get_parameters names it (e.g. threshold_rel)."))],
                value: Annotated[Any, Field(description="The value suggested.")],
                justification: Justification,
                channel: Optional[str] = None, round: Optional[Round] = None) -> Proposal:
        """Suggest a change for a person to accept or dismiss: logged in the session's decisions
        log as a proposal; nothing else changes."""
        with self.answer():
            folder = self.folder(session)
            scope = {k: v for k, v in (("round", round), ("channel", channel)) if v}
            entry = self.act(ctx, folder, f"propose {name}", lambda s: api.propose(
                s, name, value, scope=scope, justification=justification), session="read")
            return Proposal(record=self.roots.scrub(entry))

    # --- operate ----------------------------------------------------------------------

    def set_param(self, ctx: Context, session: str, scope: Scope,
                  name: Annotated[str, Field(description=(
                      "The setting: segmentation method or merge_splits; registration max_ransac_distance or "
                      "apply_warp; consensus overlap_method, match_threshold, remove_extranuclear_puncta or "
                      "remove_border_nuclei; detection threshold_rel (sensitivity), min_distance, sigma, "
                      "method, z_edge_exclude, use_tophat, tophat_radius or nuclei_only; report "
                      "distribution_bins."))],
                  value: Annotated[Any, Field(description="The new value, inside the setting's allowed range.")],
                  justification: Justification,
                  channel: Annotated[Optional[str], Field(description=(
                      "The channel, for a detection setting: 'FITC', or 'R1 - FITC' as the read tools name it. "
                      "Either way the setting is the channel's in every round it is detected in."))
                  ] = None,
                  dry_run: DryRun = True) -> Operation:
        """Change one pipeline setting of a session for its next update, refused outside the
        setting's allowed range or choices. Says which stages update would then run (get_status
        lists them too until update runs them); the outputs stay as they are until then. A
        detection setting is the channel's in every round it is detected in (rounds in the
        result). A value already set changes and writes nothing (unchanged in the result)."""
        with self.answer():
            folder = self.folder(session)
            # Checked on a read-only open first, so a refused value writes no session file.
            change = api.set_param(self.host.session(folder), scope, name, value, channel=channel, dry_run=True)
            log = str(folder / decisions.DECISIONS_FILE)
            if change.unchanged:
                return self.operation(dry_run, result=change, writes=[])
            if dry_run:
                return self.operation(True, would_run=change.would_run, result=change,
                                      writes=[str(folder / constants.SESSIONS_DIR), log])
            saved = []

            def change_it(s):
                saved.append(api.session_file_path(s))
                return api.set_param(s, scope, name, value, channel=channel, justification=justification)
            change = self.act(ctx, folder, f"set_param {scope}.{name}", change_it)
            return self.operation(False, would_run=change.would_run, result=change, writes=[str(saved[0]), log])

    def update(self, ctx: Context, session: str, dry_run: DryRun = True) -> Operation:
        """Run again the pipeline stages of a session that are not up to date with its
        settings and files (and the stages after them). A real run says which stages ran and
        what the session then holds (outcome: registration, nuclei, each channel's aligned
        count and median SNR, missing outputs, warnings): a run can finish and make little."""
        with self.answer():
            folder = self.folder(session)
            planned = api.plan_update(self.host.session(folder))
            writes = [str(folder)] if planned else []
            if dry_run:
                return self.operation(True, would_run=planned, writes=writes)
            ran = self.act(ctx, folder, "update", api.update, session="read")
            try:
                held = api.outcome(self.host.session(folder)).model_dump()
            except (OSError, ValueError, LookupError, KeyError) as exc:
                logger.warning("No outcome for %s: %s", self.roots.relative(folder), exc)
                held = None
            return self.operation(False, would_run=planned, writes=writes if ran else [],
                                  result={"stages_run": list(ran or []), "outcome": held})

    def detect_channel(self, ctx: Context, session: str, round: Round, channel: Channel,
                       params: Annotated[Optional[dict[str, Any]], Field(description=(
                           "Detection settings (method, threshold_rel, min_distance, sigma, ...); omit for the "
                           "channel's recorded settings."))] = None,
                       existing: Annotated[Literal["replace", "replace_all", "merge"], Field(description=(
                           "replace: keep puncta placed by hand; replace_all: drop them; merge: add to them."))
                       ] = "replace",
                       dry_run: DryRun = True) -> Operation:
        """Detect a channel's puncta again on its raw image, into its raw-space CSV. Its
        aligned CSV and the report are out of date until transformed and exported again."""
        with self.answer():
            if params is not None:
                DetectionParams.model_validate(params)
            folder = self.folder(session)
            plan = api.plan_operation(self.host.session(folder), "detect", round, channel)
            if dry_run:
                return self.operation(True, plan)
            result = self.act(ctx, folder, f"detect {round} - {api.round_channel(round, channel)}", lambda s: api.detect_channel(
                s, round, channel, params, existing))
            return self.operation(False, plan, result=result)

    def transform_channel(self, ctx: Context, session: str, round: Round, channel: Channel,
                          dry_run: DryRun = True) -> Operation:
        """Warp a channel's raw-space puncta into aligned space with the saved registration,
        and assign them to nuclei. The report is out of date until exported again."""
        with self.answer():
            folder = self.folder(session)
            plan = api.plan_operation(self.host.session(folder), "transform", round, channel)
            if dry_run:
                return self.operation(True, plan)
            result = self.act(ctx, folder, f"transform {round} - {api.round_channel(round, channel)}", lambda s: api.transform_channel(
                s, round, channel))
            return self.operation(False, plan, result=result)

    def export_report(self, ctx: Context, session: str, dry_run: DryRun = True) -> Operation:
        """Write the session's colocalization report, and its tables, from its files and its
        colocalization rules."""
        with self.answer():
            folder = self.folder(session)
            plan = api.plan_operation(self.host.session(folder), "report")
            if dry_run:
                return self.operation(True, plan)
            result = self.act(ctx, folder, "export_report", api.export_report)
            return self.operation(False, plan, result=result)

    def export_cells(self, ctx: Context, session: str,
                     ids: Annotated[Optional[list[int]], Field(description="Nucleus IDs; omit for every nucleus.")] = None,
                     padding_um: Annotated[float, Field(ge=0, le=50, description=(
                         "Margin around each nucleus, in micrometres."))] = 2.0,
                     dry_run: DryRun = True) -> Operation:
        """Write one OME-TIFF per nucleus (its aligned channels and mask) with its puncta."""
        with self.answer():
            folder = self.folder(session)
            plan = api.plan_operation(self.host.session(folder), "cells")
            if dry_run:
                return self.operation(True, plan)
            result = self.act(ctx, folder, "export_cells", lambda s: api.export_cells(
                s, ids=ids, padding_um=padding_um))
            return self.operation(False, plan, result=result)

    def threshold_sweep(self, ctx: Context, session: str, round: Round, channel: Channel,
                        params: Annotated[Optional[dict[str, Any]], Field(description=(
                            "The other detection settings; omit for the channel's recorded ones."))] = None
                        ) -> api.ThresholdSweep:
        """Puncta counts of a channel over a range of sensitivities and where they level off,
        with a suggested sensitivity. Stores the curve in the session; detects nothing for keeps."""
        with self.answer():
            if params is not None:
                DetectionParams.model_validate(params)
            folder = self.folder(session)
            result = self.act(ctx, folder, f"threshold_sweep {round} - {api.round_channel(round, channel)}", lambda s: api.threshold_sweep(
                s, round, channel, params))
            return self.out(api.ThresholdSweep, result)

    def run_batch(self, ctx: Context,
                  template: Annotated[str, Field(description="The batch template (.xlsx), below the root.")],
                  output: Annotated[str, Field(description="The folder for the batch's results, below the root.")],
                  workers: Annotated[int, Field(ge=1, description=(
                      "Datasets at once; the server caps it."))] = 1,
                  dry_run: DryRun = True) -> Operation:
        """Run a batch template's datasets through the whole pipeline, then check the batch
        (as flag_batch) and write qc_report.md in the output folder. A dry run lists each dataset
        and the stages it would run; a dataset finished before runs only what is due. In a real
        run's result, ok means a dataset's run finished without error; its flags are the batch
        check's, and needs_look is set when it failed or was flagged."""
        with self.answer():
            template_path, output_path = self.roots.resolve(template), self.roots.resolve(output)
            config = api.read_batch_template(template_path)
            for ds in config.datasets:
                for what, path in (("R1 image", ds.r1), ("R2 image", ds.r2),
                                   ("output folder", ds.output_dir or output_path / ds.name)):
                    if self.roots.relative(Path(path).resolve()) is None:
                        raise ValueError(f"Dataset '{ds.name}': its {what} is outside the folders this server "
                                         "may use.")
            writes = [str(output_path), str(output_path / QC_REPORT)]
            if dry_run:
                return self.operation(True, result=api.plan_batch(config, output_path), writes=writes)
            workers = min(workers, self.config.max_workers)
            result = self.act(ctx, output_path, f"run_batch with {workers} worker(s)", lambda: api.run_batch(
                config, output_path, workers=workers), session=None)
            return self.operation(False, result=result, writes=writes if result.qc_report else writes[:1])
