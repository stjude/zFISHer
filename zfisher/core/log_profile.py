"""A one-page profile of zFISHer runs from their session logs.

Every session log carries the timing lines of ``core.timing``: ``STAGE`` for
each pipeline step (wall-clock and memory), ``ACTION`` for each GUI action and
deferred save, ``STALL`` for each main-thread block over 100 ms, and ``ENV``
for the machine. ``profile(paths)`` reads the logs under ``paths`` and
``format_markdown`` writes the profile scope plan M1 asks for: minutes per
stage per dataset, seconds per manual action, the worst main-thread blocks,
the machines, and what the logs say about how zFISHer is used (batch or GUI,
how much is curated by hand). Reads only; ``zfisher-cli profile`` runs it.
"""
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .timing import parse_timing_line

_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_OUTPUT_DIR = re.compile(r"Output directory: (.+?)\s*$")

# GUI actions that are a person curating: edits, picks, deletions, undo.
_CURATION = re.compile(r"(puncta_save|puncta_data_changed|mask_save|mask_edit|mask_ids_refresh|"
                       r"delete|merge|paint|extrude|undo|fishing|add_point|pick)", re.IGNORECASE)


@dataclass
class LogRun:
    """What one session log holds."""
    path: Path
    output_dir: str = ""
    start: str = ""
    end: str = ""
    env: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)        # (name, seconds, peak_rss_mb, status)
    actions: list = field(default_factory=list)       # (name, seconds, status)
    stalls: list = field(default_factory=list)        # (blocked_ms, during, timestamp, app_active)

    @property
    def kind(self):
        """'batch' for a run of the whole pipeline without the GUI, else 'gui'."""
        widget = any(name.startswith("widget:") for name, _, _ in self.actions)
        return "gui" if widget else ("batch" if any(s[0] == "pipeline_total" for s in self.stages) else "gui")


def _float(text, default=float("nan")):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def read_log(path):
    """The ``LogRun`` of the session log at ``path``."""
    run = LogRun(Path(path))
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            stamp = _TIMESTAMP.match(line)
            if stamp:
                run.start = run.start or stamp.group(1)
                run.end = stamp.group(1)
            folder = _OUTPUT_DIR.search(line)
            if folder and not run.output_dir:
                run.output_dir = folder.group(1)
            parsed = parse_timing_line(line)
            if parsed is None:
                continue
            kind, fields = parsed
            if kind == "STAGE":
                run.stages.append((fields.get("stage", "?"), _float(fields.get("duration_s")),
                                   _float(fields.get("peak_rss_mb")), fields.get("status", "ok")))
            elif kind == "ACTION":
                run.actions.append((fields.get("action", "?"), _float(fields.get("duration_s")),
                                    fields.get("status", "ok")))
            elif kind == "STALL":
                run.stalls.append((_float(fields.get("blocked_ms"), 0.0), fields.get("during", "?"),
                                   stamp.group(1) if stamp else "", fields.get("app_active", "1") != "0"))
            elif kind == "ENV":
                run.env = fields
    return run


def find_logs(paths):
    """Every session log under ``paths``: files as given, folders searched for
    ``zfisher_*.log`` (the name session logs are written under)."""
    found = []
    for p in map(Path, paths):
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.rglob("zfisher_*.log")))
    return sorted(set(found))


def profile(paths):
    """The ``LogRun`` of every session log under ``paths`` that has timing lines."""
    runs = [read_log(p) for p in find_logs(paths)]
    return [r for r in runs if r.stages or r.actions or r.stalls]


def _quantile(values, q):
    values = sorted(v for v in values if v == v)
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def _summary(values):
    values = [v for v in values if v == v]
    return len(values), statistics.median(values) if values else float("nan"), _quantile(values, 0.9), \
        max(values) if values else float("nan"), sum(values)


def format_markdown(runs, worst=10):
    """The profile of ``runs`` as Markdown."""
    if not runs:
        return "# zFISHer profile\n\nNo session logs with timing lines were found.\n"
    lines = ["# zFISHer profile", ""]
    starts = sorted(r.start for r in runs if r.start)
    datasets = {r.output_dir or str(r.path.parent) for r in runs}
    batch = [r for r in runs if r.kind == "batch"]
    gui = [r for r in runs if r.kind == "gui"]
    lines += [f"{len(runs)} session logs, {len(datasets)} output folders, "
              f"{starts[0] if starts else '?'} to {max(r.end for r in runs)}. "
              f"{len(batch)} batch or CLI runs, {len(gui)} GUI sessions.", ""]

    lines += ["## Machines", "", "| OS | CPU | cores | RAM GB | GPU | CUDA | output drive | write MB/s | logs |",
              "|---|---|---|---|---|---|---|---|---|"]
    machines = {}
    for r in runs:
        e = r.env
        key = tuple(e.get(k, "?") for k in ("os", "cpu", "cores", "ram_gb", "gpu", "torch_cuda", "out_drive",
                                              "write_mb_s"))
        machines[key] = machines.get(key, 0) + 1
    for key, n in sorted(machines.items(), key=lambda kv: -kv[1]):
        lines.append("| " + " | ".join(key) + f" | {n} |")
    lines.append("")

    if batch:
        lines += ["## Minutes per dataset (batch and CLI runs)", "",
                  "| output folder | minutes | peak memory GB |", "|---|---|---|"]
        for r in sorted(batch, key=lambda r: r.start):
            total = [s for s in r.stages if s[0] == "pipeline_total"]
            minutes = sum(s[1] for s in total) / 60
            peak = max((s[2] for s in r.stages if s[2] == s[2]), default=float("nan")) / 1024
            lines.append(f"| {Path(r.output_dir).name or r.path.parent.name} | {minutes:.1f} | {peak:.1f} |")
        lines.append("")

    lines += ["## Pipeline stages", "", "Seconds per run of each stage, across all logs, by total time.", "",
              "| stage | runs | median s | p90 s | max s | total min | peak memory GB | failed |",
              "|---|---|---|---|---|---|---|---|"]
    by_stage = {}
    for r in runs:
        for name, sec, peak, status in r.stages:
            by_stage.setdefault(name, []).append((sec, peak, status))
    for name, rows in sorted(by_stage.items(), key=lambda kv: -sum(s for s, _, _ in kv[1] if s == s)):
        n, med, p90, mx, total = _summary([s for s, _, _ in rows])
        peak = max((p for _, p, _ in rows if p == p), default=float("nan")) / 1024
        failed = sum(1 for _, _, st in rows if st != "ok")
        lines.append(f"| {name} | {n} | {med:.1f} | {p90:.1f} | {mx:.1f} | {total / 60:.1f} | {peak:.1f} | {failed} |")
    lines.append("")

    actions = {}
    for r in runs:
        for name, sec, status in r.actions:
            actions.setdefault(name, []).append(sec)
    if actions:
        lines += ["## GUI actions", "", "Seconds per action, by total time.", "",
                  "| action | count | median s | p90 s | max s | total s |", "|---|---|---|---|---|---|"]
        for name, secs in sorted(actions.items(), key=lambda kv: -sum(v for v in kv[1] if v == v)):
            n, med, p90, mx, total = _summary(secs)
            lines.append(f"| {name} | {n} | {med:.2f} | {p90:.2f} | {mx:.2f} | {total:.1f} |")
        lines.append("")

    background = [ms for r in runs for ms, _, _, active in r.stalls if not active]
    stalls = [(ms, during, ts, r) for r in runs for ms, during, ts, active in r.stalls if active]
    if stalls or background:
        total_s = sum(ms for ms, *_ in stalls) / 1000
        lines += ["## Main-thread blocks", "",
                  f"{len(stalls)} blocks over 100 ms while zFISHer was the active app, {total_s:.0f} s in all. "
                  f"Another {len(background)} ({sum(background) / 1000:.0f} s) came while it was in the "
                  f"background, where the system may slow an idle app's timers, so they are left out below.",
                  "", "During:", "",
                  "| during | blocks | total s | worst s |", "|---|---|---|---|"]
        by_during = {}
        for ms, during, _, _ in stalls:
            by_during.setdefault(during, []).append(ms / 1000)
        for during, secs in sorted(by_during.items(), key=lambda kv: -sum(kv[1]))[:worst]:
            lines.append(f"| {during} | {len(secs)} | {sum(secs):.1f} | {max(secs):.1f} |")
        lines += ["", f"Worst {min(worst, len(stalls))}:", "", "| s | during | when | log |", "|---|---|---|---|"]
        for ms, during, ts, r in sorted(stalls, key=lambda s: -s[0])[:worst]:
            lines.append(f"| {ms / 1000:.1f} | {during} | {ts} | {r.path.name} |")
        lines.append("")

    curation = sum(1 for r in gui for name, _, _ in r.actions if _CURATION.search(name))
    lines += ["## Usage", "",
              f"- {len(batch)} runs without the GUI, {len(gui)} GUI sessions.",
              f"- {curation} logged GUI actions that are hand curation (puncta and mask edits, saves, undo), "
              f"in {sum(1 for r in gui if any(_CURATION.search(a[0]) for a in r.actions))} sessions.",
              "- Who runs zFISHer and on which datasets is not in the logs; ask the lab.", ""]
    return "\n".join(lines)
