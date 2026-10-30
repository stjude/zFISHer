"""Command-line interface for zFISHer: ``zfisher-cli``.

Runs the headless pipeline and the session operations of ``zfisher.core.api``
without a napari viewer:

    zfisher-cli --help
    zfisher-cli run R1.nd2 R2.nd2 -o ./output
    zfisher-cli batch template.xlsx -o ./batch_output
    zfisher-cli template ./my_template.xlsx
    zfisher-cli status ./output [--json]
    zfisher-cli update ./output
    zfisher-cli export-cells ./batch_output --padding 2
    zfisher-cli export ./output [--rule "R1 - FITC,R2 - Cy5,1.0"]
    zfisher-cli nucleus-ids ./output [--write]
    zfisher-cli profile D:/zFISHer_runs -o profile.md

``python -m zfisher.cli`` does the same.
"""
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .core.params import ConsensusParams, OverlapMethod, SegMethod

app = typer.Typer(help="Headless CLI for the zFISHer sequential FISH analysis pipeline.",
                  add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


def _open(folder, read_only=True):
    """The session in ``folder`` (``api.open_session``); exits with a message
    instead of a traceback when there is none or it cannot be read."""
    from .core import api
    try:
        return api.open_session(Path(folder).expanduser().resolve(), read_only=read_only)
    except (FileNotFoundError, ValueError, PermissionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)


def _stdout_progress(pct, msg):
    """Progress callback that writes to stdout. pct can be negative for indeterminate."""
    if pct is None or pct < 0:
        print(f"  [....] {msg}", flush=True)
    else:
        print(f"  [{int(pct):3d}%] {msg}", flush=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable debug logging.")):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s [%(levelname)s] %(message)s")
    # zFISHer's loggers do not propagate to the root logger (see
    # zfisher/__init__.py), so the level has to be set on them as well.
    logging.getLogger("zfisher").setLevel(level)
    from .core import decisions
    decisions.set_process_actor(decisions.HUMAN)   # a person typed the command


@app.command()
def run(r1: Path = typer.Argument(..., help="Round 1 image file (.nd2 or .tif/.ome.tif)"),
        r2: Path = typer.Argument(..., help="Round 2 image file (.nd2 or .tif/.ome.tif)"),
        output: Path = typer.Option(..., "-o", "--output", help="Output directory."),
        seg_method: SegMethod = typer.Option(SegMethod.CLASSICAL, help="Nuclear segmentation backend."),
        no_merge_splits: bool = typer.Option(False, help="Disable over-segmentation merging."),
        r1_nuclear: Optional[str] = typer.Option(None, help="R1 nuclear channel name (e.g. DAPI)."),
        r2_nuclear: Optional[str] = typer.Option(None, help="R2 nuclear channel name."),
        no_warp: bool = typer.Option(False, help="Skip deformable B-spline warping."),
        overlap_method: OverlapMethod = typer.Option(OverlapMethod.INTERSECTION, help="Consensus mask merge mode."),
        keep_extranuclear: bool = typer.Option(False, help="Keep puncta outside segmented nuclei.")):
    """Run the full pipeline on a single R1/R2 pair."""
    from .core import api
    from .core.params import DatasetSpec, RegistrationParams, SegmentationParams

    r1, r2, out = (p.expanduser().resolve() for p in (r1, r2, output))
    for path, label in [(r1, "R1"), (r2, "R2")]:
        if not path.exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            raise typer.Exit(1)
    out.mkdir(parents=True, exist_ok=True)
    print(f"zFISHer pipeline: {r1.name} + {r2.name} -> {out}")
    dataset = DatasetSpec(
        name=out.name, r1=r1, r2=r2, r1_nuclear_channel=r1_nuclear or "", r2_nuclear_channel=r2_nuclear or "",
        segmentation=SegmentationParams(method=seg_method, merge_splits=not no_merge_splits),
        registration=RegistrationParams(apply_warp=not no_warp),
        consensus=ConsensusParams(overlap_method=overlap_method, remove_extranuclear_puncta=not keep_extranuclear))
    try:
        api.run_pipeline(dataset, out, progress_callback=_stdout_progress)
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    print(f"\nDone. Outputs in: {out}")


@app.command()
def batch(template: Path = typer.Argument(..., help="Batch template .xlsx file."),
          output: Path = typer.Option(..., "-o", "--output", help="Batch output root directory."),
          force: bool = typer.Option(False, help="Continue past validation warnings."),
          workers: int = typer.Option(1, min=1, help="Datasets processed at the same time, each in its own "
                                                     "process; bounded by memory.")):
    """Run the pipeline on every dataset of a batch Excel template.

    Running it again over the same output folder skips the stages that are
    up to date (see ``update``)."""
    from .core import batch as batch_runner
    from .core.generate_batch_template import parse_batch_config, validate_channels

    template, out_base = template.expanduser().resolve(), output.expanduser().resolve()
    if not template.exists():
        print(f"Error: template not found: {template}", file=sys.stderr)
        raise typer.Exit(1)
    config, error = parse_batch_config(template)
    if error:
        print(f"Batch validation failed: {error}", file=sys.stderr)
        raise typer.Exit(1)
    warnings = list(config.warnings) + validate_channels(config)
    if warnings:
        print("Warnings during validation:")
        for w in warnings:
            print(f"  - {w}")
        if not force:
            print("Pass --force to continue anyway.", file=sys.stderr)
            raise typer.Exit(1)
    print(f"{len(config.datasets)} dataset(s) -> {out_base}" + (f", {workers} at a time" if workers > 1 else ""))

    def finished(result):
        state = f"done in {result.seconds / 60:.1f} min" if result.ok else f"FAILED - {result.error}"
        print(f"  {result.name}: {state}", flush=True)

    results = batch_runner.run_batch(
        config, out_base, workers=workers, on_result=finished,
        on_progress=(lambda name, p, text: _stdout_progress(p, f"{name}: {text}")) if workers == 1 else None)
    failed = [r for r in results if not r.ok]
    if (out_base / "qc_report.md").exists():
        print(f"\nQC report: {out_base / 'qc_report.md'}")
    print("\n--- Batch summary ---")
    for r in results:
        print(f"  {r.name}: {'OK' if r.ok else 'FAIL: ' + r.error}")
    if failed:
        raise typer.Exit(1)


@app.command()
def template(output: Path = typer.Argument(..., help="Path for the generated .xlsx file."),
             force: bool = typer.Option(False, help="Overwrite if the file exists.")):
    """Generate a blank batch Excel template."""
    from .core import outputs
    from .core.generate_batch_template import add_dropdown_validations, build_template_sheets

    save_path = output.expanduser().resolve()
    if save_path.exists() and not force:
        print(f"Error: {save_path} already exists. Pass --force to overwrite.", file=sys.stderr)
        raise typer.Exit(1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with outputs.excel_writer(save_path) as writer:
        for name, df in build_template_sheets().items():
            df.to_excel(writer, sheet_name=name, index=False)
        add_dropdown_validations(writer.book)
    print(f"Template written to: {save_path}")


@app.command()
def status(folder: Path = typer.Argument(..., help="A session's output folder or session file."),
           as_json: bool = typer.Option(False, "--json", help="Print JSON instead of text.")):
    """Describe a session and say what in it is out of date. Reads only."""
    from .core import api
    s = _open(folder)
    info, states = api.session_info(s), api.get_status(s)
    if as_json:
        print(json.dumps({"session": info.model_dump(), "status": [a.model_dump() for a in states],
                          "stages": [st.model_dump() for st in api.get_stage_status(s)]}, indent=1))
        return
    print(f"Session {info.session_file} in {info.output_dir}")
    print(f"  made with zFISHer {info.created_with_zfisher or 'unknown'}, last saved by {info.zfisher_version or 'unknown'}")
    print(f"  nuclear channel {info.nuclear_channel}; channels with puncta: {', '.join(info.channels) or 'none'}")
    print(f"  consensus mask: {'yes' if info.has_consensus_mask else 'no'}; saved registration: "
          f"{'yes' if info.has_saved_registration else 'no'}; report: {'yes' if info.has_report else 'no'}")
    for a in states:
        line = f"  {a.name}: {a.state}" + (" (edited by hand)" if a.edited else "")
        print(line + (f" - {'; '.join(a.reasons)}" if a.reasons else ""))
    stage_states = api.get_stage_status(s)
    if stage_states:
        print("Pipeline stages:")
        for st in stage_states:
            print(f"  {st.stage}{' ' + st.key if st.key else ''}: {st.state}")


@app.command()
def export(folder: Path = typer.Argument(..., help="A session's output folder or session file."),
           rule: Optional[List[str]] = typer.Option(
               None, help='A pairwise rule "source,target,cutoff in µm", as --rule "R1 - FITC,R2 - Cy5,1.0"; '
                          "repeat for more. Without any, the session's own rules are used.")):
    """Write a session's colocalization report from its files on disk."""
    from .core import api
    rules = None
    if rule:
        rules = []
        for text in rule:
            parts = [part.strip() for part in text.split(",")]
            try:
                rules.append({"source": parts[0], "target": parts[1], "threshold": float(parts[2])})
            except (IndexError, ValueError):
                print(f'Error: a rule is "source,target,cutoff", not "{text}".', file=sys.stderr)
                raise typer.Exit(1)
    s = _open(folder, read_only=False)
    try:
        result = api.export_report(s, rules=rules)
    except (ValueError, FileNotFoundError, LookupError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    print(f"Report written to: {result.path}")
    for name, rows in result.sheets.items():
        print(f"  {name}: {rows} rows")


@app.command()
def metrics(folder: Path = typer.Argument(..., help="A session's output folder or session file."),
            saturation: bool = typer.Option(False, help="Also measure saturated voxels (reads every aligned image)."),
            as_json: bool = typer.Option(False, "--json", help="Print JSON instead of text.")):
    """Quality-control numbers of a session, from its files. Reads only."""
    from .core import api
    m = api.get_metrics(_open(folder), include_saturation=saturation)
    if as_json:
        print(json.dumps(m.model_dump(), indent=1))
        return
    reg = m.registration
    rmsd = f"{reg['rmsd_um']:.2f} µm" if reg.get("rmsd_um") is not None else "not recorded in µm"
    print(f"{m.dataset}: registration {reg.get('method') or 'unknown'}, RMSD {rmsd}, "
          f"{reg.get('n_inliers')} of {reg.get('n_pairs')} pairs inliers")
    vol = m.nuclei.get("volume_um3", {})
    print(f"  nuclei: {m.nuclei['n_nuclei']}" + (f", median volume {vol['p50']:.0f} µm³" if vol else ""))
    for key, c in m.channels.items():
        snr = c["snr"].get("p50")
        print(f"  {key}: {c['n_puncta']} puncta, median SNR {snr:.1f}" if snr is not None else f"  {key}: {c['n_puncta']} puncta",
              f"| low SNR {100 * c['low_snr_fraction']:.0f}%, outside nuclei {100 * c['extranuclear_fraction']:.0f}%,"
              f" by hand {100 * c['manual_fraction']:.0f}%, detected on {c['detected_on'] or 'unknown'}")
    if m.saturation:
        for name, fraction in m.saturation.items():
            print(f"  {name}: {100 * fraction:.3f}% saturated")


@app.command("flag-batch")
def flag_batch(batch_root: Path = typer.Argument(..., help="A batch output root holding one folder per dataset."),
               saturation: bool = typer.Option(False, help="Also check saturated voxels (reads every aligned image)."),
               as_json: bool = typer.Option(False, "--json", help="Print JSON instead of text.")):
    """Flag the datasets of a batch that need a look, and write qc_report.md. Reads the datasets only."""
    from .core import api
    root = batch_root.expanduser().resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a folder.", file=sys.stderr)
        raise typer.Exit(1)
    result = api.flag_batch(root, include_saturation=saturation)
    if as_json:
        print(json.dumps(result.model_dump(), indent=1))
        return
    print(f"{result.n_datasets} datasets, {len({f['dataset'] for f in result.flags})} flagged.")
    for f in result.flags:
        print(f"  {f['dataset']}: {f['action']} - {f['reason']}")
    if result.report:
        print(f"Report: {result.report}")


@app.command("query-distances")
def query_distances(folder: Path = typer.Argument(..., help="A session's output folder or session file."),
                    source: str = typer.Argument(..., help='Source channel: "R1 - FITC", a layer name, or a channel one round has.'),
                    target: str = typer.Argument(..., help="Target channel, named the same ways."),
                    within: float = typer.Option(0.5, help="Distance in µm."),
                    bin_um: Optional[float] = typer.Option(None, "--bin", help="Histogram bin width in µm."),
                    as_json: bool = typer.Option(False, "--json", help="Print JSON instead of text.")):
    """How many source puncta have a target punctum within a distance, from the latest report. Reads only."""
    from .core import api
    try:
        q = api.query_distances(_open(folder), source, target, within, bin_um)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    if as_json:
        print(json.dumps(q.model_dump(), indent=1))
        return
    print(f"{q.n_within} of {q.n_source} {q.source} puncta ({100 * q.fraction_within:.1f}%) have a "
          f"{q.target} punctum within {q.max_um:g} µm.")
    for low, high, count in q.bins:
        print(f"  {low:g}-{high:g} µm: {count}")


@app.command("export-cells")
def export_cells(folder: Path = typer.Argument(..., help="A session's output folder, or a batch root holding "
                                                         "one folder per dataset."),
                 padding: float = typer.Option(2.0, help="Margin around each nucleus, in µm."),
                 ids: Optional[List[int]] = typer.Option(None, "--id", help="Only this nucleus; repeat for more."),
                 no_puncta: bool = typer.Option(False, help="Do not write each nucleus's puncta CSV.")):
    """Write one OME-TIFF per nucleus (all aligned channels and its mask) into cells/."""
    from . import constants
    from .core import api
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a folder.", file=sys.stderr)
        raise typer.Exit(1)
    targets = [folder] if (folder / constants.SESSIONS_DIR).exists() else sorted(
        p for p in folder.iterdir() if (p / constants.SESSIONS_DIR).exists())
    if not targets:
        print(f"Error: no session in {folder} or its subfolders.", file=sys.stderr)
        raise typer.Exit(1)
    failed = False
    for target in targets:
        try:
            result = api.export_cells(_open(target), ids=ids or None, padding_um=padding, with_puncta=not no_puncta)
            print(f"{target.name}: {result.n_cells} nuclei -> {result.folder}")
        except (FileNotFoundError, LookupError, ValueError) as exc:
            print(f"{target.name}: Error: {exc}", file=sys.stderr)
            failed = True
    if failed:
        raise typer.Exit(1)


@app.command()
def update(folder: Path = typer.Argument(..., help="A folder a pipeline run made.")):
    """Run again the pipeline stages of a folder that are not up to date."""
    from .core import api
    s = _open(folder)
    try:
        ran = api.update(s, progress_callback=_stdout_progress)
    except (LookupError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise typer.Exit(1)
    print("Everything was up to date." if not ran else "Ran: " + ", ".join(ran))


@app.command("nucleus-ids")
def nucleus_ids(folder: Path = typer.Argument(..., help="A session's output folder (with segmentation/ and reports/)."),
                write: bool = typer.Option(False, help="Write repaired copies (*.nucleus_ids_repaired.csv) beside "
                                                       "the originals.")):
    """Check an output folder's puncta Nucleus_ID against its saved consensus mask."""
    from . import constants
    from .core import io, puncta

    folder = folder.expanduser().resolve()
    mask_path = io.find_mask(folder / constants.SEGMENTATION_DIR, constants.CONSENSUS_MASKS_NAME)
    if mask_path is None:
        print(f"Error: no {constants.CONSENSUS_MASKS_NAME} mask in {folder / constants.SEGMENTATION_DIR}",
              file=sys.stderr)
        raise typer.Exit(1)
    csvs = puncta.aligned_puncta_csvs(folder / constants.REPORTS_DIR)
    if not csvs:
        print(f"Error: no aligned or warped puncta CSVs in {folder / constants.REPORTS_DIR}", file=sys.stderr)
        raise typer.Exit(1)
    mask = io.read_label_tif(mask_path)
    print(f"Consensus mask: {mask_path.name} ({len(set(mask.ravel().tolist()) - {0})} nuclei)")
    all_agree = True
    for csv in csvs:
        if write:
            out = csv.with_name(csv.stem + ".nucleus_ids_repaired.csv")
            result = puncta.repair_nucleus_ids(csv, mask, out)
        else:
            result = puncta.check_nucleus_ids(csv, mask)
        all_agree &= result["agree"] == result["rows"]
        line = (f"  {result['file']}: {result['agree']} of {result['rows']} nucleus IDs agree with the mask "
                f"({100 * result['fraction']:.1f}%)")
        if result["zero_but_inside"]:
            line += f"; {result['zero_but_inside']} stored as 0 but inside a nucleus"
        if result["repeated_ids"]:
            line += f"; {len(result['repeated_ids'])} puncta_id values repeated"
        if write:
            line += f" -> {out.name}"
        print(line)
    if write:
        print("Originals are unchanged. Each .nucleus_ids_repaired.csv has every Nucleus_ID read from the mask.")
    elif not all_agree:
        print("Run again with --write to write repaired copies beside the originals.")
        raise typer.Exit(2)


@app.command()
def profile(paths: List[Path] = typer.Argument(..., help="Log files, or folders to search for zfisher_*.log files."),
            output: Optional[Path] = typer.Option(None, "-o", "--output",
                                                  help="Write the profile to this Markdown file instead of printing it.")):
    """Summarise run times, GUI actions and stalls from session logs (read only)."""
    from .core import log_profile
    runs = log_profile.profile(paths)
    text = log_profile.format_markdown(runs)
    if output:
        out = output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Profile of {len(runs)} session logs written to: {out}")
    else:
        print(text)
    if not runs:
        raise typer.Exit(1)


def main(argv=None):
    """Run the CLI with ``argv`` (default: the command line); returns the exit code."""
    try:
        # Without standalone mode, click returns a command's exit code (typer.Exit) instead of exiting.
        code = app(args=list(sys.argv[1:] if argv is None else argv), prog_name="zfisher-cli",
                   standalone_mode=False)
        return code if isinstance(code, int) else 0
    except typer.Exit as exc:
        return exc.exit_code
    except SystemExit as exc:                 # click's own exits (--help, usage errors)
        return exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        # typer carries its own copy of click, so its usage errors are told by shape.
        if hasattr(exc, "show") and isinstance(getattr(exc, "exit_code", None), int):
            exc.show()
            return exc.exit_code
        if type(exc).__name__ == "Abort":
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
