"""Command-line interface for zFISHer.

Exposes the headless core pipeline for scripted and batch execution without
requiring a napari viewer. Invoke as:

    python -m zfisher.cli --help
    python -m zfisher.cli run R1.nd2 R2.nd2 -o ./output
    python -m zfisher.cli batch template.xlsx -o ./batch_output
    python -m zfisher.cli template ./my_template.xlsx
"""
import argparse
import logging
import sys
from pathlib import Path


def _stdout_progress(pct, msg):
    """Progress callback that writes to stdout. pct can be negative for indeterminate."""
    if pct is None or pct < 0:
        print(f"  [....] {msg}", flush=True)
    else:
        print(f"  [{int(pct):3d}%] {msg}", flush=True)


def _cmd_run(args):
    """Run the full pipeline on a single R1/R2 pair."""
    from .core import pipeline

    r1 = Path(args.r1).expanduser().resolve()
    r2 = Path(args.r2).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()

    for path, label in [(r1, "R1"), (r2, "R2")]:
        if not path.exists():
            print(f"Error: {label} file not found: {path}", file=sys.stderr)
            return 1

    out.mkdir(parents=True, exist_ok=True)
    print(f"zFISHer pipeline: {r1.name} + {r2.name} -> {out}")

    try:
        pipeline.run_full_zfisher_pipeline(
            r1_path=r1,
            r2_path=r2,
            output_dir=out,
            seg_method=args.seg_method,
            merge_splits=not args.no_merge_splits,
            r1_nuclear_channel=args.r1_nuclear,
            r2_nuclear_channel=args.r2_nuclear,
            apply_warp=not args.no_warp,
            overlap_method=args.overlap_method,
            remove_extranuclear_puncta=not args.keep_extranuclear,
            progress_callback=_stdout_progress,
        )
    except Exception as exc:
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Outputs in: {out}")
    return 0


def _cmd_batch(args):
    """Run the pipeline on every dataset in a batch Excel template."""
    from .core import pipeline
    from .core.generate_batch_template import (
        parse_batch_config,
        validate_channels,
        resolve_puncta_for_dataset,
        resolve_coloc_for_dataset,
    )

    template = Path(args.template).expanduser().resolve()
    out_base = Path(args.output).expanduser().resolve()

    if not template.exists():
        print(f"Error: template not found: {template}", file=sys.stderr)
        return 1

    config, error = parse_batch_config(template)
    if error:
        print(f"Batch validation failed: {error}", file=sys.stderr)
        return 1

    warnings = config.pop("_warnings", [])
    warnings.extend(validate_channels(config))
    if warnings:
        print("Warnings during validation:")
        for w in warnings:
            print(f"  - {w}")
        if not args.force:
            print("Pass --force to continue anyway.", file=sys.stderr)
            return 1

    out_base.mkdir(parents=True, exist_ok=True)

    datasets = config["datasets"]
    results = []
    for i, ds in enumerate(datasets, 1):
        name = ds["name"]
        item_out = ds["output_dir"] or (out_base / name)
        print(f"\n[{i}/{len(datasets)}] {name} -> {item_out}")

        puncta_cfg = resolve_puncta_for_dataset(name, config["puncta_rules"])
        pairwise, tri = resolve_coloc_for_dataset(name, config["coloc_rules"])

        try:
            pipeline.run_full_zfisher_pipeline(
                r1_path=ds["r1"],
                r2_path=ds["r2"],
                output_dir=item_out,
                seg_method=ds["seg_method"],
                merge_splits=ds["merge_splits"],
                r1_nuclear_channel=ds["r1_nuclear_channel"],
                r2_nuclear_channel=ds["r2_nuclear_channel"],
                puncta_config=puncta_cfg,
                pairwise_rules=pairwise,
                tri_rules=tri,
                apply_warp=ds.get("apply_warp", True),
                max_ransac_distance=ds.get("max_ransac_distance", 0),
                overlap_method=ds.get("overlap_method", "Intersection"),
                match_threshold=ds.get("match_threshold", 0),
                remove_extranuclear_puncta=ds.get("remove_extranuclear_puncta", True),
                progress_callback=_stdout_progress,
            )
            results.append((name, "OK"))
        except Exception as exc:
            logging.error("Dataset '%s' failed: %s", name, exc, exc_info=True)
            results.append((name, f"FAIL: {exc}"))

    print("\n--- Batch summary ---")
    for name, status in results:
        print(f"  {name}: {status}")
    return 0 if all(s == "OK" for _, s in results) else 1


def _cmd_template(args):
    """Generate a batch processing Excel template."""
    import pandas as pd
    from .core.generate_batch_template import build_template_sheets, add_dropdown_validations

    save_path = Path(args.output).expanduser().resolve()
    if save_path.exists() and not args.force:
        print(f"Error: {save_path} already exists. Pass --force to overwrite.", file=sys.stderr)
        return 1

    save_path.parent.mkdir(parents=True, exist_ok=True)
    sheets = build_template_sheets()
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        add_dropdown_validations(writer.book)

    print(f"Template written to: {save_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m zfisher.cli",
        description="Headless CLI for the zFISHer sequential FISH analysis pipeline.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run the full pipeline on a single R1/R2 pair.")
    p_run.add_argument("r1", help="Round 1 image file (.nd2 or .tif/.ome.tif)")
    p_run.add_argument("r2", help="Round 2 image file (.nd2 or .tif/.ome.tif)")
    p_run.add_argument("-o", "--output", required=True, help="Output directory.")
    p_run.add_argument("--seg-method", default="Classical",
                       choices=["Classical", "Cellpose"], help="Nuclear segmentation backend.")
    p_run.add_argument("--no-merge-splits", action="store_true",
                       help="Disable over-segmentation merging.")
    p_run.add_argument("--r1-nuclear", default=None, help="R1 nuclear channel name (e.g. DAPI).")
    p_run.add_argument("--r2-nuclear", default=None, help="R2 nuclear channel name.")
    p_run.add_argument("--no-warp", action="store_true", help="Skip deformable B-spline warping.")
    p_run.add_argument("--overlap-method", default="Intersection",
                       choices=["Intersection", "Union"], help="Consensus mask merge mode.")
    p_run.add_argument("--keep-extranuclear", action="store_true",
                       help="Keep puncta outside segmented nuclei.")
    p_run.set_defaults(func=_cmd_run)

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="Run the pipeline on a batch Excel template.")
    p_batch.add_argument("template", help="Batch template .xlsx file.")
    p_batch.add_argument("-o", "--output", required=True, help="Batch output root directory.")
    p_batch.add_argument("--force", action="store_true",
                         help="Continue past validation warnings.")
    p_batch.set_defaults(func=_cmd_batch)

    # --- template ---
    p_tmpl = subparsers.add_parser("template", help="Generate a blank batch Excel template.")
    p_tmpl.add_argument("output", help="Path for the generated .xlsx file.")
    p_tmpl.add_argument("--force", action="store_true", help="Overwrite if the file exists.")
    p_tmpl.set_defaults(func=_cmd_template)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
    )

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
