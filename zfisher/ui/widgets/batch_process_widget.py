from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import os
import logging
import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QFrame, QFileDialog

from ...core import pipeline
from ... import constants
from .. import popups
from ..decorators import error_handler
from ..style import COLORS

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.nd2', '.tif', '.tiff'}

# Valid values for template columns
_VALID_SEG_METHODS = {"Classical", "Cellpose"}
_VALID_PUNCTA_ALGORITHMS = {
    "Local Maxima", "Laplacian of Gaussian", "Difference of Gaussian",
    "Radial Symmetry",
}
_VALID_COLOC_TYPES = {"pairwise", "tri"}


# ------------------------------------------------------------------
# Template helpers
# ------------------------------------------------------------------

def _add_dropdown_validations(workbook):
    """Add Excel data-validation dropdowns to columns with fixed choices."""
    from openpyxl.worksheet.datavalidation import DataValidation

    # Max rows to apply validation (generous upper bound)
    MAX_ROW = 500

    # --- Datasets sheet ---
    ws = workbook["Datasets"]
    # Seg_Method (column E)
    seg_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(sorted(_VALID_SEG_METHODS)) + '"',
        allow_blank=True,
    )
    seg_dv.error = "Invalid segmentation method."
    seg_dv.errorTitle = "Seg_Method"
    seg_dv.prompt = "Choose a segmentation method."
    seg_dv.promptTitle = "Seg_Method"
    ws.add_data_validation(seg_dv)
    seg_dv.add(f"E2:E{MAX_ROW}")

    # Merge_Splits (column F)
    bool_dv = DataValidation(
        type="list", formula1='"TRUE,FALSE"', allow_blank=True,
    )
    ws.add_data_validation(bool_dv)
    bool_dv.add(f"F2:F{MAX_ROW}")

    # --- Puncta sheet ---
    ws = workbook["Puncta"]
    # Algorithm (column C)
    algo_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(sorted(_VALID_PUNCTA_ALGORITHMS)) + '"',
        allow_blank=True,
    )
    algo_dv.error = "Invalid puncta algorithm."
    algo_dv.errorTitle = "Algorithm"
    algo_dv.prompt = "Choose a detection algorithm."
    algo_dv.promptTitle = "Algorithm"
    ws.add_data_validation(algo_dv)
    algo_dv.add(f"C2:C{MAX_ROW}")

    # Nuclei_Only (column G)
    nuc_dv = DataValidation(
        type="list", formula1='"TRUE,FALSE"', allow_blank=True,
    )
    ws.add_data_validation(nuc_dv)
    nuc_dv.add(f"G2:G{MAX_ROW}")

    # Tophat (column H)
    tophat_dv = DataValidation(
        type="list", formula1='"TRUE,FALSE"', allow_blank=True,
    )
    ws.add_data_validation(tophat_dv)
    tophat_dv.add(f"H2:H{MAX_ROW}")

    # --- Colocalization sheet ---
    ws = workbook["Colocalization"]
    # Type (column B)
    type_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(sorted(_VALID_COLOC_TYPES)) + '"',
        allow_blank=True,
    )
    type_dv.error = "Invalid colocalization type."
    type_dv.errorTitle = "Type"
    type_dv.prompt = "Choose pairwise or tri."
    type_dv.promptTitle = "Type"
    ws.add_data_validation(type_dv)
    type_dv.add(f"B2:B{MAX_ROW}")


def _build_template_sheets():
    """Return an {sheet_name: DataFrame} dict for the batch template."""
    datasets = pd.DataFrame({
        "Name": pd.Series(dtype=str),
        "R1": pd.Series(dtype=str),
        "R2": pd.Series(dtype=str),
        "Output_Dir": pd.Series(dtype=str),
        "Seg_Method": pd.Series(dtype=str),
        "Merge_Splits": pd.Series(dtype=bool),
    })

    puncta = pd.DataFrame({
        "Dataset": ["ALL"],
        "Channel": [""],
        "Algorithm": ["Local Maxima"],
        "Sensitivity": [constants.PUNCTA_THRESHOLD_REL],
        "Min_Distance": [constants.PUNCTA_MIN_DISTANCE],
        "Sigma": [constants.PUNCTA_SIGMA],
        "Nuclei_Only": [True],
        "Tophat": [False],
        "Tophat_Radius": [constants.PUNCTA_TOPHAT_RADIUS],
    })

    colocalization = pd.DataFrame({
        "Dataset": pd.Series(dtype=str),
        "Type": pd.Series(dtype=str),
        "Source": pd.Series(dtype=str),
        "Target": pd.Series(dtype=str),
        "Channel_B": pd.Series(dtype=str),
        "Cutoff_um": pd.Series(dtype=float),
    })

    return {"Datasets": datasets, "Puncta": puncta, "Colocalization": colocalization}


def _parse_batch_config(excel_path):
    """Parse the multi-sheet batch Excel into a structured config dict.

    Returns
    -------
    (config, None) on success, or (None, error_string) on failure.

    config = {
        'datasets': [
            {'name': str, 'r1': Path, 'r2': Path, 'output_dir': Path|None,
             'seg_method': str, 'merge_splits': bool},
            ...
        ],
        'puncta_rules': [
            {'dataset': str, 'channel': str, 'params': dict},
            ...
        ],
        'coloc_rules': [
            {'dataset': str, 'type': str, 'source': str, 'target': str,
             'channel_b': str|None, 'cutoff': float},
            ...
        ],
    }
    """
    try:
        sheets = pd.read_excel(excel_path, sheet_name=None)
    except Exception as exc:
        return None, f"Could not read Excel file:\n{exc}"

    # --- Datasets sheet (required) ---
    if "Datasets" not in sheets:
        return None, "Missing required sheet: 'Datasets'"
    ds_df = sheets["Datasets"]

    required_cols = {"Name", "R1", "R2"}
    missing = required_cols - set(ds_df.columns)
    if missing:
        return None, f"Datasets sheet missing columns: {', '.join(sorted(missing))}"
    if ds_df.empty:
        return None, "Datasets sheet has no data rows."

    errors = []
    datasets = []
    dataset_names = set()
    for idx, row in ds_df.iterrows():
        row_num = idx + 2
        name = str(row.get("Name", "")).strip()
        if not name:
            errors.append(f"Datasets row {row_num}: Name is empty.")
            continue
        if name in dataset_names:
            errors.append(f"Datasets row {row_num}: Duplicate name '{name}'.")
        dataset_names.add(name)

        for col in ("R1", "R2"):
            raw = str(row.get(col, "")).strip()
            if not raw:
                errors.append(f"Datasets row {row_num} ({name}): {col} path is empty.")
                continue
            p = Path(raw)
            if not p.is_file():
                errors.append(f"Datasets row {row_num} ({name}): {col} file not found:\n  {p}")
            elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                errors.append(
                    f"Datasets row {row_num} ({name}): {col} unsupported type "
                    f"({p.suffix}). Expected: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )

        seg = str(row.get("Seg_Method", "Classical")).strip()
        if seg and seg not in _VALID_SEG_METHODS:
            errors.append(
                f"Datasets row {row_num} ({name}): Invalid Seg_Method '{seg}'. "
                f"Use: {', '.join(sorted(_VALID_SEG_METHODS))}"
            )

        merge = row.get("Merge_Splits", True)
        if pd.isna(merge):
            merge = True
        else:
            merge = bool(merge)

        out_dir_raw = str(row.get("Output_Dir", "")).strip()
        out_dir = Path(out_dir_raw) if out_dir_raw and out_dir_raw != "nan" else None

        datasets.append({
            "name": name,
            "r1": Path(str(row["R1"]).strip()),
            "r2": Path(str(row["R2"]).strip()),
            "output_dir": out_dir,
            "seg_method": seg if seg else "Classical",
            "merge_splits": merge,
        })

    # --- Puncta sheet (optional) ---
    puncta_rules = []
    if "Puncta" in sheets:
        p_df = sheets["Puncta"]
        for idx, row in p_df.iterrows():
            row_num = idx + 2
            ds = str(row.get("Dataset", "ALL")).strip()
            if not ds or ds == "nan":
                ds = "ALL"
            ch = str(row.get("Channel", "")).strip()
            if not ch or ch == "nan":
                continue  # skip rows with no channel

            if ds != "ALL" and ds not in dataset_names:
                errors.append(f"Puncta row {row_num}: Dataset '{ds}' not found in Datasets sheet.")

            algo = str(row.get("Algorithm", "Local Maxima")).strip()
            if algo and algo not in _VALID_PUNCTA_ALGORITHMS:
                errors.append(
                    f"Puncta row {row_num}: Invalid Algorithm '{algo}'. "
                    f"Use: {', '.join(sorted(_VALID_PUNCTA_ALGORITHMS))}"
                )

            def _float(col, default):
                v = row.get(col, default)
                return default if pd.isna(v) else float(v)

            def _int(col, default):
                v = row.get(col, default)
                return default if pd.isna(v) else int(v)

            def _bool(col, default):
                v = row.get(col, default)
                return default if pd.isna(v) else bool(v)

            puncta_rules.append({
                "dataset": ds,
                "channel": ch,
                "params": {
                    "method": algo if algo else "Local Maxima",
                    "threshold_rel": _float("Sensitivity", constants.PUNCTA_THRESHOLD_REL),
                    "min_distance": _int("Min_Distance", constants.PUNCTA_MIN_DISTANCE),
                    "sigma": _float("Sigma", constants.PUNCTA_SIGMA),
                    "nuclei_only": _bool("Nuclei_Only", True),
                    "use_tophat": _bool("Tophat", False),
                    "tophat_radius": _int("Tophat_Radius", constants.PUNCTA_TOPHAT_RADIUS),
                },
            })

    # --- Colocalization sheet (optional) ---
    coloc_rules = []
    if "Colocalization" in sheets:
        c_df = sheets["Colocalization"]
        for idx, row in c_df.iterrows():
            row_num = idx + 2
            ds = str(row.get("Dataset", "ALL")).strip()
            if not ds or ds == "nan":
                ds = "ALL"
            ctype = str(row.get("Type", "")).strip().lower()
            if not ctype or ctype == "nan":
                continue
            if ctype not in _VALID_COLOC_TYPES:
                errors.append(
                    f"Colocalization row {row_num}: Invalid Type '{ctype}'. "
                    f"Use: {', '.join(sorted(_VALID_COLOC_TYPES))}"
                )
                continue

            source = str(row.get("Source", "")).strip()
            target = str(row.get("Target", "")).strip()
            ch_b = str(row.get("Channel_B", "")).strip()
            if ch_b == "nan":
                ch_b = ""
            cutoff = row.get("Cutoff_um", 1.0)
            if pd.isna(cutoff):
                cutoff = 1.0

            if not source or source == "nan":
                errors.append(f"Colocalization row {row_num}: Source is empty.")
                continue
            if not target or target == "nan":
                errors.append(f"Colocalization row {row_num}: Target is empty.")
                continue
            if ctype == "tri" and (not ch_b):
                errors.append(f"Colocalization row {row_num}: Tri-colocalization requires Channel_B.")
                continue

            if ds != "ALL" and ds not in dataset_names:
                errors.append(f"Colocalization row {row_num}: Dataset '{ds}' not found in Datasets sheet.")

            coloc_rules.append({
                "dataset": ds,
                "type": ctype,
                "source": source,
                "target": target,
                "channel_b": ch_b if ch_b else None,
                "cutoff": float(cutoff),
            })

    if errors:
        return None, "\n".join(errors)

    return {
        "datasets": datasets,
        "puncta_rules": puncta_rules,
        "coloc_rules": coloc_rules,
    }, None


def _resolve_puncta_for_dataset(dataset_name, puncta_rules):
    """Return a {channel: params_dict} for this dataset, merging ALL + per-dataset overrides."""
    result = {}
    # First pass: ALL defaults
    for rule in puncta_rules:
        if rule["dataset"] == "ALL":
            result[rule["channel"]] = dict(rule["params"])
    # Second pass: per-dataset overrides
    for rule in puncta_rules:
        if rule["dataset"] == dataset_name:
            result[rule["channel"]] = dict(rule["params"])
    return result


def _resolve_coloc_for_dataset(dataset_name, coloc_rules):
    """Return (pairwise_rules, tri_rules) for this dataset, merging ALL + per-dataset."""
    pairwise = []
    tri = []
    # ALL first
    for rule in coloc_rules:
        if rule["dataset"] == "ALL":
            entry = {"source": rule["source"], "target": rule["target"],
                     "threshold": rule["cutoff"]}
            if rule["type"] == "pairwise":
                pairwise.append(entry)
            else:
                entry["channel_b"] = rule["channel_b"]
                tri.append(entry)
    # Per-dataset overrides replace ALL entirely if any exist
    has_override = any(r["dataset"] == dataset_name for r in coloc_rules)
    if has_override:
        pairwise = []
        tri = []
        for rule in coloc_rules:
            if rule["dataset"] == dataset_name:
                entry = {"source": rule["source"], "target": rule["target"],
                         "threshold": rule["cutoff"]}
                if rule["type"] == "pairwise":
                    pairwise.append(entry)
                else:
                    entry["channel_b"] = rule["channel_b"]
                    tri.append(entry)
    return pairwise, tri


# ------------------------------------------------------------------
# Widget
# ------------------------------------------------------------------

class BatchProcessWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        self._header = Label(value="Batch Process")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(
            value="<i>Run the full pipeline on multiple datasets from an Excel file.<br>"
                  "Use <b>Generate Template</b> to create a multi-sheet workbook<br>"
                  "with Datasets, Puncta, and Colocalization sheets.</i>"
        )
        self._info.native.setObjectName("widgetInfo")
        self._batch_file = FileEdit(
            label="Batch File:",
            filter="*.xlsx *.xls"
        )
        self._batch_output_dir = FileEdit(
            label="Output:",
            mode='d',
            value=Path.home() / "zFISHer_Batch_Output"
        )
        self._generate_template_btn = PushButton(text="Generate Template")
        self._batch_run_btn = PushButton(text="Run Batch Processing")

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        self._form = Container(labels=True)
        self._form.extend([self._batch_file, self._batch_output_dir])

        _layout = self.native.layout()
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._form.native)
        _layout.addWidget(self._generate_template_btn.native)
        _layout.addWidget(self._batch_run_btn.native)

    def _connect_signals(self):
        self._generate_template_btn.clicked.connect(self._on_generate_template)
        self._batch_run_btn.clicked.connect(self._on_batch_run)

    def _on_generate_template(self):
        """Save a multi-sheet batch template Excel file to a user-chosen location."""
        save_path, _ = QFileDialog.getSaveFileName(
            self.native,
            "Save Batch Template",
            str(Path.home() / "zFISHer_batch_template.xlsx"),
            "Excel Files (*.xlsx)",
        )
        if not save_path:
            return

        sheets = _build_template_sheets()
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
            _add_dropdown_validations(writer.book)

        logger.info("Batch template saved to %s", save_path)
        popups.show_info_popup(
            self.native,
            "Template Saved",
            f"Batch template saved to:\n{save_path}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    @error_handler("Batch Processing Failed")
    def _on_batch_run(self):
        excel_path = Path(self._batch_file.value)
        output_base = Path(self._batch_output_dir.value)

        if not excel_path.is_file():
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "No Excel File",
                "Please select a valid batch Excel file (.xlsx)."
            )
            return

        if not output_base:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "No Output Directory",
                "Please select an output directory."
            )
            return

        # Parse and validate all sheets
        config, error = _parse_batch_config(excel_path)
        if error:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Batch Validation Failed",
                error
            )
            return

        output_base.mkdir(parents=True, exist_ok=True)

        num_items = len(config["datasets"])
        results = []

        with popups.BatchProgressDialog(
            self._viewer.window._qt_window,
            title="Batch Processing",
            text="Preparing..."
        ) as dialog:
            for i, ds in enumerate(config["datasets"]):
                name = ds["name"]
                item_output = ds["output_dir"] or (output_base / name)

                batch_pct = int((i / num_items) * 100)
                dialog.update_batch_progress(
                    batch_pct,
                    f"Batch: {i + 1} / {num_items} — {name}"
                )
                dialog.update_item_progress(0, f"Starting {name}...")

                # Resolve per-dataset puncta and colocalization config
                puncta_config = _resolve_puncta_for_dataset(name, config["puncta_rules"])
                pairwise_rules, tri_rules = _resolve_coloc_for_dataset(name, config["coloc_rules"])

                try:
                    pipeline.run_full_zfisher_pipeline(
                        r1_path=ds["r1"],
                        r2_path=ds["r2"],
                        output_dir=item_output,
                        seg_method=ds["seg_method"],
                        merge_splits=ds["merge_splits"],
                        puncta_config=puncta_config,
                        pairwise_rules=pairwise_rules,
                        tri_rules=tri_rules,
                        progress_callback=dialog.update_item_progress,
                    )
                    results.append((name, "Success"))
                    logger.info("Batch item '%s' completed successfully.", name)
                except Exception as exc:
                    logger.error("Batch item '%s' failed: %s", name, exc, exc_info=True)
                    results.append((name, f"Failed: {exc}"))

            dialog.update_batch_progress(100, "Batch complete.")

        # Summary popup
        lines = [f"  {name}: {status}" for name, status in results]
        summary = "\n".join(lines)
        popups.show_info_popup(
            self._viewer.window._qt_window,
            "Batch Processing Complete",
            f"Processed {num_items} item(s):\n\n{summary}"
        )
