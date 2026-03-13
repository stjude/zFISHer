from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import os
import logging
import napari
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QFrame, QFileDialog, QMessageBox

from ...core import pipeline, io
from ... import constants
from .. import popups
from ..decorators import error_handler
from ..style import COLORS

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.nd2', '.tif', '.tiff'}

# Prefix used for example rows in the template — stripped during parsing.
_EXAMPLE_PREFIX = "[EXAMPLE]"

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

def _build_instructions_rows():
    """Return a list of (Section, Description) rows for the Instructions sheet."""
    return [
        ("OVERVIEW", ""),
        ("",
         "This workbook defines a batch processing run for zFISHer. "
         "Fill out the Datasets, Puncta, and Colocalization sheets, then load this file in the Batch Process tab."),
        ("",
         "Rows whose Name/Dataset column starts with '[EXAMPLE]' are ignored during processing — "
         "they are provided as reference only. You can leave them or delete them."),
        ("", ""),
        ("DATASETS SHEET (required)", ""),
        ("Name",
         "A unique label for each dataset (e.g. 'FOV1', 'Sample_A'). Must be unique across all rows."),
        ("R1",
         "Full file path to the Round 1 image (.nd2, .tif, or .tiff). Example: C:\\Data\\FOV1_R1.nd2"),
        ("R2",
         "Full file path to the Round 2 image (.nd2, .tif, or .tiff). Example: C:\\Data\\FOV1_R2.nd2"),
        ("Output_Dir",
         "Optional. Per-dataset output directory override. If blank, "
         "a subfolder named after the dataset is created under the base output directory."),
        ("R1_Nuclear_Channel",
         "The name of the nuclear stain channel in the R1 image (e.g. DAPI, HOECHST). "
         "Dropdown provides common stains. Must exactly match a channel name in your R1 file. Defaults to DAPI."),
        ("R2_Nuclear_Channel",
         "The name of the nuclear stain channel in the R2 image (e.g. DAPI, HOECHST). "
         "Dropdown provides common stains. Must exactly match a channel name in your R2 file. Defaults to DAPI."),
        ("Seg_Method",
         "Nuclei segmentation algorithm: 'Classical' (fast, watershed-based) or 'Cellpose' (deep learning). "
         "Defaults to Classical if left blank."),
        ("Merge_Splits",
         "TRUE or FALSE. Whether to merge over-segmented (split) nuclei after segmentation. Defaults to TRUE."),
        ("", ""),
        ("PUNCTA SHEET (optional)", ""),
        ("",
         "Defines puncta detection parameters per channel. If this sheet is empty, "
         "all non-nuclear channels are detected with default parameters."),
        ("Dataset",
         "'ALL' applies the row as a default for every dataset. "
         "Use a specific dataset Name to override defaults for that dataset only. "
         "Must match a Name from the Datasets sheet."),
        ("Channel",
         "The fluorescent channel to detect puncta in (e.g. Cy5, GFP, AF647). "
         "Must exactly match a channel name in your input files. "
         "Dropdown provides common names but you can type any name."),
        ("Algorithm",
         "Detection algorithm: 'Local Maxima', 'Laplacian of Gaussian', "
         "'Difference of Gaussian', or 'Radial Symmetry'. Defaults to Local Maxima."),
        ("Sensitivity",
         f"Relative intensity threshold (0-1). Lower = more sensitive. Default: {constants.PUNCTA_THRESHOLD_REL}"),
        ("Min_Distance",
         f"Minimum distance (pixels) between detected puncta. Default: {constants.PUNCTA_MIN_DISTANCE}"),
        ("Sigma",
         f"Gaussian smoothing sigma applied before detection. 0 = no smoothing. Default: {constants.PUNCTA_SIGMA}"),
        ("Nuclei_Only",
         "TRUE or FALSE. If TRUE, only puncta inside nuclei masks are kept. Default: TRUE."),
        ("Tophat",
         "TRUE or FALSE. Apply top-hat background subtraction before detection. Default: FALSE."),
        ("Tophat_Radius",
         f"Radius for top-hat filter (pixels). Only used if Tophat=TRUE. Default: {constants.PUNCTA_TOPHAT_RADIUS}"),
        ("", ""),
        ("COLOCALIZATION SHEET (optional)", ""),
        ("",
         "Defines which channel pairs (or triples) to analyze for colocalization. "
         "If this sheet is empty, no colocalization analysis is performed."),
        ("Dataset",
         "'ALL' applies the rule to every dataset. Use a specific Name to override. "
         "Per-dataset rows REPLACE all 'ALL' rules for that dataset."),
        ("Type",
         "'pairwise' for two-channel colocalization, or 'tri' for three-channel tri-colocalization."),
        ("Source",
         "The anchor/reference channel name (e.g. Cy5). For pairwise: the first channel. "
         "For tri: the anchor channel."),
        ("Target",
         "The second channel name (e.g. GFP). For pairwise: the target channel. "
         "For tri: the first comparison channel (Channel A)."),
        ("Channel_B",
         "Only for tri-colocalization. The third channel name (e.g. AF555). "
         "Leave blank for pairwise rules."),
        ("Cutoff_um",
         "Maximum distance in microns to consider two puncta as colocalized. Default: 1.0"),
    ]


def _add_dropdown_validations(workbook):
    """Add Excel data-validation dropdowns to columns with fixed choices."""
    from openpyxl.worksheet.datavalidation import DataValidation

    # Max rows to apply validation (generous upper bound)
    MAX_ROW = 500

    # Build channel list from constants at generation time
    channel_names = sorted(set(constants.CHANNEL_COLORS.keys()))
    nuclear_names = sorted(set(constants.NUCLEAR_STAIN_NAMES))

    # --- Datasets sheet ---
    ws = workbook["Datasets"]
    # R1_Nuclear_Channel (column E)
    r1_nuc_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(nuclear_names) + '"',
        allow_blank=True,
    )
    r1_nuc_dv.prompt = "Select or type the R1 nuclear channel name."
    r1_nuc_dv.promptTitle = "R1_Nuclear_Channel"
    r1_nuc_dv.showErrorMessage = False  # Allow custom values
    ws.add_data_validation(r1_nuc_dv)
    r1_nuc_dv.add(f"E2:E{MAX_ROW}")

    # R2_Nuclear_Channel (column F)
    r2_nuc_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(nuclear_names) + '"',
        allow_blank=True,
    )
    r2_nuc_dv.prompt = "Select or type the R2 nuclear channel name."
    r2_nuc_dv.promptTitle = "R2_Nuclear_Channel"
    r2_nuc_dv.showErrorMessage = False  # Allow custom values
    ws.add_data_validation(r2_nuc_dv)
    r2_nuc_dv.add(f"F2:F{MAX_ROW}")

    # Seg_Method (column G)
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
    seg_dv.add(f"G2:G{MAX_ROW}")

    # Merge_Splits (column H)
    bool_dv = DataValidation(
        type="list", formula1='"TRUE,FALSE"', allow_blank=True,
    )
    ws.add_data_validation(bool_dv)
    bool_dv.add(f"H2:H{MAX_ROW}")

    # --- Puncta sheet ---
    ws = workbook["Puncta"]
    # Channel (column B) — populated from CHANNEL_COLORS
    ch_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(channel_names) + '"',
        allow_blank=True,
    )
    ch_dv.error = "Unknown channel. You can type a custom name."
    ch_dv.errorTitle = "Channel"
    ch_dv.prompt = "Select or type the channel name."
    ch_dv.promptTitle = "Channel"
    ch_dv.showErrorMessage = False  # Allow custom values
    ws.add_data_validation(ch_dv)
    ch_dv.add(f"B2:B{MAX_ROW}")

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

    # Source (column C) — channel dropdown
    src_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(channel_names) + '"',
        allow_blank=True,
    )
    src_dv.showErrorMessage = False
    ws.add_data_validation(src_dv)
    src_dv.add(f"C2:C{MAX_ROW}")

    # Target (column D) — channel dropdown
    tgt_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(channel_names) + '"',
        allow_blank=True,
    )
    tgt_dv.showErrorMessage = False
    ws.add_data_validation(tgt_dv)
    tgt_dv.add(f"D2:D{MAX_ROW}")

    # Channel_B (column E) — channel dropdown
    chb_dv = DataValidation(
        type="list",
        formula1='"' + ",".join(channel_names) + '"',
        allow_blank=True,
    )
    chb_dv.showErrorMessage = False
    ws.add_data_validation(chb_dv)
    chb_dv.add(f"E2:E{MAX_ROW}")


def _build_template_sheets():
    """Return an {sheet_name: DataFrame} dict for the batch template.

    Each data sheet is pre-filled with 50 rows of default values so that
    users just need to fill in names, paths, and channels.  The parser
    gracefully ignores rows where Name/Dataset/Channel are empty.
    """
    N = 50  # number of pre-filled rows

    # --- Instructions ---
    instr_rows = _build_instructions_rows()
    instructions = pd.DataFrame(instr_rows, columns=["Section", "Description"])

    # --- Datasets: 1 example + 50 default rows ---
    ds_example = {
        "Name": f"{_EXAMPLE_PREFIX} FOV1",
        "R1": r"C:\Data\FOV1_R1.nd2",
        "R2": r"C:\Data\FOV1_R2.nd2",
        "Output_Dir": "",
        "R1_Nuclear_Channel": "DAPI",
        "R2_Nuclear_Channel": "DAPI",
        "Seg_Method": "Classical",
        "Merge_Splits": True,
    }
    ds_default = {
        "Name": "",
        "R1": "",
        "R2": "",
        "Output_Dir": "",
        "R1_Nuclear_Channel": "DAPI",
        "R2_Nuclear_Channel": "DAPI",
        "Seg_Method": "Classical",
        "Merge_Splits": True,
    }
    datasets = pd.DataFrame([ds_example] + [dict(ds_default) for _ in range(N)])

    # --- Puncta: 2 examples + 50 default rows ---
    p_example1 = {
        "Dataset": f"{_EXAMPLE_PREFIX} ALL",
        "Channel": "Cy5",
        "Algorithm": "Local Maxima",
        "Sensitivity": constants.PUNCTA_THRESHOLD_REL,
        "Min_Distance": constants.PUNCTA_MIN_DISTANCE,
        "Sigma": constants.PUNCTA_SIGMA,
        "Nuclei_Only": True,
        "Tophat": False,
        "Tophat_Radius": constants.PUNCTA_TOPHAT_RADIUS,
    }
    p_example2 = {
        "Dataset": f"{_EXAMPLE_PREFIX} FOV1",
        "Channel": "GFP",
        "Algorithm": "Laplacian of Gaussian",
        "Sensitivity": 0.03,
        "Min_Distance": constants.PUNCTA_MIN_DISTANCE,
        "Sigma": 1.0,
        "Nuclei_Only": True,
        "Tophat": True,
        "Tophat_Radius": constants.PUNCTA_TOPHAT_RADIUS,
    }
    p_default = {
        "Dataset": "ALL",
        "Channel": "",
        "Algorithm": "Local Maxima",
        "Sensitivity": constants.PUNCTA_THRESHOLD_REL,
        "Min_Distance": constants.PUNCTA_MIN_DISTANCE,
        "Sigma": constants.PUNCTA_SIGMA,
        "Nuclei_Only": True,
        "Tophat": False,
        "Tophat_Radius": constants.PUNCTA_TOPHAT_RADIUS,
    }
    puncta = pd.DataFrame([p_example1, p_example2] + [dict(p_default) for _ in range(N)])

    # --- Colocalization: 2 examples + 50 default rows ---
    c_example1 = {
        "Dataset": f"{_EXAMPLE_PREFIX} ALL",
        "Type": "pairwise",
        "Source": "Cy5",
        "Target": "GFP",
        "Channel_B": "",
        "Cutoff_um": 1.0,
    }
    c_example2 = {
        "Dataset": f"{_EXAMPLE_PREFIX} ALL",
        "Type": "tri",
        "Source": "Cy5",
        "Target": "GFP",
        "Channel_B": "AF555",
        "Cutoff_um": 1.0,
    }
    c_default = {
        "Dataset": "ALL",
        "Type": "",
        "Source": "",
        "Target": "",
        "Channel_B": "",
        "Cutoff_um": 1.0,
    }
    colocalization = pd.DataFrame([c_example1, c_example2] + [dict(c_default) for _ in range(N)])

    return {
        "Instructions": instructions,
        "Datasets": datasets,
        "Puncta": puncta,
        "Colocalization": colocalization,
    }


def _is_example_row(value):
    """Return True if a string value starts with the example prefix."""
    return str(value).strip().startswith(_EXAMPLE_PREFIX)


def _parse_batch_config(excel_path):
    """Parse the multi-sheet batch Excel into a structured config dict.

    Rows whose Name/Dataset column starts with '[EXAMPLE]' are skipped.

    Returns
    -------
    (config, None) on success, or (None, error_string) on failure.

    config = {
        'datasets': [
            {'name': str, 'r1': Path, 'r2': Path, 'output_dir': Path|None,
             'r1_nuclear_channel': str, 'r2_nuclear_channel': str,
             'seg_method': str, 'merge_splits': bool},
            ...
        ],
        'puncta_rules': [...],
        'coloc_rules': [...],
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

    # Filter out example rows and completely empty rows
    ds_df = ds_df[~ds_df["Name"].apply(lambda v: _is_example_row(v))]
    ds_df = ds_df[ds_df["Name"].apply(lambda v: bool(str(v).strip()) and str(v).strip() != "nan")]
    if ds_df.empty:
        return None, "Datasets sheet has no data rows (only example/empty rows found)."

    errors = []
    datasets = []
    dataset_names = set()
    for idx, row in ds_df.iterrows():
        row_num = idx + 2
        name = str(row.get("Name", "")).strip()
        if name in dataset_names:
            errors.append(f"Datasets row {row_num}: Duplicate name '{name}'.")
        dataset_names.add(name)

        for col in ("R1", "R2"):
            raw = str(row.get(col, "")).strip()
            if not raw or raw == "nan":
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

        # Nuclear channels (separate for R1 and R2)
        r1_nuc_raw = str(row.get("R1_Nuclear_Channel", "")).strip()
        r1_nuc = r1_nuc_raw if (r1_nuc_raw and r1_nuc_raw != "nan") else "DAPI"
        r2_nuc_raw = str(row.get("R2_Nuclear_Channel", "")).strip()
        r2_nuc = r2_nuc_raw if (r2_nuc_raw and r2_nuc_raw != "nan") else "DAPI"

        seg = str(row.get("Seg_Method", "Classical")).strip()
        if seg and seg != "nan" and seg not in _VALID_SEG_METHODS:
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
            "r1_nuclear_channel": r1_nuc,
            "r2_nuclear_channel": r2_nuc,
            "seg_method": seg if (seg and seg != "nan") else "Classical",
            "merge_splits": merge,
        })

    # --- Puncta sheet (optional) ---
    puncta_rules = []
    if "Puncta" in sheets:
        p_df = sheets["Puncta"]
        # Filter out example rows
        p_df = p_df[~p_df["Dataset"].apply(lambda v: _is_example_row(v))]
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
            if algo and algo != "nan" and algo not in _VALID_PUNCTA_ALGORITHMS:
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
                    "method": algo if (algo and algo != "nan") else "Local Maxima",
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
        # Filter out example rows
        c_df = c_df[~c_df["Dataset"].apply(lambda v: _is_example_row(v))]
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


def _validate_channels(config):
    """Validate that channel names in the config exist in the actual input files.

    Reads channel metadata from each R1/R2 file (lightweight, no pixel loading)
    and checks that:
      - The nuclear channel exists in both R1 and R2
      - Puncta channel names exist in the relevant input files
      - Colocalization channel names exist in the relevant input files

    Returns a list of warning/error strings.  Empty list = all good.
    """
    warnings = []

    # Cache channel lists per file path to avoid re-reading
    _channel_cache = {}

    def _get_channels(path):
        key = str(path)
        if key not in _channel_cache:
            _channel_cache[key] = io.peek_channel_names(path)
        return _channel_cache[key]

    # Collect all available channels per dataset (union of R1 + R2)
    dataset_channels = {}
    for ds in config["datasets"]:
        r1_chs = _get_channels(ds["r1"])
        r2_chs = _get_channels(ds["r2"])

        if r1_chs is None:
            warnings.append(f"[{ds['name']}] Could not read channel names from R1: {ds['r1']}")
        if r2_chs is None:
            warnings.append(f"[{ds['name']}] Could not read channel names from R2: {ds['r2']}")

        # Nuclear channel check — each must be in its respective round
        r1_nuc = ds["r1_nuclear_channel"]
        r2_nuc = ds["r2_nuclear_channel"]
        if r1_chs and r1_nuc not in r1_chs:
            warnings.append(
                f"[{ds['name']}] R1 nuclear channel '{r1_nuc}' not found in R1 channels: {r1_chs}"
            )
        if r2_chs and r2_nuc not in r2_chs:
            warnings.append(
                f"[{ds['name']}] R2 nuclear channel '{r2_nuc}' not found in R2 channels: {r2_chs}"
            )

        all_chs = set()
        if r1_chs:
            all_chs.update(r1_chs)
        if r2_chs:
            all_chs.update(r2_chs)
        dataset_channels[ds["name"]] = all_chs

    # Validate puncta channels
    for rule in config["puncta_rules"]:
        ch = rule["channel"]
        targets = (
            config["datasets"] if rule["dataset"] == "ALL"
            else [ds for ds in config["datasets"] if ds["name"] == rule["dataset"]]
        )
        for ds in targets:
            chs = dataset_channels.get(ds["name"], set())
            if chs and ch not in chs:
                warnings.append(
                    f"[{ds['name']}] Puncta channel '{ch}' not found in input file channels: "
                    f"{sorted(chs)}"
                )

    # Validate colocalization channels
    for rule in config["coloc_rules"]:
        targets = (
            config["datasets"] if rule["dataset"] == "ALL"
            else [ds for ds in config["datasets"] if ds["name"] == rule["dataset"]]
        )
        for ds in targets:
            chs = dataset_channels.get(ds["name"], set())
            if not chs:
                continue
            for col_name, col_label in [
                ("source", "Source"), ("target", "Target"), ("channel_b", "Channel_B")
            ]:
                val = rule.get(col_name)
                if val and val not in chs:
                    warnings.append(
                        f"[{ds['name']}] Colocalization {col_label} '{val}' not found in "
                        f"input file channels: {sorted(chs)}"
                    )

    return warnings


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

            # Auto-size the Instructions columns for readability
            ws = writer.book["Instructions"]
            ws.column_dimensions['A'].width = 22
            ws.column_dimensions['B'].width = 110

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

        # Deep validation: check file channels match config
        channel_warnings = _validate_channels(config)

        if channel_warnings:
            msg = (
                "The following issues were found during validation:\n\n"
                + "\n".join(f"  - {w}" for w in channel_warnings)
                + "\n\nDo you want to continue anyway?"
            )
            reply = QMessageBox.warning(
                self._viewer.window._qt_window,
                "Batch Validation Warnings",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        else:
            # No warnings — confirm start
            reply = QMessageBox.question(
                self._viewer.window._qt_window,
                "Batch Validation Passed",
                f"All {len(config['datasets'])} dataset(s) validated successfully.\n\n"
                "Start batch processing?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
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
                        r1_nuclear_channel=ds["r1_nuclear_channel"],
                        r2_nuclear_channel=ds["r2_nuclear_channel"],
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
