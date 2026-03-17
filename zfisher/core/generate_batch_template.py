"""
Batch template generation, parsing, validation, and resolution helpers.

This module contains all the pure logic for:
- Building the multi-sheet Excel batch template
- Parsing a filled-out batch Excel file into a structured config
- Validating channel names against actual input files
- Resolving per-dataset puncta and colocalization rules
"""
import logging
from pathlib import Path

import pandas as pd

from .. import constants
from . import io

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
# Template construction
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


def add_dropdown_validations(workbook):
    """Add Excel data-validation dropdowns to columns with fixed choices.

    Channel/name fields use ``errorStyle="information"`` so that Excel
    shows a dropdown for convenience but still accepts custom typed values.
    Fields with a strict set of options (Seg_Method, TRUE/FALSE, Type) use
    the default ``errorStyle="stop"`` which rejects invalid input.
    """
    from openpyxl.worksheet.datavalidation import DataValidation

    MAX_ROW = 500

    # Build channel list from constants at generation time
    channel_names = sorted(set(constants.CHANNEL_COLORS.keys()))
    nuclear_names = sorted(set(constants.NUCLEAR_STAIN_NAMES))

    def _soft_list(items, prompt="", title=""):
        """Dropdown that ALLOWS custom text (informational error style)."""
        dv = DataValidation(
            type="list",
            formula1='"' + ",".join(items) + '"',
            allow_blank=True,
            errorStyle="information",
        )
        dv.error = "Value not in dropdown. Click OK to keep your custom value."
        dv.errorTitle = title or "Custom Value"
        if prompt:
            dv.prompt = prompt
            dv.promptTitle = title
        return dv

    def _strict_list(items, prompt="", title=""):
        """Dropdown that REJECTS values not in the list."""
        dv = DataValidation(
            type="list",
            formula1='"' + ",".join(items) + '"',
            allow_blank=True,
        )
        if prompt:
            dv.prompt = prompt
            dv.promptTitle = title
        return dv

    # --- Datasets sheet ---
    ws = workbook["Datasets"]

    dv = _soft_list(nuclear_names, "Select or type the R1 nuclear channel.", "R1_Nuclear_Channel")
    ws.add_data_validation(dv)
    dv.add(f"E2:E{MAX_ROW}")

    dv = _soft_list(nuclear_names, "Select or type the R2 nuclear channel.", "R2_Nuclear_Channel")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{MAX_ROW}")

    dv = _strict_list(sorted(_VALID_SEG_METHODS), "Choose a segmentation method.", "Seg_Method")
    ws.add_data_validation(dv)
    dv.add(f"G2:G{MAX_ROW}")

    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"H2:H{MAX_ROW}")

    # --- Puncta sheet ---
    ws = workbook["Puncta"]

    dv = _soft_list(channel_names, "Select or type the channel name.", "Channel")
    ws.add_data_validation(dv)
    dv.add(f"B2:B{MAX_ROW}")

    dv = _strict_list(sorted(_VALID_PUNCTA_ALGORITHMS), "Choose a detection algorithm.", "Algorithm")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{MAX_ROW}")

    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"G2:G{MAX_ROW}")

    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"H2:H{MAX_ROW}")

    # --- Colocalization sheet ---
    ws = workbook["Colocalization"]

    dv = _strict_list(sorted(_VALID_COLOC_TYPES), "Choose pairwise or tri.", "Type")
    ws.add_data_validation(dv)
    dv.add(f"B2:B{MAX_ROW}")

    dv = _soft_list(channel_names, "Select or type the channel name.", "Source")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{MAX_ROW}")

    dv = _soft_list(channel_names, "Select or type the channel name.", "Target")
    ws.add_data_validation(dv)
    dv.add(f"D2:D{MAX_ROW}")

    dv = _soft_list(channel_names, "Select or type the channel name.", "Channel_B")
    ws.add_data_validation(dv)
    dv.add(f"E2:E{MAX_ROW}")


def build_template_sheets():
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


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def _is_example_row(value):
    """Return True if a string value starts with the example prefix."""
    return str(value).strip().startswith(_EXAMPLE_PREFIX)


def parse_batch_config(excel_path):
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


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validate_channels(config):
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


# ------------------------------------------------------------------
# Resolution helpers
# ------------------------------------------------------------------

def resolve_puncta_for_dataset(dataset_name, puncta_rules):
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


def resolve_coloc_for_dataset(dataset_name, coloc_rules):
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
