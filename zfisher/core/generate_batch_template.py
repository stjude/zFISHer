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
        # ── Overview ──────────────────────────────────────────────────
        ("OVERVIEW", ""),
        ("", ""),
        ("What is this file?",
         "This Excel workbook defines a batch processing run for zFISHer. "
         "Each dataset (pair of Round 1 and Round 2 images) is processed through the full zFISHer pipeline: "
         "nuclei segmentation, puncta detection, image registration, consensus nuclei matching, and colocalization analysis."),
        ("", ""),
        ("How to use it",
         "1. Fill out the 'Datasets' sheet with your image file paths and settings. "
         "2. Optionally configure per-channel puncta detection in the 'Puncta' sheet. "
         "3. Optionally define colocalization rules in the 'Colocalization' sheet. "
         "4. Save this file, then load it in zFISHer's Batch Process tab and click 'Run Batch Processing'."),
        ("", ""),
        ("Example rows",
         "Rows whose Name or Dataset column starts with '[EXAMPLE]' are provided as reference only. "
         "They are completely ignored during processing. You can leave them in the file or delete them."),
        ("", ""),
        ("Defaults",
         "Most columns have sensible defaults pre-filled. You only need to fill in Name, R1, and R2 paths. "
         "All other columns can be left at their defaults unless you need to customize behavior."),
        ("", ""),
        ("Column dropdowns",
         "Many columns have dropdown menus for convenience. For channel names, the dropdown lists common names "
         "but you can type any custom name — just make sure it exactly matches a channel in your input files."),
        ("", ""),
        ("", ""),

        # ── Datasets Sheet ────────────────────────────────────────────
        ("DATASETS SHEET", ""),
        ("(required)", "This is the main sheet. Each row defines one dataset to process."),
        ("", ""),

        ("--- Input Files ---", ""),
        ("Name",
         "A short, unique label for this dataset (e.g. 'FOV1', 'Sample_A', 'Exp3_Slide2'). "
         "This name is used to create output subfolders and label results. Must be unique across all rows."),
        ("R1",
         "Full file path to the Round 1 microscopy image. Supported formats: .nd2, .tif, .tiff. "
         "Example: C:\\Data\\Experiment1\\FOV1_R1.nd2"),
        ("R2",
         "Full file path to the Round 2 microscopy image. Must be from the same field of view as R1. "
         "Supported formats: .nd2, .tif, .tiff. Example: C:\\Data\\Experiment1\\FOV1_R2.nd2"),
        ("Output_Dir",
         "Optional. Override the output directory for this specific dataset. "
         "If left blank, a subfolder named after the dataset (the Name column) will be created "
         "under the base output directory you select in the Batch Process tab. "
         "Example: D:\\Results\\FOV1"),
        ("", ""),

        ("--- Nuclear Channels ---", ""),
        ("R1_Nuclear_Channel",
         "The name of the nuclear stain channel in the Round 1 image (e.g. DAPI, Hoechst, HOECHST). "
         "This channel is used for nuclei segmentation. The name must exactly match a channel name in your R1 file. "
         "A dropdown provides common nuclear stain names, but you can type any custom name. Defaults to 'DAPI'."),
        ("R2_Nuclear_Channel",
         "The name of the nuclear stain channel in the Round 2 image. "
         "Same rules as R1_Nuclear_Channel. Can be different from R1 if your rounds use different stains. "
         "Defaults to 'DAPI'."),
        ("", ""),

        ("--- Segmentation ---", ""),
        ("Seg_Method",
         "The algorithm used to segment nuclei from the nuclear stain channel. "
         "Options: 'Classical' (fast, watershed-based — good for most data) or 'Cellpose' (deep learning — "
         "better for difficult samples but slower, requires cellpose to be installed). "
         "Defaults to 'Classical' if left blank."),
        ("Merge_Splits",
         "TRUE or FALSE. When TRUE, the pipeline automatically merges nuclei that were over-segmented "
         "(split into multiple fragments by the segmentation algorithm). Recommended for most datasets. "
         "Set to FALSE only if you observe that merging is incorrectly combining separate nuclei. "
         "Defaults to TRUE."),
        ("", ""),

        ("--- Registration ---", ""),
        ("Apply_Warp",
         "TRUE or FALSE. When TRUE, a deformable B-spline warp is applied after rigid alignment to correct "
         "for tissue deformation between imaging rounds. This produces more accurate alignment but takes longer. "
         "Set to FALSE for rigid-only alignment (faster, sufficient when tissue deformation is minimal). "
         "Defaults to TRUE."),
        ("Max_RANSAC_Distance",
         "Maximum distance in pixels for RANSAC inlier matching during the registration step. "
         "This controls how strict the centroid matching is: a smaller value requires closer matches, "
         "a larger value is more permissive. Set to 0 for automatic detection (recommended for most data). "
         "Only increase this if registration is failing due to large tissue shifts. "
         "Defaults to 0 (auto-detect)."),
        ("", ""),

        ("--- Consensus Nuclei ---", ""),
        ("Overlap_Method",
         "How to combine the aligned R1 and R2 nuclei masks into a single consensus mask. "
         "Options: 'Intersection' keeps only voxels where BOTH rounds agree a nucleus exists "
         "(conservative — fewer, higher-confidence nuclei). 'Union' keeps voxels from EITHER round "
         "(permissive — more nuclei, but may include artifacts). "
         "Defaults to 'Intersection'."),
        ("Match_Threshold",
         "Maximum centroid distance in pixels to consider two nuclei (one from R1, one from R2) as the same cell. "
         "Set to 0 for automatic detection based on the data distribution (recommended). "
         "Increase if nuclei are being incorrectly split across rounds; decrease if different nuclei are being merged. "
         "Defaults to 0 (auto-detect)."),
        ("Remove_Extranuclear_Puncta",
         "TRUE or FALSE. When TRUE, puncta that fall outside the consensus nuclei mask are removed from the final results. "
         "This ensures only intranuclear puncta are analyzed. Set to FALSE to keep all detected puncta regardless of "
         "nuclear location (useful if you need to analyze cytoplasmic or extracellular puncta). "
         "Defaults to TRUE."),
        ("", ""),
        ("", ""),

        # ── Puncta Sheet ──────────────────────────────────────────────
        ("PUNCTA SHEET", ""),
        ("(optional)", "Defines puncta detection parameters per fluorescent channel."),
        ("", ""),
        ("",
         "If this sheet is left empty (no data rows), zFISHer will automatically detect puncta "
         "on ALL non-nuclear channels using default parameters. Use this sheet to customize "
         "detection sensitivity, algorithm, or to limit detection to specific channels."),
        ("", ""),
        ("",
         "HOW OVERRIDES WORK: Rows with Dataset='ALL' set the default parameters for every dataset. "
         "Rows with a specific dataset Name override the 'ALL' defaults for that dataset only. "
         "This lets you tune parameters per-FOV when needed while keeping a single set of defaults."),
        ("", ""),

        ("--- Columns ---", ""),
        ("Dataset",
         "Which dataset(s) this row applies to. Use 'ALL' to set defaults for every dataset. "
         "Use a specific Name from the Datasets sheet to override defaults for just that dataset. "
         "The Name must exactly match a value in the Datasets sheet's Name column."),
        ("Channel",
         "The fluorescent channel to detect puncta in (e.g. Cy5, GFP, AF647, FITC). "
         "Must exactly match a channel name in your input files. "
         "The dropdown provides common channel names, but you can type any name. "
         "Leave blank to skip a row."),
        ("Algorithm",
         "The spot detection algorithm. Options:\n"
         "  - 'Local Maxima': Fastest. Finds intensity peaks separated by Min_Distance. Best for well-separated, bright spots.\n"
         "  - 'Laplacian of Gaussian': Blob-aware. Better for crowded fields where spots overlap. Uses Sigma for scale.\n"
         "  - 'Difference of Gaussian': Similar to LoG but faster. Good for spots of known size.\n"
         "  - 'Radial Symmetry': Best for high-density fields with minimal filtering needed.\n"
         "Defaults to 'Local Maxima'."),
        ("Sensitivity",
         f"Relative intensity threshold between 0 and 1. Controls how dim a spot can be and still be detected. "
         f"Lower values detect dimmer spots but may increase false positives. Higher values are more stringent. "
         f"Start with the default ({constants.PUNCTA_THRESHOLD_REL}) and adjust based on your results."),
        ("Min_Distance",
         f"Minimum distance in pixels between detected puncta. Prevents double-counting nearby spots. "
         f"Increase if you see clusters of detections on single spots. Decrease if closely spaced spots are being merged. "
         f"Default: {constants.PUNCTA_MIN_DISTANCE} pixels."),
        ("Sigma",
         f"Gaussian smoothing sigma (in pixels) applied before detection. "
         f"Match this to the approximate radius of your puncta. "
         f"Set to 0 for no pre-smoothing (use with Local Maxima on clean data). "
         f"For LoG/DoG, this defines the expected spot size. Default: {constants.PUNCTA_SIGMA}."),
        ("Nuclei_Only",
         "TRUE or FALSE. When TRUE, only puncta located inside nuclei masks are kept (extranuclear spots are discarded). "
         "Set to FALSE to detect puncta everywhere in the image, including cytoplasm and extracellular space. "
         "Defaults to TRUE."),
        ("Tophat",
         "TRUE or FALSE. Apply a top-hat background subtraction filter before running detection. "
         "Useful for images with uneven illumination or high autofluorescence. "
         "The filter removes large-scale intensity variations while preserving small bright spots. "
         "Defaults to FALSE."),
        ("Tophat_Radius",
         f"Radius in pixels for the top-hat filter. Only used when Tophat=TRUE. "
         f"Should be larger than your puncta but smaller than background intensity gradients. "
         f"Default: {constants.PUNCTA_TOPHAT_RADIUS} pixels."),
        ("", ""),
        ("", ""),

        # ── Colocalization Sheet ──────────────────────────────────────
        ("COLOCALIZATION SHEET", ""),
        ("(optional)", "Defines which channel pairs or triples to analyze for spatial colocalization."),
        ("", ""),
        ("",
         "If this sheet is left empty, no colocalization analysis is performed and the final report "
         "will contain only per-channel puncta counts and nucleus assignments."),
        ("", ""),
        ("",
         "PAIRWISE colocalization asks: 'For each punctum in the Source channel, is there a punctum in the "
         "Target channel within the cutoff distance?' This produces counts of colocalized vs. non-colocalized spots."),
        ("", ""),
        ("",
         "TRI-COLOCALIZATION asks: 'For each punctum in the Source (anchor) channel, is there a punctum in "
         "BOTH Target (Channel A) AND Channel B within the cutoff distance?' "
         "This identifies triple-positive spots where all three channels converge."),
        ("", ""),
        ("",
         "HOW OVERRIDES WORK: Same as the Puncta sheet. 'ALL' rows set defaults; per-dataset rows "
         "REPLACE all 'ALL' rules for that specific dataset. This lets you run different colocalization "
         "comparisons on different FOVs if needed."),
        ("", ""),

        ("--- Columns ---", ""),
        ("Dataset",
         "Which dataset(s) this rule applies to. Use 'ALL' for every dataset, "
         "or a specific Name from the Datasets sheet to apply only to that dataset. "
         "Per-dataset rows completely replace any 'ALL' rules for that dataset."),
        ("Type",
         "'pairwise' for standard two-channel colocalization, or 'tri' for three-channel tri-colocalization. "
         "Case-insensitive."),
        ("Source",
         "The anchor/reference channel name (e.g. Cy5). "
         "For pairwise: distances are measured FROM each Source punctum TO the nearest Target punctum. "
         "For tri: this is the anchor — it must be near both Target and Channel_B."),
        ("Target",
         "The second channel name (e.g. GFP). "
         "For pairwise: the channel to search for nearby puncta. "
         "For tri: the first comparison channel (must colocalize with Source)."),
        ("Channel_B",
         "Only used for tri-colocalization. The third channel name (e.g. AF555). "
         "The anchor (Source) must be near both Target AND Channel_B to count as tri-colocalized. "
         "Leave blank for pairwise rules."),
        ("Cutoff_um",
         "Maximum distance in microns to consider two puncta as colocalized. "
         "A Source punctum is 'colocalized' with a Target punctum if the nearest Target is within this distance. "
         "Typical values: 0.5-2.0 um depending on your imaging resolution and expected spot proximity. "
         "Default: 1.0 um."),
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

    def _numeric_range(min_val, max_val, allow_decimal=True, prompt="", title=""):
        """Numeric validation that constrains to a range."""
        dv = DataValidation(
            type="decimal" if allow_decimal else "whole",
            operator="between",
            formula1=str(min_val),
            formula2=str(max_val),
            allow_blank=True,
        )
        dv.error = f"Value must be between {min_val} and {max_val}."
        dv.errorTitle = title or "Invalid Value"
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

    # Apply_Warp (col I)
    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"I2:I{MAX_ROW}")

    # Overlap_Method (col K)
    dv = _strict_list(["Intersection", "Union"], "Choose overlap method.", "Overlap_Method")
    ws.add_data_validation(dv)
    dv.add(f"K2:K{MAX_ROW}")

    # Max_RANSAC_Distance (col J) — integer 0-100
    dv = _numeric_range(0, 100, allow_decimal=False, prompt="0 = auto-detect. Range: 0-100.", title="Max RANSAC Distance")
    ws.add_data_validation(dv)
    dv.add(f"J2:J{MAX_ROW}")

    # Match_Threshold (col L) — integer 0-100
    dv = _numeric_range(0, 100, allow_decimal=False, prompt="0 = auto-detect. Range: 0-100.", title="Match Threshold")
    ws.add_data_validation(dv)
    dv.add(f"L2:L{MAX_ROW}")

    # Remove_Extranuclear_Puncta (col M)
    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"M2:M{MAX_ROW}")

    # --- Puncta sheet ---
    ws = workbook["Puncta"]

    dv = _soft_list(channel_names, "Select or type the channel name.", "Channel")
    ws.add_data_validation(dv)
    dv.add(f"B2:B{MAX_ROW}")

    dv = _strict_list(sorted(_VALID_PUNCTA_ALGORITHMS), "Choose a detection algorithm.", "Algorithm")
    ws.add_data_validation(dv)
    dv.add(f"C2:C{MAX_ROW}")

    # Sensitivity (col D) — decimal 0-1
    dv = _numeric_range(0.0, 1.0, allow_decimal=True, prompt="Relative threshold 0-1. Lower = more sensitive.", title="Sensitivity")
    ws.add_data_validation(dv)
    dv.add(f"D2:D{MAX_ROW}")

    # Min_Distance (col E) — integer 1-20
    dv = _numeric_range(1, 20, allow_decimal=False, prompt="Min separation in pixels. Range: 1-20.", title="Min Distance")
    ws.add_data_validation(dv)
    dv.add(f"E2:E{MAX_ROW}")

    # Sigma (col F) — decimal 0-5
    dv = _numeric_range(0.0, 5.0, allow_decimal=True, prompt="Gaussian sigma in pixels. 0 = no smoothing. Range: 0-5.", title="Sigma")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{MAX_ROW}")

    # Nuclei_Only (col G)
    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"G2:G{MAX_ROW}")

    # Tophat (col H)
    dv = _strict_list(["TRUE", "FALSE"])
    ws.add_data_validation(dv)
    dv.add(f"H2:H{MAX_ROW}")

    # Tophat_Radius (col I) — integer 1-50
    dv = _numeric_range(1, 50, allow_decimal=False, prompt="Top-hat filter radius in pixels. Range: 1-50.", title="Tophat Radius")
    ws.add_data_validation(dv)
    dv.add(f"I2:I{MAX_ROW}")

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

    # Cutoff_um (col F) — decimal 0.1-10
    dv = _numeric_range(0.1, 10.0, allow_decimal=True, prompt="Distance in microns. Typical: 0.5-2.0. Range: 0.1-10.", title="Cutoff (um)")
    ws.add_data_validation(dv)
    dv.add(f"F2:F{MAX_ROW}")


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
        "Apply_Warp": True,
        "Max_RANSAC_Distance": 0,
        "Overlap_Method": "Intersection",
        "Match_Threshold": 0,
        "Remove_Extranuclear_Puncta": True,
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
        "Apply_Warp": True,
        "Max_RANSAC_Distance": 0,
        "Overlap_Method": "Intersection",
        "Match_Threshold": 0,
        "Remove_Extranuclear_Puncta": True,
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
    warnings = []
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

        apply_warp = row.get("Apply_Warp", True)
        if pd.isna(apply_warp):
            apply_warp = True
        else:
            apply_warp = bool(apply_warp)

        max_ransac = row.get("Max_RANSAC_Distance", 0)
        if pd.isna(max_ransac):
            max_ransac = 0
        else:
            try:
                max_ransac = max(0, min(100, int(float(max_ransac))))
            except (ValueError, TypeError):
                warnings.append(f"Datasets row {row_num} ({name}): Invalid Max_RANSAC_Distance. Using 0 (auto).")
                max_ransac = 0

        overlap_method = str(row.get("Overlap_Method", "Intersection")).strip()
        if not overlap_method or overlap_method == "nan":
            overlap_method = "Intersection"
        if overlap_method not in ("Intersection", "Union"):
            errors.append(
                f"Datasets row {row_num} ({name}): Invalid Overlap_Method '{overlap_method}'. "
                f"Use: Intersection or Union"
            )

        match_threshold = row.get("Match_Threshold", 0)
        if pd.isna(match_threshold):
            match_threshold = 0
        else:
            try:
                match_threshold = max(0, min(100, int(float(match_threshold))))
            except (ValueError, TypeError):
                warnings.append(f"Datasets row {row_num} ({name}): Invalid Match_Threshold. Using 0 (auto).")
                match_threshold = 0

        remove_extra = row.get("Remove_Extranuclear_Puncta", True)
        if pd.isna(remove_extra):
            remove_extra = True
        else:
            remove_extra = bool(remove_extra)

        datasets.append({
            "name": name,
            "r1": Path(str(row["R1"]).strip()),
            "r2": Path(str(row["R2"]).strip()),
            "output_dir": out_dir,
            "r1_nuclear_channel": r1_nuc,
            "r2_nuclear_channel": r2_nuc,
            "seg_method": seg if (seg and seg != "nan") else "Classical",
            "merge_splits": merge,
            "apply_warp": apply_warp,
            "max_ransac_distance": max_ransac,
            "overlap_method": overlap_method,
            "match_threshold": match_threshold,
            "remove_extranuclear_puncta": remove_extra,
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

            def _float(col, default, min_val=None, max_val=None):
                v = row.get(col, default)
                if pd.isna(v):
                    return default
                try:
                    val = float(v)
                except (ValueError, TypeError):
                    warnings.append(f"Puncta row {row_num}: '{col}' has invalid value '{v}'. Using default ({default}).")
                    return default
                if min_val is not None and val < min_val:
                    warnings.append(f"Puncta row {row_num}: '{col}' value {val} below minimum {min_val}. Clamped.")
                    val = min_val
                if max_val is not None and val > max_val:
                    warnings.append(f"Puncta row {row_num}: '{col}' value {val} above maximum {max_val}. Clamped.")
                    val = max_val
                return val

            def _int(col, default, min_val=None, max_val=None):
                v = row.get(col, default)
                if pd.isna(v):
                    return default
                try:
                    val = int(float(v))
                except (ValueError, TypeError):
                    warnings.append(f"Puncta row {row_num}: '{col}' has invalid value '{v}'. Using default ({default}).")
                    return default
                if min_val is not None and val < min_val:
                    warnings.append(f"Puncta row {row_num}: '{col}' value {val} below minimum {min_val}. Clamped.")
                    val = min_val
                if max_val is not None and val > max_val:
                    warnings.append(f"Puncta row {row_num}: '{col}' value {val} above maximum {max_val}. Clamped.")
                    val = max_val
                return val

            def _bool(col, default):
                v = row.get(col, default)
                if pd.isna(v):
                    return default
                if isinstance(v, bool):
                    return v
                s = str(v).strip().upper()
                if s in ("TRUE", "1", "YES"):
                    return True
                if s in ("FALSE", "0", "NO"):
                    return False
                return default

            puncta_rules.append({
                "dataset": ds,
                "channel": ch,
                "params": {
                    "method": algo if (algo and algo != "nan") else "Local Maxima",
                    "threshold_rel": _float("Sensitivity", constants.PUNCTA_THRESHOLD_REL, min_val=0.0, max_val=1.0),
                    "min_distance": _int("Min_Distance", constants.PUNCTA_MIN_DISTANCE, min_val=1, max_val=20),
                    "sigma": _float("Sigma", constants.PUNCTA_SIGMA, min_val=0.0, max_val=5.0),
                    "nuclei_only": _bool("Nuclei_Only", True),
                    "use_tophat": _bool("Tophat", False),
                    "tophat_radius": _int("Tophat_Radius", constants.PUNCTA_TOPHAT_RADIUS, min_val=1, max_val=50),
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
            else:
                try:
                    cutoff = float(cutoff)
                    if cutoff < 0.1 or cutoff > 10.0:
                        warnings.append(f"Colocalization row {row_num}: Cutoff_um {cutoff} outside range 0.1-10. Clamped.")
                        cutoff = max(0.1, min(10.0, cutoff))
                except (ValueError, TypeError):
                    warnings.append(f"Colocalization row {row_num}: Invalid Cutoff_um. Using 1.0.")
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

    config = {
        "datasets": datasets,
        "puncta_rules": puncta_rules,
        "coloc_rules": coloc_rules,
    }
    # Attach warnings to config so the caller can display them
    if warnings:
        config["_warnings"] = warnings
    return config, None


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
