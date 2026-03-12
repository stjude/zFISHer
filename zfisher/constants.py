"""
Centralized constants for the zFISHer application.

This module contains hardcoded values, file paths, and parameters
used throughout the core processing and UI logic to ensure consistency
and ease of modification.
"""

# --- File and Directory Names ---
SESSION_FILENAME = "zfisher_session.json"
SEGMENTATION_DIR = "segmentation"
ALIGNED_DIR = "aligned"
CAPTURES_DIR = "captures"
INPUT_DIR = "input"
REPORTS_DIR = "reports"

# --- Layer Name Prefixes/Suffixes ---
ALIGNED_PREFIX = "Aligned"
WARPED_PREFIX = "Warped"
CENTROIDS_SUFFIX = "_centroids"
MASKS_SUFFIX = "_masks"
PUNCTA_SUFFIX = "_puncta"
CONSENSUS_MASKS_NAME = "Consensus_Nuclei_Masks"
CONSENSUS_IDS_SUFFIX = "_IDs"

# --- Segmentation Parameters ---
# Classical Nuclei Segmentation
NUC_SEG_Z_STEP = 2
NUC_SEG_SCALE_FACTOR = 0.5
NUC_SEG_GAUSSIAN_SIGMA = 3
NUC_SEG_OTSU_MIN_SIZE = 50
NUC_SEG_PEAK_MIN_DIST = 7
NUC_SEG_Z_XY_RATIO = 4.0           # Typical Z/XY physical spacing ratio for FISH data
NUC_SEG_MERGE_BOUNDARY_RATIO = 0.3 # Min shared boundary / smaller label surface ratio to merge adjacent labels

# Cellpose 3D Nuclei Segmentation (legacy, unused)
NUC_SEG_3D_Z_STEP = 2
NUC_SEG_3D_SCALE_FACTOR = 0.25
NUC_SEG_3D_STITCH_THRESH = 0.5
NUC_SEG_3D_BATCH_SIZE = 16

# Cellpose 2D-Stitched Nuclei Segmentation
NUC_SEG_CP2D_Z_STEP = 2
NUC_SEG_CP2D_SCALE_FACTOR = 0.25
NUC_SEG_CP2D_STITCH_THRESH = 0.3
NUC_SEG_CP2D_DIAMETER = None       # None = auto-estimate

# Puncta Detection
PUNCTA_MIN_DISTANCE = 2
PUNCTA_THRESHOLD_REL = 0.05
PUNCTA_SIGMA = 0.0
PUNCTA_TOPHAT_RADIUS = 15

# --- Registration Parameters ---
RANSAC_N_LIMIT = 2000
RANSAC_BIN_SIZE = 5.0
RANSAC_SEARCH_RADIUS = 100.0
RANSAC_RESIDUAL_THRESHOLD = 15
RANSAC_MAX_TRIALS = 2000
RANSAC_DEVIATION_THRESHOLD = 50.0
DEFORMABLE_DOWNSAMPLE_FACTOR = 16
DEFORMABLE_MESH_SIZE = 4
DEFORMABLE_ITERATIONS = 100
DEFORMABLE_SAMPLING_PERC = 0.01
DEFORMATION_GRID_SPACING = 50 # For visualization vector field
DEFORMATION_FIELD_NAME = "Deformation_Field"
WARPED_CHECKERBOARD_NAME = "Warped_Checkerboard"

# --- Analysis & Reporting ---
EXCEL_SUFFIX = ".xlsx"
METADATA_SHEET = "Metadata"
DISTANCES_SHEET = "Distances"
COLOCALIZATION_SHEET = "Colocalization"
TRI_COLOCALIZATION_SHEET = "Tri-Colocalization"
PER_NUCLEI_SHEET = "ROI per Nuclei"
STATS_SHEET = "Stats"
DISTRIBUTION_SHEET = "Distribution"

# --- UI Constants ---
# Channel color mapping
CHANNEL_COLORS = {
    # Nuclear stains
    "DAPI": "blue",
    "HOECHST": "blue",
    # Green fluorophores
    "FITC": "green",
    "GFP": "green",
    "EGFP": "green",
    "AF488": "green",
    "ALEXA488": "green",
    # Yellow/orange fluorophores
    "CY3": "yellow",
    "TRITC": "yellow",
    "YFP": "yellow",
    "AF555": "yellow",
    "ALEXA555": "yellow",
    # Red fluorophores
    "CY5": "red",
    "RFP": "red",
    "MCHERRY": "red",
    "AF647": "red",
    "ALEXA647": "red",
    # Magenta fluorophores
    "TXRED": "magenta",
    "AF594": "magenta",
    "ALEXA594": "magenta",
    # Cyan fluorophores
    "CFP": "cyan",
    "BFP": "blue",
}
DAPI_CHANNEL_NAME = "DAPI"  # Default fallback; actual name resolved per-session

# Known nuclear stain channel names (case-insensitive substring matching)
NUCLEAR_STAIN_NAMES = ["DAPI", "HOECHST", "DRAQ5", "DRAQ7", "405", "BFP"]