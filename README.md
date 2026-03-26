# zFISHer

**3D colocalization analysis for multiplexed sequential FISH (Fluorescence In Situ Hybridization), built on the napari viewer.**

zFISHer is a Python application that provides an end-to-end pipeline for analyzing 3D microscopy data from multiplexed sequential FISH experiments. It handles everything from raw image loading through nuclei segmentation, puncta detection, multi-round registration, deformable warping, consensus nuclei matching, and colocalization analysis — all within an interactive napari-based GUI.

The tool is designed for workflows where two imaging rounds (R1 and R2) each contain a shared nuclear stain channel (e.g., DAPI) and one or more FISH signal channels. zFISHer segments nuclei per-round, detects puncta on raw images, aligns the rounds using nuclear centroids, warps all channels and puncta coordinates into a common space, builds consensus nuclei masks, and quantifies colocalization across channels.

## Pipeline Overview

zFISHer is organized into five workflow steps, accessible from the sidebar:

### 1. Session & I/O

- **New Session** — Select Round 1 and Round 2 image files (`.nd2`, `.tif`, `.tiff`, `.ome.tif`) and an output directory. zFISHer automatically splits multi-channel stacks into individual channels and loads them into the viewer.
- **Load Session** — Resume a previous analysis from a saved `zfisher_session.json` file. All layers, masks, puncta, and settings are restored. A new session file is created (e.g., `_2.json`) to preserve the original.
- **Autorun** — Executes the full pipeline (segmentation, puncta detection, registration, warping, consensus, puncta transformation) in one click with configurable parameters.
- **Batch Processing** — Run the full pipeline on multiple datasets from an Excel template. Generate a template, fill in dataset paths and parameters, then process all FOVs unattended. See [Batch Processing](#batch-processing) for details.

All intermediate results (aligned TIFFs, segmentation masks, puncta CSVs) are saved to the output directory and tracked in the session file for reproducibility.

### 2. Nuclei Segmentation

#### Nuclei Mapping
Segments nuclei from the nuclear stain channel using one of two methods:

- **Classical (Fast)** — Otsu thresholding + watershed on a downsampled MIP, then expanded back to full resolution. Fast and parameter-free. Best for most datasets.
- **Cellpose (Deep Learning)** — GPU-accelerated deep learning segmentation using the Cellpose `nuclei` model. More accurate for dense or irregular nuclei but slower. Requires cellpose to be installed.

Options:
- **Merge Over-segmented Nuclei** — Automatically merges nuclei fragments that were incorrectly split by the segmentation algorithm (default: on).

Produces labeled masks where each nucleus has a unique integer ID, a centroids layer for registration, and an `_IDs` overlay layer displaying nucleus IDs at each centroid. All layers are locked against accidental deletion.

#### Mask Editor
Interactive tools for manually refining segmentation masks:

- **Target Layer** — Select which mask layer to edit (R1 or R2). Selecting a layer in the dropdown also makes it visible in the viewer.
- **Merge Nuclei** — Specify two nucleus IDs (A and B) to merge them into a single label. Nucleus A is absorbed into Nucleus B.
- **Paint New Mask** — Brush tool to paint a new nucleus with the next available ID. Includes an eyedropper to pick existing IDs and a configurable brush size.
- **Extrude Mask** — Fill a specified nucleus through all Z slices using its largest XY cross-section. Useful for correcting incomplete 3D segmentation.
- **Erase** — Brush eraser with configurable size. Includes an eyedropper and a Delete ID button to remove an entire nucleus by ID.
- **Hover Edit Mode** — Highlights the nucleus under the cursor in red. Press **C** to delete the highlighted nucleus.
- **Undo** — Reverts the last mask editing operation (merge, paint, erase, delete, extrude).

All edits are auto-saved to disk. Deleting a mask layer automatically removes its associated `_IDs` and `_centroids` layers.

### 3. Puncta Detection

Puncta are detected on **raw (unaligned) images** using per-round nuclei masks. This ensures detection is performed on the original signal before any interpolation from alignment.

#### Algorithmic Detection
Detects fluorescent puncta using one of four algorithms:

| Algorithm | Best For | Key Parameters |
|---|---|---|
| **Local Maxima** | Well-separated, bright spots | Min Distance, Sigma (pre-blur) |
| **Laplacian of Gaussian** | Varying spot sizes, crowded fields | Sigma, Z/XY Scale Ratio |
| **Difference of Gaussian** | General-purpose (fast LoG approximation) | Sigma, Z/XY Scale Ratio |
| **Radial Symmetry** | High-density fields (very sensitive) | — (minimal filtering) |

Detection parameters:
- **Sensitivity (0-1)** — Relative intensity threshold. Lower values detect dimmer spots but may increase false positives.
- **Min Distance (px)** — Minimum separation between detected puncta. Prevents double-counting.
- **Spot Radius, px (Sigma)** — Gaussian sigma matching the approximate puncta radius.
- **Z/XY Scale Ratio** — Corrects for non-cubic voxels. Auto-computed from image metadata.
- **Nuclei Only** — Discard puncta outside the nuclei mask (default: on).
- **Top-hat Background Subtraction** — Removes uneven background illumination per-slice before detection.

When re-running detection on a channel that already has puncta, a dialog asks whether to **Replace** (clear and re-detect), **Merge** (add new detections, deduplicated by minimum distance), or **Cancel**.

Each detected spot is annotated with: coordinates (Z, Y, X), nucleus ID, peak intensity, and signal-to-noise ratio (SNR). Results are saved as CSV files.

#### Manual Puncta Editor
Interactive tools for curating detected puncta:

- **Tools** — Mode buttons for Delete Selected, Add Point, Select, and Move Camera. Only one mode is active at a time.
- **Fishing Hook** — Smart placement mode: hold **F** + click to cast a ray through the volume along the camera direction, find the brightest voxel, and optionally refine to the local intensity peak. Ideal for accurate 3D placement from any view angle. Configurable volume optimization radius.
- **Erase** — Delete selected puncta, clear all points in a layer, or delete an entire layer (with undo support).
- **Undo** — Reverts the last editing operation, including layer deletion.

The editor also provides custom layer controls for each puncta layer: point size, symbol color, text size, text color, and a "draw on top of masks" toggle.

### 4. Alignment & Consensus

This step aligns both rounds into a common coordinate space and builds consensus nuclei. It can be run automatically or stepped through manually.

#### Automated Preprocessing
Runs the full alignment pipeline end-to-end with configurable parameters:

- **Max RANSAC Distance, px (0=auto)** — Controls centroid pair matching strictness.
- **Apply B-spline Warp** — Enable elastic warping for tissue deformation correction. Disable for rigid-only alignment.
- **Create Consensus Nuclei Mask** — Toggle consensus mask generation.
- **Overlap Method** — Intersection (conservative) or Union (permissive).
- **Match Threshold, px (0=auto)** — Controls nuclei matching across rounds.
- **Remove Extranuclear Puncta** — Discard puncta outside the consensus mask.
- **Show Checkerboard / Deformation Field** — Toggle diagnostic visualization layers.

#### Manual Workflow

**Registration** — Aligns R1 and R2 using their nuclear centroid clouds:

1. **Coarse alignment** — Vector voting via histogram binning finds a rough translation. Robust to outliers and unequal point counts. Input is subsampled for speed on large datasets.
2. **Fine alignment** — RANSAC with a 3-DOF translation model refines the shift using matched centroid pairs.
3. **Validation** — Reports RMSD and rejects the refined shift if it deviates too far from the coarse estimate.

A confirmation dialog warns before overwriting an existing alignment.

**Global Canvas (Warp)** — Creates the aligned multi-channel canvas:

1. **Rigid alignment** — Pads and shifts all channels based on the registration translation. Integer and fractional shift components are separated for sub-pixel accuracy.
2. **B-spline elastic warping** — Computes a B-spline transform from the aligned nuclear pairs to correct non-rigid tissue deformation. Applied per-channel with appropriate interpolation:
   - Nuclear/intensity channels — B-spline interpolation (smooth)
   - Label/mask channels — Nearest-neighbor interpolation (preserves integer IDs)
   - FISH signal channels — Linear interpolation (avoids ringing artifacts)
3. **Puncta coordinate transformation** — Raw puncta are mathematically transformed into aligned space. R1 puncta are rigidly shifted; R2 puncta are shifted then inverse B-spline warped via fixed-point iteration. Original raw layers are replaced with aligned layers.

Existing aligned layers are automatically removed before regeneration to prevent duplicates.

**Nuclei Matching (Consensus)** — Builds a consensus nuclear segmentation:

- Matches nuclei across rounds by centroid proximity (configurable threshold)
- Relabels R2 masks to match R1 IDs
- Merges via **Intersection** (only nuclei present in both rounds) or **Union** (all nuclei from either round)
- Removes small artifacts with `remove_small_objects`
- Outputs a single `Consensus_Nuclei_masks` layer with consistent IDs
- **Reassigns all puncta Nucleus_IDs** from the consensus mask
- **Remove Extranuclear Puncta** — Deletes puncta outside the consensus mask (default: on)

### 5. Export & Visualization

#### Colocalization Analysis
Quantifies spatial relationships between puncta from different channels:

- **Pairwise Rules** — Define source → target channel pairs with a distance cutoff (in microns). Measures nearest-neighbor distances using KD-trees in world coordinates.
- **Three-Channel Rules** — Define an anchor channel that must be within the cutoff of both Channel A and Channel B simultaneously. Uses greedy triplet matching to prevent over-counting.
- **Duplicate Prevention** — Identical rules are silently rejected.
- **Excel Export** — Generates a multi-sheet `.xlsx` report with:
  - Metadata (session info, parameters, file paths)
  - Pairwise nearest-neighbor distances
  - Colocalization calls (within/outside threshold)
  - Tri-colocalization results
  - Per-nucleus ROI summaries
  - Descriptive statistics
  - Distribution data

#### Puncta Cleanup (Refilter)
Remove puncta that fall outside a specified nuclei mask. Useful after mask editing to re-enforce nuclear boundaries:

- Select a mask layer and which puncta channels to filter
- Undo support (up to 5 operations)

#### Capture & Annotation
Tools for creating publication-quality images:

- **Screenshot Capture** — Saves the current viewer canvas as PNG. Hotkey: **Shift+P**
- **Region Capture** — Click and drag to capture a specific region. Hotkey: **Shift+G**
- **Arrow Annotations** — Draw arrows on the canvas for highlighting features:
  - **A** — Start/end arrow drawing
  - **D** — Delete nearest arrow
  - **Ctrl+Z** — Undo last arrow
  - **Escape** — Cancel current arrow
- **Scale Bar** — Draggable on-canvas scale bar with toggles for visibility, lock position, and pixel display
- Captures are saved to the `captures/` directory

## Keyboard Shortcuts

| Shortcut | Action | Context |
|---|---|---|
| **Shift+P** | Capture screenshot | Always |
| **Shift+G** | Region capture (click+drag) | Always |
| **X** | Delete puncta point under cursor | Always |
| **C** | Delete mask/nucleus under cursor | Always |
| **F** (hold) + click | Fishing hook puncta placement | Puncta Editor with fishing hook enabled |
| **A** | Start/end arrow annotation | Arrow drawing mode |
| **D** | Delete nearest arrow | Arrow drawing mode |
| **Ctrl+Z** | Undo last arrow | Arrow drawing mode |
| **Escape** | Cancel arrow in progress | Arrow drawing mode |

## Batch Processing

zFISHer supports unattended processing of multiple datasets via an Excel template:

1. Click **Generate Template** in the Batch Process tab to create a pre-formatted `.xlsx` file
2. Fill in the **Datasets** sheet with image paths and per-dataset settings
3. Optionally configure **Puncta** sheet for per-channel detection parameters
4. Optionally configure **Colocalization** sheet for analysis rules
5. Load the file and click **Run Batch Processing**

The template includes:
- **Instructions sheet** — Detailed field-by-field documentation
- **Dropdown validations** — Constrained choices for methods, algorithms, and boolean fields
- **Numeric range validations** — Prevents out-of-range values for sensitivity, distances, etc.
- **Example rows** — Pre-filled reference rows (ignored during processing)
- **Per-dataset overrides** — Puncta and colocalization rules can use `ALL` for defaults or a specific dataset name for overrides

All errors and warnings (missing files, invalid values, unrecognized channel names) are reported in a popup before processing begins, with the option to continue or cancel.

## Installation

zFISHer requires a Conda environment with Python 3.10.

1. **Clone the repository:**
    ```bash
    git clone https://github.com/stjude/zFISHer.git
    cd zFISHer
    ```

2. **Create and activate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate zFISHer
    ```

    If `environment.yml` is not available, install manually:
    ```bash
    conda create -n zFISHer python=3.10
    conda activate zFISHer
    pip install napari[all] magicgui scikit-image scipy numpy pandas tifffile SimpleITK cellpose openpyxl
    ```

## Usage

Launch the application from the project root:

```bash
python main.py
```

This opens the napari viewer with the zFISHer sidebar containing all five workflow steps.

### Typical Workflow

1. **Session & I/O** — Create a new session, select R1 and R2 image files, and choose an output directory
2. **Nuclei Segmentation** — Segment nuclear channels to create per-round masks, then edit masks as needed
3. **Puncta Detection** — Run algorithmic detection on each raw FISH channel, then manually curate with the editor and fishing hook
4. **Alignment & Consensus** — Register rounds → Elastic warping to common space → Transform puncta → Match nuclei into consensus mask → Remove puncta outside nuclei
5. **Export & Visualization** — Define colocalization rules, run analysis, export the Excel report, and capture images

Alternatively, use **Autorun** from Session & I/O to execute steps 2–4 automatically, or **Batch Processing** for multiple datasets.

### Session Persistence

All processing results are saved incrementally:
- Aligned TIFFs in `aligned/`
- Segmentation masks and centroids in `segmentation/`
- Puncta CSVs and analysis reports in `reports/`
- Screenshots and annotations in `captures/`
- Input files (converted OME-TIFFs and metadata) in `input/`
- Session state in `zfisher_session_N.json`

Loading a session restores all layers and processing state, so you can close and resume at any point.

## Project Structure

```
zFISHer/
  main.py                    # Application entry point
  environment.yml            # Conda environment specification
  zfisher/
    constants.py             # Centralized parameters and naming conventions
    version.py               # Version string (auto-appends git hash for dev builds)
    core/                    # Scientific computation (no UI dependencies)
      segmentation.py        # Classical + Cellpose nuclei segmentation, consensus matching
      registration.py        # RANSAC alignment, B-spline warping, parallel channel processing
      puncta.py              # Spot detection, coordinate transformation, quality metrics
      pipeline.py            # Headless full pipeline (used by autorun and batch processing)
      analysis.py            # Colocalization distances, per-nucleus aggregation
      report.py              # Excel report generation
      session.py             # Session state management and persistence
      io.py                  # ND2/TIFF file I/O, channel splitting, metadata extraction
      generate_batch_template.py  # Batch Excel template generation and parsing
    ui/                      # napari GUI layer
      viewer.py              # Main viewer setup, scale bar, toolbar, hotkey bindings
      events.py              # Layer event handlers (auto-save, cascade deletion, layer locking)
      style.py               # Theme colors and Qt stylesheets
      decorators.py          # @require_active_session, @error_handler
      popups.py              # Progress dialogs, error popups, confirmation dialogs
      viewer_helpers.py      # Layer creation, update, and ID management utilities
      widgets/               # Individual UI panels
        home_widget.py             # Home screen with version, workflow overview, reset
        start_session_widget.py    # Session & I/O composite (new, load, batch)
        new_session_widget.py      # New session creation
        load_session_widget.py     # Session loading
        batch_process_widget.py    # Batch processing from Excel template
        nuclei_segmentation_widget.py  # Step 2 composite (mapping + mask editor)
        dapi_segmentation_widget.py    # Nuclear channel segmentation
        mask_editor_widget.py          # Interactive mask editing (merge, paint, erase, extrude)
        puncta_picking_widget.py   # Step 3 composite (detection + editor)
        puncta_widget.py           # Algorithmic puncta detection
        puncta_editor_widget.py    # Manual puncta editor (fishing hook, tools, undo)
        alignment_consensus_widget.py  # Step 4 composite (auto + manual)
        automated_preprocessing_widget.py  # One-click alignment pipeline
        registration_widget.py     # RANSAC registration
        canvas_widget.py           # Global canvas generation + puncta transform
        nuclei_matching_widget.py  # Consensus nuclei + puncta reassignment
        export_visualization_widget.py  # Step 5 composite (coloc + capture)
        colocalization_widget.py   # Pairwise + tri-channel colocalization rules + export
        refilter_puncta_widget.py  # Post-edit puncta cleanup by mask
        capture_widget.py          # Screenshot, region capture, arrow annotations, scale bar
        _shared.py                 # Shared UI helpers (dividers, headers, nuclear channel detection)
```

## Technical Details

- **Registration**: Two-stage approach — vector voting via histogram binning for coarse alignment (robust to outliers, subsampled for speed), then RANSAC with a constrained 3-DOF translation model for robust refinement.
- **Elastic Warping**: SimpleITK B-spline transform computed from aligned nuclear pairs. The transform is reused across all channels (since channels in a round share the same physical distortion), with per-channel interpolation strategy to preserve data integrity (nearest-neighbor for labels, linear for signals).
- **Puncta Detection**: Four algorithms with optional top-hat background subtraction. Each spot is scored with intensity and SNR metrics. Puncta are detected on raw images using per-round masks, then mathematically transformed into aligned space. Nuclear assignment is updated from the consensus mask.
- **Puncta Coordinate Transform**: R1 puncta are rigidly shifted by the canvas offset. R2 puncta are shifted then inverse B-spline warped using an iterative fixed-point method (converges to sub-pixel accuracy). Extranuclear puncta can be automatically removed.
- **Colocalization**: KD-tree nearest-neighbor search in world coordinates (microns). Supports pairwise and three-channel analysis with configurable distance thresholds. Greedy triplet matching ensures no punctum participates in more than one triplet.
- **Performance**: Channel warping and TIFF I/O are parallelized via `ThreadPoolExecutor`. SimpleITK's C++ backend releases the GIL, enabling true concurrent execution. Progress dialogs use targeted widget repainting to avoid GL context corruption during long operations.
- **Layer Management**: Custom layer controls hide unnecessary napari defaults and add context-specific controls (text size/color, symbol color, draw-on-top toggle). Layer locking prevents accidental deletion of critical data. Vispy buffer updates use a `__indices_view` cache-clearing workaround to prevent IndexError/GL crashes when point counts change.
- **Versioning**: Version string is defined in `version.py` and auto-appends the git commit hash for development builds (e.g., `0.1.0-dev+abc1234`).

## Contributing

This project was developed by Seth Staller. At this time, it is not actively seeking external contributions. For questions or licensing inquiries, please contact Seth.Staller@STJUDE.ORG.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
