# zFISHer

**Advanced 3D colocalization analysis for multiplexed sequential FISH (Fluorescence In Situ Hybridization), built for the Napari viewer.**

zFISHer is a Python application that provides an end-to-end pipeline for analyzing 3D microscopy data from multiplexed sequential FISH experiments. It handles everything from raw image loading through segmentation, puncta detection, registration, and colocalization analysis — all within an interactive napari-based GUI.

The tool is designed for workflows where two imaging rounds (R1 and R2) each contain a shared DAPI channel and one or more FISH signal channels. zFISHer segments nuclei per-round, detects puncta on raw images, then aligns the rounds using DAPI, warps all channels and puncta coordinates into a common space, builds consensus nuclei masks, and quantifies colocalization across channels.

## Pipeline Overview

zFISHer is organized into five workflow steps:

### 1. Session & I/O

- **New Session** — Select Round 1 and Round 2 image files (`.nd2`, `.tif`, `.tiff`, `.ome.tif`) and an output directory. zFISHer automatically splits multi-channel stacks into individual channels (DAPI, FITC, CY3, CY5, TXRED, etc.) and loads them into the viewer.
- **Load Session** — Resume a previous analysis from a saved `zfisher_session.json` file.
- **Autorun** — Executes the full pipeline (segmentation, puncta detection, registration, warping, consensus, puncta transformation) in one click.
- **Batch Processing** — Run the full pipeline on multiple datasets from an Excel file.

All intermediate results (aligned TIFFs, segmentation masks, puncta CSVs) are saved to the output directory and tracked in the session file for reproducibility.

### 2. Nuclei Segmentation

#### DAPI Mapping
Segments nuclei from DAPI channels using one of two methods:

- **Classical** — Otsu thresholding + watershed on a downsampled MIP, then expanded back to full resolution. Fast and parameter-free.
- **Cellpose 3D** — GPU-accelerated deep learning segmentation using the Cellpose `nuclei` model. Processes slices in batches with 3D stitching for volumetric consistency. More accurate for dense or irregular nuclei.

Produces labeled masks where each nucleus has a unique integer ID, a centroids layer for registration, and an `_IDs` overlay layer displaying nucleus IDs at each centroid. Protected layers (raw channels, DAPI masks, centroids, ID overlays) are locked against accidental deletion.

#### Mask Editor
Interactive tools for manually refining segmentation masks:

- **Merge Nuclei** — Click two adjacent nuclei to merge them into a single label
- **Paint New Mask** — Brush tool to paint a new nucleus with a unique ID. Configurable brush radius.
- **Erase** — Removes mask regions under the cursor. Configurable erase radius. Activated with **Shift+E** hotkey.
- **Delete Mask Under Cursor** — Removes an entire nucleus label at the cursor position. Hotkey: **C**
- **Extrude to 3D** — Takes a 2D painted mask and extrudes it through a specified Z-range
- **Undo/Redo** — Reverts the last mask editing operation

All edits are auto-saved to the session output directory. Deleting a mask layer automatically removes its associated `_IDs` overlay and vice versa.

### 3. Puncta Picking

Puncta are detected on **raw (unaligned) images** using per-round DAPI masks. This ensures detection is performed on the original signal before any interpolation from alignment. The puncta widget auto-selects the correct nuclei mask based on the chosen channel (R1 → R1 DAPI mask, R2 → R2 DAPI mask, aligned/warped → consensus mask).

#### Algorithmic Detection
Detects fluorescent puncta using one of four algorithms:

| Algorithm | Best For | Key Parameters |
|---|---|---|
| **Local Maxima** | Well-separated, bright spots with low background | Min Distance, Sigma (pre-blur) |
| **Laplacian of Gaussian (LoG)** | Varying spot sizes, anisotropic Z | Sigma, Z-Anisotropy |
| **Difference of Gaussian (DoG)** | General-purpose FISH data (fast LoG approximation) | Sigma, Z-Anisotropy |
| **Radial Symmetry** | High-density transcript fields (very sensitive) | — (min distance = 1px) |

Additional options:
- **Nuclei Only** — Discard extranuclear puncta (keeps only spots within a labeled nucleus)
- **Deconvolve** — Richardson-Lucy deconvolution to sharpen crowded fields before detection
- **Top-hat Background Subtraction** — Removes uneven background illumination per-slice
- **Z-Anisotropy Scale** — Auto-detected from layer metadata; compensates for different Z vs XY resolution

Each detected spot is annotated with: coordinates (Z, Y, X), nucleus ID, peak intensity, and signal-to-noise ratio (SNR). Results are saved as CSV files.

#### Manual Puncta Editor
Interactive tools for curating detected puncta:

- **Add Mode** — Click to place new puncta on the active points layer
- **Delete Point Under Cursor** — Hotkey: **X**
- **Color Picker** — Change the display color of puncta for a channel
- Auto-saves edits to CSV via attached event listeners

### 4. Alignment & Consensus

This step aligns both rounds into a common coordinate space and builds consensus nuclei. It can be run automatically or stepped through manually.

#### Automated Preprocessing
Runs the full alignment pipeline end-to-end: DAPI segmentation → registration → B-spline warping → consensus nuclei → puncta coordinate transformation. Useful when segmentation and puncta picking have not yet been done.

#### Manual Workflow

**Registration** — Aligns R1 and R2 using their DAPI centroid clouds:

1. **Coarse alignment** — Nearest-neighbor median voting finds a rough translation vector.
2. **Fine alignment** — RANSAC with a 3-DOF translation model refines the shift using matched centroid pairs. Operates in physical space (microns) to handle anisotropic voxels correctly.
3. **Validation** — Reports RMSD and rejects the refined shift if it deviates too far from the coarse estimate.

**Global Canvas (Warp)** — After registration, creates the aligned multi-channel canvas and transforms puncta:

1. **Rigid alignment** — Pads and shifts all channels based on the RANSAC translation.
2. **B-spline deformable warping** — Computes a B-spline transform from the aligned DAPI pair to correct non-rigid tissue deformation. The transform is then applied to all channels:
   - **DAPI channels** — B-spline interpolation (smooth, artifact-free on dense signals)
   - **Label/mask channels** — Nearest-neighbor interpolation (preserves integer labels)
   - **FISH signal channels** — Linear interpolation (avoids ringing/Gibbs artifacts on sparse bright puncta)
3. **Puncta coordinate transformation** — All existing raw puncta Points layers are mathematically transformed into the aligned/warped coordinate space. R1 puncta are rigidly shifted; R2 puncta are shifted and then inverse B-spline warped. Original raw puncta layers are replaced with properly named aligned layers (e.g., "Aligned R1 - FITC_puncta", "Warped R2 - Cy5_puncta").
4. **Parallel execution** — Per-channel warping and TIFF saves are parallelized using `ThreadPoolExecutor` since SimpleITK releases the GIL.

**Nuclei Matching (Consensus)** — Builds a consensus nuclear segmentation from the aligned R1 and R2 DAPI masks:

- Matches nuclei across rounds by centroid proximity
- Relabels R2 masks to match R1 IDs
- Merges via **Union** (include all nuclei from both rounds) or **Intersection** (only nuclei present in both rounds)
- Removes small artifacts with `remove_small_objects`
- Outputs a single `Consensus_Nuclei_Masks` layer with consistent IDs for downstream analysis
- **Reassigns all puncta Nucleus_IDs** from the consensus mask so every puncta point references the correct consensus nucleus
- **Remove Extranuclear Puncta** (default on) — Deletes puncta that fall outside the consensus mask boundaries. Points outside any nucleus are discarded from both the viewer and saved CSVs. When toggled off, extranuclear puncta are kept but assigned `Nucleus_ID = 0`.

### 5. Export & Visualization

#### Colocalization Analysis
Quantifies spatial relationships between puncta from different channels:

- **Rule Builder** — Define colocalization rules by selecting source and target puncta layers with a distance threshold (in microns)
- **Nearest-Neighbor Distances** — Computes all pairwise nearest-neighbor distances between point clouds using KD-trees in physical (world) coordinates
- **Per-Nucleus Statistics** — Aggregates counts and colocalization results per consensus nucleus ID
- **Excel Export** — Generates a multi-sheet `.xlsx` report with:
  - Metadata (session info, parameters)
  - Pairwise distances
  - Colocalization calls
  - Tri-colocalization (3-way overlap)
  - Per-nuclei ROI summaries
  - Descriptive statistics
  - Distribution data

#### Capture & Annotation
Tools for creating publication-quality images:

- **Screenshot Capture** — Saves the current viewer canvas. Hotkey: **Shift+P**
- **Region Capture** — Select and capture a specific region. Hotkey: **Ctrl+A**
- Captures are saved to the `captures/` directory in the session output folder

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| **Shift+P** | Capture screenshot |
| **Ctrl+A** | Region capture |
| **X** | Delete puncta point under cursor |
| **C** | Delete mask/nucleus under cursor |
| **Shift+E** | Erase mask region at cursor |

## Installation

zFISHer requires a Conda environment with Python 3.10.

1. **Clone the repository:**
    ```bash
    git clone <repository-url>
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
2. **Nuclei Segmentation** — Segment DAPI channels to create per-round masks, then edit masks as needed
3. **Puncta Picking** — Run algorithmic detection on each raw FISH channel (using per-round DAPI masks), then manually curate with the editor
4. **Alignment & Consensus** — Register rounds → Warp to common space → Transform puncta coordinates → Match nuclei into consensus mask → Remove extranuclear puncta
5. **Export & Visualization** — Define colocalization rules, run analysis, export the Excel report, and capture images

Alternatively, use **Autorun** from Session & I/O to execute steps 2–4 automatically in one click.

### Session Persistence

All processing results are saved incrementally:
- Aligned TIFFs in `aligned/`
- Segmentation masks in `segmentation/`
- Puncta CSVs in `reports/`
- Captures in `captures/`
- Session state in `zfisher_session.json`

Loading a session restores all layers and processing state, so you can close and resume at any point.

## Project Structure

```
zFISHer/
  main.py                    # Application entry point
  environment.yml            # Conda environment specification
  zfisher/
    constants.py             # Centralized parameters and file paths
    core/                    # Scientific computation (no UI dependencies)
      segmentation.py        # Classical + Cellpose nuclei segmentation, consensus matching
      registration.py        # RANSAC alignment, B-spline warping, parallel channel processing
      puncta.py              # Spot detection, coordinate transformation, quality metrics
      pipeline.py            # Headless full pipeline (used by autorun and batch processing)
      analysis.py            # Colocalization distances, per-nucleus aggregation
      report.py              # Excel report generation
      session.py             # Session state management and persistence
      io.py                  # ND2/TIFF file I/O and channel splitting
    ui/                      # napari GUI layer
      viewer.py              # Main viewer setup, toolbar, hotkey bindings
      events.py              # Layer event handlers (auto-save, auto-select, layer locking)
      style.py               # Theme colors and Qt stylesheets
      decorators.py          # @require_active_session, @error_handler
      popups.py              # Progress dialogs and error popups
      viewer_helpers.py      # Layer creation and update utilities
      widgets/               # Individual UI panels
        start_session_widget.py    # Session & I/O (new, load, autorun)
        new_session_widget.py      # New session creation + autorun pipeline
        batch_process_widget.py    # Batch processing from Excel
        nuclei_segmentation_widget.py  # Step 2 composite (DAPI + mask editor)
        dapi_segmentation_widget.py    # DAPI mapping
        mask_editor_widget.py          # Interactive mask editing
        puncta_picking_widget.py   # Step 3 composite (detection + editor)
        puncta_widget.py           # Algorithmic puncta detection
        puncta_editor_widget.py    # Manual puncta curation
        alignment_consensus_widget.py  # Step 4 composite (registration + canvas + consensus)
        automated_preprocessing_widget.py  # One-click alignment pipeline
        registration_widget.py     # RANSAC registration
        canvas_widget.py           # Global canvas + puncta transform
        nuclei_matching_widget.py  # Consensus nuclei + puncta reassignment
        export_visualization_widget.py  # Step 5 composite (coloc + capture)
        colocalization_widget.py   # Colocalization analysis
        capture_widget.py          # Screenshot and region capture
    utils/                   # Logging configuration
```

## Technical Details

- **Registration**: Two-stage approach — nearest-neighbor median voting for coarse alignment, then RANSAC with a constrained 3-DOF translation model for robust refinement. Operates in physical space to handle anisotropic voxels.
- **Deformable Warping**: SimpleITK B-spline transform computed from aligned DAPI pairs. The spatial transform is reused across all channels (since all channels in a round share the same physical distortion), with per-channel interpolation strategy to avoid artifacts.
- **Puncta Detection**: Supports 4 algorithms with optional preprocessing (deconvolution, top-hat filtering). Each spot is scored with intensity and SNR metrics. Puncta are detected on raw images using per-round masks, then mathematically transformed into aligned space. Nuclear assignment is updated from the consensus mask after alignment.
- **Puncta Coordinate Transform**: R1 puncta are rigidly shifted by the canvas offset. R2 puncta are shifted, then inverse B-spline warped using an iterative fixed-point method. Extranuclear puncta can be automatically removed after consensus matching.
- **Colocalization**: KD-tree nearest-neighbor search in world coordinates (microns). Supports pairwise and tri-channel analysis with configurable distance thresholds.
- **Performance**: Channel warping and TIFF I/O are parallelized via `ThreadPoolExecutor`. SimpleITK's C++ backend releases the GIL, enabling true concurrent execution.

## Contributing

This project was developed by Seth Staller. At this time, it is not actively seeking external contributions. For questions or licensing inquiries, please contact Seth.Staller@STJUDE.ORG.

## License

All rights reserved. Contact St. Jude Children's Research Hospital for licensing details.
