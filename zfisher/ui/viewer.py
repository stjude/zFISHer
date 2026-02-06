import napari
import json
from magicgui import magicgui, widgets
import webbrowser
from pathlib import Path
from zfisher.core.io import load_nd2
from zfisher.core.registration import (
    segment_nuclei_classical, 
    align_centroids_ransac, 
    align_and_pad_images,
    calculate_deformable_transform,
    apply_deformable_transform
)
from zfisher.core.segmentation import detect_spots_3d
import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon
from qtpy.QtCore import QTimer
import os
import concurrent.futures
from qtpy.QtWidgets import QToolBox, QVBoxLayout, QWidget

# Define your paths as constants at the top for easy editing later
DEFAULT_R1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
DEFAULT_R2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")

# Global variable to store the calculated shift
CALCULATED_SHIFT = None

# Global Session State
SESSION_DATA = {
    "output_dir": None,
    "r1_path": None,
    "r2_path": None,
    "shift": None,
    "processed_files": {} # key: layer_name, value: relative_path
}

def save_session():
    """Saves the current session state to a JSON file."""
    if not SESSION_DATA.get("output_dir"): return
    try:
        out_path = Path(SESSION_DATA["output_dir"]) / "zfisher_session.json"
        with open(out_path, 'w') as f:
            # Use default=str to handle Path objects and numpy arrays if needed
            json.dump(SESSION_DATA, f, indent=4, default=str)
        print(f"Session saved: {out_path}")
    except Exception as e:
        print(f"Failed to save session: {e}")

# Helper to map metadata names to colors
CHANNEL_COLORS = {
    "DAPI": "blue",
    "FITC": "green",
    "CY3": "yellow",
    "CY5": "red",
    "TXRED": "magenta"
}

@magicgui(
    call_button="Load Data",
    round1_path={"label": "Round 1 (.nd2)", "filter": "*.nd2"},
    round2_path={"label": "Round 2 (.nd2)", "filter": "*.nd2"},
    output_dir={"label": "Output Directory", "mode": "d"},
    auto_call=False,
)
def file_selector_widget( # No viewer argument
    round1_path: Path = DEFAULT_R1,
    round2_path: Path = DEFAULT_R2,
    output_dir: Path = Path.home() / "zFISHer_Output",
    _save_session: bool = True # Flag to control session saving
):
    """Loads ND2 files, sets up output directories, and initializes session."""
    viewer = napari.current_viewer()
    
    # 1. Setup Output Directories
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    (output_dir / "segmentation").mkdir(exist_ok=True)
    (output_dir / "aligned").mkdir(exist_ok=True)
    
    # If starting a new session, clear old data and save initial state.
    # If loading, this is skipped to avoid overwriting the loaded session.
    if _save_session:
        viewer.layers.clear()
        global CALCULATED_SHIFT
        CALCULATED_SHIFT = None
        SESSION_DATA["output_dir"] = str(output_dir)
        SESSION_DATA["r1_path"] = str(round1_path)
        SESSION_DATA["r2_path"] = str(round2_path)
        SESSION_DATA["shift"] = None
        SESSION_DATA["processed_files"] = {}
        save_session()
    
    for path, prefix in [(round1_path, "R1"), (round2_path, "R2")]:
        if not path.exists():
            print(f"Error: {path} not found.")
            continue
            
        session = load_nd2(str(path))
        
        # YOUR DATA SHAPE: (71, 3, 2044, 2048) -> (Z, C, Y, X)
        # NAPARI EXPECTS CHANNELS AT INDEX 1 IF WE WANT TO SPLIT THEM
        # We move axis 1 (Channels) to the front so it becomes (C, Z, Y, X)
        data_swapped = np.moveaxis(session.data, 1, 0)
        
        # Print dimensions for the user
        print(f"Loaded {prefix}: {data_swapped.shape[0]} channels, {data_swapped.shape[1]} Z-slices. Full shape: {data_swapped.shape}")
        
        # Now shape is (3, 71, 2044, 2048)
        # Axis 0 = 3 channels
        # Axis 1 = 71 Z-slices
        
        new_layers = viewer.add_image(
            data_swapped,
            name=[f"{prefix} - {ch}" for ch in session.channels],
            channel_axis=0,        # Now correctly sees 3 channels
            scale=session.voxels,   # Matches the (71, 2044, 2048) ZYX stack
            blending="additive"
        )

        # Apply colors
        for layer in new_layers:
            for ch_name, color in CHANNEL_COLORS.items():
                if ch_name.upper() in layer.name.upper():
                    layer.colormap = color
            
            if "DAPI" not in layer.name.upper():
                layer.visible = False

    # Force the Z-slider to appear for the 71 slices
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.reset_view()

@magicgui(
    call_button="Run DAPI Mapping",
    r1_layer={"label": "Round 1 (DAPI)"},
    r2_layer={"label": "Round 2 (DAPI)"},
    auto_call=False,
)
def dapi_segmentation_widget(
    r1_layer: "napari.layers.Image",
    r2_layer: "napari.layers.Image"
):
    """Runs segmentation on selected DAPI channels."""
    viewer = napari.current_viewer()
    layers_to_process = [l for l in [r1_layer, r2_layer] if l is not None]
    
    if not layers_to_process:
        viewer.status = "No channels selected."
        return

    viewer.status = f"Segmenting {len(layers_to_process)} layer(s)..."

    for layer in layers_to_process:
        masks, centroids = segment_nuclei_classical(layer.data)
        
        # Save outputs if session is active
        if SESSION_DATA.get("output_dir"):
            seg_dir = Path(SESSION_DATA["output_dir"]) / "segmentation"
            
        # Add Masks Layer (Required for assigning puncta to cells)
        if masks is not None:
            viewer.add_labels(masks, name=f"{layer.name}_masks", opacity=0.3, visible=False, scale=layer.scale)
            if SESSION_DATA.get("output_dir"):
                mask_path = seg_dir / f"{layer.name}_masks.tif"
                tifffile.imwrite(mask_path, masks)
                SESSION_DATA["processed_files"][f"{layer.name}_masks"] = str(mask_path)
            
        if centroids is not None:
            viewer.add_points(
                centroids,
                name=f"{layer.name}_centroids",
                size=5,
                face_color='orange',
                scale=layer.scale
            )
            if SESSION_DATA.get("output_dir"):
                cent_path = seg_dir / f"{layer.name}_centroids.npy"
                np.save(cent_path, centroids)
                SESSION_DATA["processed_files"][f"{layer.name}_centroids"] = str(cent_path)
    
    save_session()
    viewer.status = "Segmentation complete."

@magicgui(
    call_button="Calculate Shift (RANSAC)",
    r1_points={"label": "R1 Centroids"},
    r2_points={"label": "R2 Centroids"}
)
def registration_widget(
    r1_points: "napari.layers.Points",
    r2_points: "napari.layers.Points"
):
    """Calculates the XYZ shift between two point clouds."""
    viewer = napari.current_viewer()
    if r1_points is None or r2_points is None:
        viewer.status = "Please select both centroid layers."
        return

    p1 = r1_points.data # (N, 3) -> Z, Y, X
    p2 = r2_points.data # (M, 3) -> Z, Y, X
    
    viewer.status = "Running RANSAC..."
    shift = align_centroids_ransac(p1, p2)
    
    # Store shift in global variable for the next step
    global CALCULATED_SHIFT
    CALCULATED_SHIFT = shift
    SESSION_DATA["shift"] = shift.tolist()
    save_session()
    
    # Output results
    msg = f"Calculated Shift: Z={shift[0]:.2f}, Y={shift[1]:.2f}, X={shift[2]:.2f}"
    print(msg)
    viewer.status = msg
    
    # Show a message box (optional, but helpful)
    from qtpy.QtWidgets import QMessageBox
    msg_box = QMessageBox()
    msg_box.setText(f"Registration Complete.\n\n{msg}\n\nNext Step: Generate Global Canvas.")
    msg_box.exec_()

@magicgui(
    call_button="Generate Global Canvas",
    apply_warp={"label": "Apply Deformable Warping?"}
)
def canvas_widget(
    apply_warp: bool = True
):
    """Applies the calculated shift to all layers and creates a global canvas."""
    viewer = napari.current_viewer()
    global CALCULATED_SHIFT
    shift = CALCULATED_SHIFT
    
    # Retrieve output directory from session
    output_dir = None
    if SESSION_DATA.get("output_dir"):
        output_dir = Path(SESSION_DATA["output_dir"]) / "aligned"
        output_dir.mkdir(exist_ok=True, parents=True)
    
    if shift is None:
        viewer.status = "No shift calculated. Run Registration first."
        print("Error: No shift found in metadata.")
        return
        
    # SAFETY CHECK: Prevent OOM crashes from bad shifts
    # If the shift creates a volume > 10GB (approx), abort.
    # Assuming roughly 2k x 2k image * 2 bytes (uint16) = 8MB per slice.
    # 10GB = ~1200 slices.
    estimated_z_expansion = abs(shift[0])
    if estimated_z_expansion > 1000:
        msg = f"ABORTING: Calculated Z-shift ({shift[0]:.2f}) is dangerously large and would crash the application.\nPlease re-run registration or check your data."
        print(msg)
        viewer.status = "Error: Shift too large (OOM Protection)"
        return

    viewer.status = f"Generating Canvas with Shift: {shift}"
    
    # 1. Rigid Align All Channels
    # We store the results to process DAPI first for warping
    aligned_data = {} # channel: (r1_aligned, r2_aligned, r1_layer_ref, r2_layer_ref)
    
    r1_layers = [l for l in viewer.layers if "R1" in l.name and isinstance(l, napari.layers.Image)]
    r2_layers = [l for l in viewer.layers if "R2" in l.name and isinstance(l, napari.layers.Image)]

    for r1 in r1_layers:
        channel_name = r1.name.split("-")[-1].strip()
        r2 = next((l for l in r2_layers if channel_name in l.name), None)
        
        if r2:
            print(f"Rigid aligning {channel_name}...")
            aligned_r1, aligned_r2 = align_and_pad_images(r1.data, r2.data, shift)
            aligned_data[channel_name] = (aligned_r1, aligned_r2, r1, r2)

    # 2. Calculate Deformable Transform (on DAPI)
    transform = None
    if apply_warp:
        if "DAPI" in aligned_data:
            print("Calculating deformable registration on DAPI...")
            viewer.status = "Calculating AI Warp (this may take a moment)..."
            dapi_r1, dapi_r2, _, _ = aligned_data["DAPI"]
            transform = calculate_deformable_transform(dapi_r1, dapi_r2)
        else:
            print("Warning: No DAPI channel found. Skipping deformable registration.")

    # 3. Apply Transform (Parallelized)
    def warp_worker(item):
        channel_name, (r1_data, r2_data, r1_layer, r2_layer) = item
        final_r2 = r2_data
        r2_name_prefix = "Aligned"
        
        if transform:
            print(f"Applying warp to {channel_name}...")
            final_r2 = apply_deformable_transform(r2_data, transform, r1_data)
            r2_name_prefix = "Warped"
        return channel_name, r1_data, final_r2, r1_layer, r2_layer, r2_name_prefix

    results = []
    if transform:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(warp_worker, aligned_data.items()))
    else:
        results = [warp_worker(item) for item in aligned_data.items()]

    # 4. Add to Viewer
    for channel_name, r1_data, final_r2, r1_layer, r2_layer, r2_name_prefix in results:
            
        # Add to viewer
        viewer.add_image(
            r1_data, 
            name=f"Aligned R1 - {channel_name}", 
            colormap=r1_layer.colormap, 
            scale=r1_layer.scale, 
            blending='additive'
        )
        viewer.add_image(
            final_r2, 
            name=f"{r2_name_prefix} R2 - {channel_name}", 
            colormap=r2_layer.colormap, 
            scale=r2_layer.scale, 
            blending='additive'
        )
        
        # Save automatically
        if output_dir:
            out_name_r1 = output_dir / f"Aligned_R1_{channel_name}.tif"
            out_name_r2 = output_dir / f"{r2_name_prefix}_R2_{channel_name}.tif"
            tifffile.imwrite(out_name_r1, r1_data)
            tifffile.imwrite(out_name_r2, final_r2)
            SESSION_DATA["processed_files"][f"Aligned R1 - {channel_name}"] = str(out_name_r1)
            SESSION_DATA["processed_files"][f"{r2_name_prefix} R2 - {channel_name}"] = str(out_name_r2)
            print(f"Saved {out_name_r1}")

    save_session()

    viewer.status = "Global Canvas Generation Complete."

@magicgui(
    call_button="Detect Puncta",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Optional)"},
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01}
)
def puncta_widget(
    image_layer: "napari.layers.Image",
    nuclei_layer: "napari.layers.Labels",
    threshold: float = 0.05
):
    """Detects spots in the selected image layer."""
    viewer = napari.current_viewer()
    if image_layer is None:
        return
        
    viewer.status = f"Detecting spots in {image_layer.name}..."
    
    # Run detection
    coords = detect_spots_3d(image_layer.data, threshold_rel=threshold)
    
    layer_name = f"{image_layer.name}_puncta"
    
    # Add points to viewer
    pts_layer = viewer.add_points(
        coords,
        name=layer_name,
        size=3,
        face_color="yellow",
        scale=image_layer.scale
    )
    
    # Save puncta if session is active
    if SESSION_DATA.get("output_dir"):
        seg_dir = Path(SESSION_DATA["output_dir"]) / "segmentation"
        puncta_path = seg_dir / f"{layer_name}.npy"
        np.save(puncta_path, coords)
        SESSION_DATA["processed_files"][layer_name] = str(puncta_path)
        save_session()

    # Count per nucleus (if masks provided)
    msg = f"Found {len(coords)} spots."
    if nuclei_layer is not None:
        # Simple lookup: check mask value at spot coordinate
        # Note: coords are (Z, Y, X), mask is (Z, Y, X)
        # We need to be careful with float coords vs int indices
        # For peak_local_max, coords are integers.
        
        # Filter out spots not in a nucleus (optional logic)
        # For now, just print total count
        pass
        
    print(msg)
    viewer.status = msg

@magicgui(
    call_button="Load Session",
    session_file={"label": "Session File (.json)", "filter": "*.json"}
)
def load_session_widget(session_file: Path):
    """Restores a previous analysis session."""
    viewer = napari.current_viewer()
    if not session_file.exists():
        return
        
    viewer.layers.clear()
        
    with open(session_file, 'r') as f:
        data = json.load(f)
        
    # Restore Global State
    global SESSION_DATA, CALCULATED_SHIFT
    SESSION_DATA.update(data)
    
    if SESSION_DATA.get("shift"):
        CALCULATED_SHIFT = np.array(SESSION_DATA["shift"])
        print(f"Restored Shift: {CALCULATED_SHIFT}")

    # Load Raw Data
    if SESSION_DATA.get("r1_path") and SESSION_DATA.get("r2_path"):
        # Call file selector but prevent it from overwriting the session file
        file_selector_widget(
            round1_path=Path(SESSION_DATA["r1_path"]), 
            round2_path=Path(SESSION_DATA["r2_path"]),
            output_dir=Path(SESSION_DATA["output_dir"]),
            _save_session=False
        )

    # Determine scale from loaded raw data layers
    scale = (1, 1, 1)
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            scale = layer.scale
            break

    # Load Processed Files (Masks/Centroids/Puncta/Aligned)
    for name, path_str in SESSION_DATA.get("processed_files", {}).items():
        path = Path(path_str)
        if path.exists():
            if path.suffix == '.npy':
                data = np.load(path)
                if "centroids" in name.lower():
                    viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale)
                else: # Assume it's puncta
                    viewer.add_points(data, name=name, size=3, face_color='yellow', scale=scale)
            elif path.suffix in ['.tif', '.tiff']:
                data = tifffile.imread(path)
                if "masks" in name.lower():
                    viewer.add_labels(data, name=name, opacity=0.3, visible=False, scale=scale)
                else:
                    # Restore colormap based on channel name
                    c_map = 'gray'
                    for ch, color in CHANNEL_COLORS.items():
                        if ch.upper() in name.upper():
                            c_map = color
                            break
                    viewer.add_image(data, name=name, blending='additive', scale=scale, colormap=c_map)
            print(f"Restored layer: {name}")
            
    viewer.status = "Session Restored."

def create_welcome_widget(viewer):
    """Creates a welcome widget with instructions and a help button."""
    container = widgets.Container(labels=False)
    
    # HTML-subset styling is supported in Qt labels
    container.append(widgets.Label(value="<h1 style='color: #00FFFF;'>zFISHer</h1>"))
    container.append(widgets.Label(value="<em>Multiplexed Sequential FISH Analysis</em>"))
    
    container.append(widgets.Label(value="<h3>Workflow:</h3>"))
    container.append(widgets.Label(value="1. <b>Load Data</b> (.nd2 files)"))
    container.append(widgets.Label(value="2. <b>Segment Nuclei</b> (DAPI)"))
    container.append(widgets.Label(value="3. <b>Register Rounds</b> (RANSAC)"))
    container.append(widgets.Label(value="4. <b>Generate Canvas</b> (Warp)"))
    container.append(widgets.Label(value="5. <b>Detect Puncta</b> (Spots)"))
    
    # Buttons Row
    btn_row = widgets.Container(layout="horizontal", labels=False)
    help_btn = widgets.PushButton(text="Open README / Help")
    reset_btn = widgets.PushButton(text="Reset")
    btn_row.extend([help_btn, reset_btn])
    container.append(btn_row)
    
    @help_btn.changed.connect
    def open_help():
        # Look for README in project root
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        if readme_path.exists():
            webbrowser.open(readme_path.as_uri())
        else:
            print(f"README not found at {readme_path}")
            
    @reset_btn.changed.connect
    def reset_viewer():
        viewer.layers.clear()
        global CALCULATED_SHIFT
        CALCULATED_SHIFT = None
        viewer.status = "Viewer cleared."
            
    return container
    
def launch_zfisher():
    # Set App Icon
    app = QApplication.instance() or QApplication([])
    icon_path = Path(__file__).parent.parent.parent / "icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    viewer = napari.Viewer(title="zFISHer - 3D Colocalization", ndisplay=2)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    
    # Create local references
    widgets_to_add = [
        (create_welcome_widget(viewer), "Home"),
        (load_session_widget, "Resume Session"),
        (file_selector_widget, "1. File Selection"),
        (dapi_segmentation_widget, "2. DAPI Mapping"),
        (registration_widget, "3. Registration"),
        (canvas_widget, "4. Global Canvas"),
        (puncta_widget, "5. Puncta Detection")
    ]

    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    
    for widget, name in widgets_to_add:
        # This is the secret sauce: 
        # Manually trigger a refresh so the 'choices' list isn't ()
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        
        toolbox.addItem(widget.native, name)

    viewer.window.add_dock_widget(toolbox, area="right", name="zFISHer Workflow")

    def on_layer_inserted(event):
        layer = event.value
        
        def update_widgets():
            # Refresh the choices again so the layer is 'valid' before we select it
            dapi_segmentation_widget.reset_choices()
            registration_widget.reset_choices()
            puncta_widget.reset_choices()

            if isinstance(layer, napari.layers.Image):
                if "DAPI" in layer.name.upper():
                    if "R1" in layer.name.upper():
                        dapi_segmentation_widget.r1_layer.value = layer
                    elif "R2" in layer.name.upper():
                        dapi_segmentation_widget.r2_layer.value = layer
            
            if isinstance(layer, napari.layers.Points):
                if "R1" in layer.name.upper():
                    registration_widget.r1_points.value = layer
                elif "R2" in layer.name.upper():
                    registration_widget.r2_points.value = layer
        
        QTimer.singleShot(100, update_widgets) # Increased delay to 100ms for safety

    viewer.layers.events.inserted.connect(on_layer_inserted)
    viewer.layers.events.removed.connect(lambda e: [w.reset_choices() for w in [dapi_segmentation_widget, registration_widget, puncta_widget]])
    napari.run()