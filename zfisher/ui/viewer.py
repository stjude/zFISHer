import napari
import pandas as pd
from scipy.spatial import cKDTree
from magicgui import magicgui, widgets
import webbrowser
from pathlib import Path
from zfisher.core.io import load_nd2
from zfisher.core.registration import segment_nuclei_classical, align_centroids_ransac
from zfisher.core.segmentation import detect_spots_3d
from zfisher.core.report import calculate_distances, export_report
import zfisher.core.session as session
from zfisher.core.pipeline import generate_global_canvas
import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon
from qtpy.QtCore import QTimer
import os
import concurrent.futures
from qtpy.QtWidgets import QToolBox, QVBoxLayout, QWidget, QMessageBox

# Define your paths as constants at the top for easy editing later
DEFAULT_R1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
DEFAULT_R2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")

# Helper to map metadata names to colors
CHANNEL_COLORS = {
    "DAPI": "blue",
    "FITC": "green",
    "CY3": "yellow",
    "CY5": "red",
    "TXRED": "magenta"
}

def attach_puncta_listener(layer, name):
    """Attaches a listener to a points layer to auto-save changes to session."""
    def sync_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir:
            seg_dir = Path(out_dir) / "segmentation"
            seg_dir.mkdir(exist_ok=True, parents=True)
            puncta_path = seg_dir / f"{name}.npy"
            np.save(puncta_path, layer.data)
            session.set_processed_file(name, str(puncta_path))
            session.save_session()
            
    layer.events.data.connect(sync_data)

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
        session.clear_session()
        session.update_data("output_dir", str(output_dir))
        session.update_data("r1_path", str(round1_path))
        session.update_data("r2_path", str(round2_path))
        session.save_session()
    
    for path, prefix in [(round1_path, "R1"), (round2_path, "R2")]:
        if not path.exists():
            print(f"Error: {path} not found.")
            continue
            
        nd2_session = load_nd2(str(path))
        
        # YOUR DATA SHAPE: (71, 3, 2044, 2048) -> (Z, C, Y, X)
        # NAPARI EXPECTS CHANNELS AT INDEX 1 IF WE WANT TO SPLIT THEM
        # We move axis 1 (Channels) to the front so it becomes (C, Z, Y, X)
        data_swapped = np.moveaxis(nd2_session.data, 1, 0)
        
        # Print dimensions for the user
        print(f"Loaded {prefix}: {data_swapped.shape[0]} channels, {data_swapped.shape[1]} Z-slices. Full shape: {data_swapped.shape}")
        
        # Now shape is (3, 71, 2044, 2048)
        # Axis 0 = 3 channels
        # Axis 1 = 71 Z-slices
        
        new_layers = viewer.add_image(
            data_swapped,
            name=[f"{prefix} - {ch}" for ch in nd2_session.channels],
            channel_axis=0,        # Now correctly sees 3 channels
            scale=nd2_session.voxels,   # Matches the (71, 2044, 2048) ZYX stack
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
        
        out_dir = session.get_data("output_dir")
        # Save outputs if session is active
        if out_dir:
            seg_dir = Path(out_dir) / "segmentation"
            
        # Add Masks Layer (Required for assigning puncta to cells)
        if masks is not None:
            viewer.add_labels(masks, name=f"{layer.name}_masks", opacity=0.3, visible=False, scale=layer.scale)
            if out_dir:
                mask_path = seg_dir / f"{layer.name}_masks.tif"
                tifffile.imwrite(mask_path, masks)
                session.set_processed_file(f"{layer.name}_masks", mask_path)
            
        if centroids is not None:
            viewer.add_points(
                centroids,
                name=f"{layer.name}_centroids",
                size=5,
                face_color='orange',
                scale=layer.scale
            )
            if out_dir:
                cent_path = seg_dir / f"{layer.name}_centroids.npy"
                np.save(cent_path, centroids)
                session.set_processed_file(f"{layer.name}_centroids", cent_path)
    
    session.save_session()
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
    
    session.update_data("shift", shift.tolist())
    session.save_session()
    
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
    shift_list = session.get_data("shift")
    shift = np.array(shift_list) if shift_list else None
    
    # Retrieve output directory from session
    output_dir = None
    if session.get_data("output_dir"):
        output_dir = Path(session.get_data("output_dir")) / "aligned"
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
    
    # Extract layer data to pass to pipeline
    r1_layers_data = []
    r2_layers_data = []
    
    for l in viewer.layers:
        if isinstance(l, napari.layers.Image):
            layer_info = {'name': l.name, 'data': l.data, 'colormap': l.colormap.name, 'scale': l.scale}
            if "R1" in l.name:
                r1_layers_data.append(layer_info)
            elif "R2" in l.name:
                r2_layers_data.append(layer_info)

    # Run Pipeline
    results = generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp)

    # Add results to viewer
    for res in results:
        r1 = res['r1']
        r2 = res['r2']
        viewer.add_image(
            r1['data'], 
            name=r1['name'], 
            colormap=r1['colormap'], 
            scale=r1['scale'], 
            blending='additive'
        )
        viewer.add_image(
            r2['data'], 
            name=r2['name'], 
            colormap=r2['colormap'], 
            scale=r2['scale'], 
            blending='additive'
        )

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
    
    if layer_name in viewer.layers:
        pts_layer = viewer.layers[layer_name]
        if len(coords) > 0:
            if len(pts_layer.data) > 0:
                combined = np.vstack((pts_layer.data, coords))
                pts_layer.data = np.unique(combined, axis=0)
            else:
                pts_layer.data = coords
        coords = pts_layer.data
    else:
        # Add points to viewer
        pts_layer = viewer.add_points(
            coords,
            name=layer_name,
            size=3,
            face_color="yellow",
            scale=image_layer.scale
        )
        # Attach auto-save listener and trigger initial save
        attach_puncta_listener(pts_layer, layer_name)
        pts_layer.events.data(value=pts_layer.data) # Trigger initial save

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

# Add editing tools to the Puncta Widget
edit_btn = widgets.PushButton(text="Enable Edit Mode (Select)")
clear_btn = widgets.PushButton(text="Clear All Puncta")
puncta_widget.append(widgets.Label(value="<b>Editing Tools:</b>"))
puncta_widget.append(edit_btn)
puncta_widget.append(clear_btn)

@edit_btn.clicked.connect
def _on_edit_puncta():
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            viewer.layers.selection.active = viewer.layers[p_name]
            viewer.layers[p_name].mode = 'select'
            viewer.status = f"Editing {p_name}. Select points and press Backspace/Delete to remove."
        else:
            viewer.status = f"Layer {p_name} not found. Run detection first."

@clear_btn.clicked.connect
def _on_clear_puncta():
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            viewer.layers[p_name].data = np.empty((0, 3))
            viewer.status = f"Cleared all points in {p_name}."

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
        
    # Restore Global State
    session.load_session_file(session_file)
    
    shift = session.get_data("shift")
    if shift:
        print(f"Restored Shift: {shift}")

    # Load Raw Data
    r1_path = session.get_data("r1_path")
    r2_path = session.get_data("r2_path")
    if r1_path and r2_path:
        # Call file selector but prevent it from overwriting the session file
        file_selector_widget(
            round1_path=Path(r1_path), 
            round2_path=Path(r2_path),
            output_dir=Path(session.get_data("output_dir")),
            _save_session=False
        )

    # Determine scale from loaded raw data layers
    scale = (1, 1, 1)
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Image):
            scale = layer.scale
            break

    # Load Processed Files (Masks/Centroids/Puncta/Aligned)
    for name, path_str in session.get_data("processed_files").items():
        path = Path(path_str)
        if path.exists():
            if path.suffix == '.npy':
                data = np.load(path)
                if "centroids" in name.lower():
                    viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale)
                else: # Assume it's puncta
                    l = viewer.add_points(data, name=name, size=3, face_color='yellow', scale=scale)
                    attach_puncta_listener(l, name)
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

@magicgui(
    call_button="Calculate & Export Distances",
    output_filename={"label": "Filename (.xlsx)", "value": "puncta_distances.xlsx"}
)
def distance_widget(output_filename: str = "puncta_distances.xlsx"):
    """Calculates nearest neighbor distances between all puncta layers."""
    viewer = napari.current_viewer()
    
    # Find all points layers
    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    
    if len(points_layers) < 2:
        viewer.status = "Need at least 2 points layers."
        print("Error: Not enough points layers found.")
        return

    viewer.status = "Calculating distances..."
    
    # Prepare data for core function
    points_data = []
    for l in points_layers:
        points_data.append({
            'name': l.name,
            'data': l.data,
            'scale': l.scale
        })
        
    df = calculate_distances(points_data)
                
    if df.empty:
        viewer.status = "No distances calculated."
        return
        
    # Export
    try:
        # Determine output path
        if session.get_data("output_dir"):
            save_path = Path(session.get_data("output_dir")) / output_filename
        else:
            save_path = Path.home() / output_filename
            
        final_path = export_report(
            df, 
            save_path, 
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=session.get_data("output_dir")
        )
        
        print(f"Saved distances to {final_path}")
        viewer.status = f"Exported: {final_path.name}"
        
        # Track file
        if session.get_data("output_dir"):
             session.set_processed_file("Distance_Report", final_path)
             session.save_session()
        
        # Show success popup
        msg = QMessageBox()
        msg.setWindowTitle("Export Complete")
        msg.setText(f"Analysis exported successfully.\n\nFile: {final_path.name}\nPath: {final_path}")
        msg.exec_()
             
    except Exception as e:
        print(f"Export failed: {e}")
        viewer.status = "Export failed (check console)."

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
        session.clear_session()
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
        (puncta_widget, "5. Puncta Detection"),
        (distance_widget, "6. Analysis Export")
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
            distance_widget.reset_choices()

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
    viewer.layers.events.removed.connect(lambda e: [w.reset_choices() for w in [dapi_segmentation_widget, registration_widget, puncta_widget, distance_widget]])
    napari.run()