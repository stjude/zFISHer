import napari
from pathlib import Path
from magicgui import magicgui
import numpy as np
import tifffile

import zfisher.core.session as session
from .. import popups
from ..constants import CHANNEL_COLORS
from ._shared import load_raw_data_into_viewer

@magicgui(
    call_button="Load Session",
    session_file={"label": "Session File (.json)", "filter": "*.json"}
)
def load_session_widget(session_file: Path):
    """Restores a previous analysis session."""
    viewer = napari.current_viewer()
    if not session_file.exists() or session_file.is_dir():
        if session_file.is_dir():
            viewer.status = "Error: Please select a session file, not a directory."
        return

    dialog = popups.ProgressDialog(viewer.window._qt_window, "Loading Session...")
    try:
        viewer.layers.clear()
        
        # Restore Global State
        dialog.update_progress(10, "Loading session file...")
        session.load_session_file(session_file)
        
        shift = session.get_data("shift")
        if shift:
            print(f"Restored Shift: {shift}")

        # Load Raw Data
        # This section is allocated 60% of the progress bar (10% -> 70%)
        r1_path = session.get_data("r1_path")
        r2_path = session.get_data("r2_path")
        if r1_path and r2_path:
            # Define a callback that scales the progress from the loader (0-100)
            # to the 10-70 range of our main progress bar.
            def progress_callback(p, text):
                # p is 0-100, scale it to a 60-point range (0-60) and add offset of 10
                scaled_progress = 10 + int(p * 0.6) 
                dialog.update_progress(scaled_progress, text)

            load_raw_data_into_viewer(
                viewer, 
                r1_path, 
                r2_path,
                progress_callback=progress_callback
            )

        # Determine scale from loaded raw data layers
        scale = (1, 1, 1)
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Image):
                scale = layer.scale
                break

        # Load Processed Files (Masks/Centroids/Puncta/Aligned)
        # This section is allocated 25% of the progress bar (70% -> 95%)
        processed_files = session.get_data("processed_files").items()
        num_files = len(processed_files)
        if num_files == 0:
            num_files = 1 # Avoid division by zero

        for i, (name, path_str) in enumerate(processed_files):
            # Scale progress to the 70-95 range
            progress = 70 + int(25 * (i + 1) / num_files)
            dialog.update_progress(progress, f"Loading: {name}")

            path = Path(path_str)
            if path.exists():
                if path.suffix == '.npy':
                    data = np.load(path, allow_pickle=True)

                    # Handle structured arrays (for points with properties)
                    if data.dtype.names:
                        coords = np.vstack(data['coord'])
                        properties = {name: data[name] for name in data.dtype.names if name != 'coord'}
                        
                        # Special case for consensus IDs layer
                        if "consensus_nuclei_masks_ids" in name.lower():
                            text_params = {
                                'string': '{label}',
                                'size': 10,
                                'color': 'cyan',
                                'translation': np.array([0, -5, 0])
                            }
                            viewer.add_points(
                                coords,
                                name=name,
                                size=0,
                                scale=scale,
                                properties=properties,
                                text=text_params,
                                blending='translucent_no_depth'
                            )
                        else: # Other structured arrays if they exist in the future
                             viewer.add_points(
                                coords,
                                name=name,
                                scale=scale,
                                properties=properties
                            )

                    # Handle simple coordinate arrays
                    else:
                        if name == "Arrows":
                            viewer.add_vectors(
                                data,
                                name=name,
                                opacity=1.0,
                                edge_width=2,
                                length=10,
                                edge_color='cyan'
                            )
                        elif "centroids" in name.lower():
                            viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale)
                        else: # Assume it's puncta
                            properties = {'id': np.arange(len(data)) + 1}
                            text_params = {
                                'string': '{id}',
                                'size': 8,
                                'color': 'white',
                                'translation': np.array([0, 5, 5]),
                            }
                            viewer.add_points(
                                data, 
                                name=name, 
                                size=3, 
                                face_color='yellow', 
                                scale=scale,
                                properties=properties,
                                text=text_params
                            )
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
        
        dialog.update_progress(95, "Finalizing...")
        viewer.status = "Session Restored."

        # Reset scale bar position to bottom right
        if hasattr(viewer.window, 'custom_scale_bar'):
            viewer.window.custom_scale_bar.move_to_bottom_right()
        
        dialog.update_progress(100, "Done.")

    finally:
        dialog.close()
