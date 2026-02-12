import napari
from pathlib import Path
from magicgui import magicgui
import numpy as np
import tifffile

import zfisher.core.session as session
from ..constants import CHANNEL_COLORS
from ._shared import load_raw_data_into_viewer

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
        # Call the helper to load data without creating a new session
        load_raw_data_into_viewer(viewer, r1_path, r2_path)

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
            
    viewer.status = "Session Restored."
