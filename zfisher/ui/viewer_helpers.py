import napari
import numpy as np
import tifffile
from pathlib import Path
from packaging.version import parse as parse_version

from .constants import CHANNEL_COLORS
from . import style

# Global to track the zoom listener, but now it's scoped to this module
_current_scale_updater = None

def add_image_session_to_viewer(viewer: napari.Viewer, image_session, prefix: str):
    """
    Adds image data from a FISHSession or TiffSession object to the napari viewer.

    This function handles:
    - Adding the image layers with correct names and scale.
    - Applying default colormaps based on channel names.
    - Setting up the text overlay for voxel size and FOV.
    """
    global _current_scale_updater

    # Data from io.py is (Z, C, Y, X), napari expects (C, Z, Y, X) for channel_axis=0
    data_swapped = np.moveaxis(image_session.data, 1, 0)
    
    print(f"Loaded {prefix}: {data_swapped.shape[0]} channels, {data_swapped.shape[1]} Z-slices. Full shape: {data_swapped.shape}")
    
    new_layers = viewer.add_image(
        data_swapped,
        name=[f"{prefix} - {ch}" for ch in image_session.channels],
        channel_axis=0,
        scale=image_session.voxels,
        blending="additive"
    )

    # Apply colors
    for layer in new_layers:
        for ch_name, color in CHANNEL_COLORS.items():
            if ch_name.upper() in layer.name.upper():
                layer.colormap = color
        
        if "DAPI" not in layer.name.upper():
            layer.visible = False

    # Update Text Overlay with Pixel Info
    dz, dy, dx = image_session.voxels
    
    def update_scale_text(event=None):
        try:
            width = viewer.canvas.size[1]
            zoom = viewer.camera.zoom
            if zoom > 0:
                fov_px = width / zoom
                fov_um = fov_px * dx
                px_per_um = 1.0 / dx if dx > 0 else 0
                viewer.text_overlay.text = (
                    f"Voxel Size: {dx:.4f} x {dy:.4f} x {dz:.4f} um\n"
                    f"Scale: 1 um = {px_per_um:.2f} px\n"
                    f"FOV Width: {fov_um:.1f} um ({int(fov_px)} px)"
                )
        except Exception:
            pass

    # Prevent duplicate listeners
    if _current_scale_updater is not None:
        try: viewer.camera.events.zoom.disconnect(_current_scale_updater)
        except (TypeError, RuntimeError): pass
    _current_scale_updater = update_scale_text
    viewer.camera.events.zoom.connect(update_scale_text)
    
    viewer.text_overlay.visible = True
    viewer.text_overlay.color = style.COLORS['text']
    viewer.text_overlay.font_size = style.TEXT_OVERLAY_FONT_SIZE
    viewer.text_overlay.position = "top_right"
    
    update_scale_text()

def _load_points_layer(viewer, name, path, scale, file_info):
    data = np.load(path, allow_pickle=True)
    subtype = file_info.get('subtype')

    if subtype == 'structured_ids':
        coords = np.vstack(data['coord'])
        properties = {p_name: data[p_name] for p_name in data.dtype.names if p_name != 'coord'}
        text_params = {
            'string': '{label}', 'size': 10, 'color': '#40b5d8',
            'translation': np.array([0, -5, 0])
        }
        viewer.add_points(
            coords, name=name, size=0, scale=scale, properties=properties,
            text=text_params, blending='translucent_no_depth'
        )
    elif subtype == 'centroids':
        viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale)
    elif subtype == 'puncta':
        properties = {'id': np.arange(len(data)) + 1}
        text_params = {
            'string': '{id}', 'size': 8, 'color': 'white',
            'translation': np.array([0, 5, 5]),
        }
        viewer.add_points(
            data, name=name, size=3, face_color='yellow', scale=scale,
            properties=properties, text=text_params
        )
    else: # Generic points
        viewer.add_points(data, name=name, scale=scale)

def _load_labels_layer(viewer, name, path, scale, file_info):
    data = tifffile.imread(path)
    viewer.add_labels(data, name=name, opacity=0.3, visible=False, scale=scale)

def _load_image_layer(viewer, name, path, scale, file_info):
    data = tifffile.imread(path)
    c_map = 'gray'
    for ch, color in CHANNEL_COLORS.items():
        if ch.upper() in name.upper():
            c_map = color
            break
    viewer.add_image(data, name=name, blending='additive', scale=scale, colormap=c_map)

def _load_vectors_layer(viewer, name, path, scale, file_info):
    data = np.load(path, allow_pickle=True)
    vector_params = {
        'data': data, 'name': name, 'opacity': 1.0, 'edge_width': 1,
        'length': 1, 'edge_color': 'white',
    }
    if parse_version(napari.__version__) >= parse_version("0.7.0"):
        vector_params['head_width'] = 4
        vector_params['head_length'] = 6
    viewer.add_vectors(**vector_params)

def restore_processed_layers(viewer: napari.Viewer, processed_files: dict, default_scale: tuple, progress_callback=None):
    """
    Loads processed files (masks, points, etc.) from a session into the viewer.
    """
    num_files = len(processed_files)
    if num_files == 0:
        return

    for i, (name, file_info) in enumerate(processed_files.items()):
        if progress_callback:
            progress = int(((i + 1) / num_files) * 100)
            progress_callback(progress, f"Loading: {name}")

        # Handle old format for backward compatibility, or just ignore non-dict values
        if not isinstance(file_info, dict):
            print(f"Skipping layer '{name}' with old/invalid format in session file.")
            continue

        path_str = file_info.get('path')
        layer_type = file_info.get('type')

        if layer_type == 'report': # Don't load reports as layers
            continue

        if not path_str or not layer_type:
            print(f"Skipping layer '{name}' due to missing path or type information.")
            continue

        path = Path(path_str)
        if not path.exists():
            print(f"Warning: File not found for layer '{name}': {path_str}")
            continue

        try:
            if layer_type == 'points':
                _load_points_layer(viewer, name, path, default_scale, file_info)
            elif layer_type == 'labels':
                _load_labels_layer(viewer, name, path, default_scale, file_info)
            elif layer_type == 'image':
                _load_image_layer(viewer, name, path, default_scale, file_info)
            elif layer_type == 'vectors':
                _load_vectors_layer(viewer, name, path, default_scale, file_info)
            else:
                print(f"Warning: Unknown layer type '{layer_type}' for layer '{name}'.")

            print(f"Restored layer: {name}")
        except Exception as e:
            print(f"Error restoring layer '{name}': {e}")