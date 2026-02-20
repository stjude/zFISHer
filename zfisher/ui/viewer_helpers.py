import napari
import numpy as np
import tifffile
from pathlib import Path
import pandas as pd
from packaging.version import parse as parse_version
from ..core import session, segmentation

from .. import constants
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
        for ch_name, color in constants.CHANNEL_COLORS.items():
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
    if path.suffix.lower() == '.csv':
        try:
            # Puncta are saved as CSV with header: Z,Y,X,...
            df = pd.read_csv(path)
            data = df[['Z', 'Y', 'X']].to_numpy()
        except (pd.errors.EmptyDataError, KeyError, FileNotFoundError):
            data = np.empty((0, 3)) # Return empty array if file is empty or has wrong columns
    else: # Assume .npy
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
    for ch, color in constants.CHANNEL_COLORS.items():
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

def add_or_update_puncta_layer(viewer: napari.Viewer, source_layer: napari.layers.Image, coords: np.ndarray):
    """
    Adds a new puncta layer to the viewer or updates an existing one.

    This function handles creating the layer name, merging coordinates if the
    layer already exists, and performing the initial save to the session.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    source_layer : napari.layers.Image
        The image layer from which the puncta were detected.
    coords : np.ndarray
        The (N, 3) array of detected puncta coordinates.
    """
    if coords is None or len(coords) == 0:
        return

    layer_name = f"{source_layer.name}{constants.PUNCTA_SUFFIX}"

    if layer_name in viewer.layers:
        pts_layer = viewer.layers[layer_name]
        pts_layer.data = segmentation.merge_puncta(pts_layer.data, coords)
        pts_layer.properties = {'id': np.arange(len(pts_layer.data)) + 1}
        pts_layer.text = {'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, 5, 5])}
    else:
        properties = {'id': np.arange(len(coords)) + 1}
        text_params = {'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, 5, 5])}

        viewer.add_points(
            coords,
            name=layer_name,
            size=3,
            face_color="yellow",
            scale=source_layer.scale,
            properties=properties,
            text=text_params
        )
        # Handle the initial save explicitly to avoid race conditions with the event system.
        out_dir = session.get_data("output_dir")
        if out_dir:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            puncta_path = seg_dir / f"{layer_name}.npy"
            np.save(puncta_path, coords)
            session.set_processed_file(layer_name, str(puncta_path), layer_type='points', metadata={'subtype': 'puncta'})

def add_segmentation_results_to_viewer(viewer: napari.Viewer, source_layer: napari.layers.Image, masks: np.ndarray, centroids: np.ndarray):
    """
    Adds nuclei segmentation results (masks and centroids) to the viewer.

    Handles layer creation, naming, styling, and saving to the session.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    source_layer : napari.layers.Image
        The DAPI image layer that was segmented.
    masks : np.ndarray
        The (Z, Y, X) labeled integer mask of the segmented nuclei.
    centroids : np.ndarray
        The (N, 3) array of centroid coordinates.
    """
    out_dir = session.get_data("output_dir")
    seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR if out_dir else None

    if masks is not None:
        mask_layer_name = f"{source_layer.name}{constants.MASKS_SUFFIX}"
        viewer.add_labels(masks, name=mask_layer_name, opacity=0.3, visible=False, scale=source_layer.scale)
        if seg_dir:
            mask_path = seg_dir / f"{mask_layer_name}.tif"
            tifffile.imwrite(mask_path, masks)
            session.set_processed_file(mask_layer_name, str(mask_path), layer_type='labels', metadata={'subtype': 'mask'})

    if centroids is not None:
        centroid_layer_name = f"{source_layer.name}{constants.CENTROIDS_SUFFIX}"
        ids = np.arange(len(centroids)) + 1
        viewer.add_points(
            centroids,
            name=centroid_layer_name,
            size=5,
            face_color='orange',
            scale=source_layer.scale,
            properties={'id': ids},
            text={'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, -5, 0])},
            blending='translucent_no_depth'
        )
        if seg_dir:
            cent_path = seg_dir / f"{centroid_layer_name}.npy"
            np.save(cent_path, centroids)
            session.set_processed_file(centroid_layer_name, str(cent_path), layer_type='points', metadata={'subtype': 'centroids'})

def add_consensus_nuclei_to_viewer(viewer: napari.Viewer, r1_mask_layer: napari.layers.Labels, merged_mask: np.ndarray, pts1: list):
    """
    Adds consensus nuclei results (merged mask and ID points) to the viewer.

    Handles layer creation, naming, styling, and saving to the session.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    r1_mask_layer : napari.layers.Labels
        The reference mask layer, used for scale information.
    merged_mask : np.ndarray
        The consensus mask data to be added as a new layer.
    pts1 : list
        A list of dictionaries [{'coord':..., 'label':...}] for the ID points.
    """
    output_dir = session.get_data("output_dir")
    layer_name = constants.CONSENSUS_MASKS_NAME

    # Add merged mask layer
    viewer.add_labels(
        merged_mask,
        name=layer_name,
        scale=r1_mask_layer.scale,
        opacity=0.5
    )

    # Save mask to disk and session
    if output_dir:
        try:
            seg_dir = Path(output_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_save_path = seg_dir / f"{layer_name}.tif"
            tifffile.imwrite(mask_save_path, merged_mask)
            session.set_processed_file(layer_name, str(mask_save_path), layer_type='labels', metadata={'subtype': 'consensus_mask'})
            print(f"Saved consensus mask to {mask_save_path}")
        except Exception as e:
            print(f"Failed to save consensus mask: {e}")

    # Add ID points layer
    if pts1:
        ids_layer_name = f"{layer_name}{constants.CONSENSUS_IDS_SUFFIX}"
        coords = np.array([p['coord'] for p in pts1])
        labels = np.array([p['label'] for p in pts1])

        viewer.add_points(
            coords, name=ids_layer_name, size=0, scale=r1_mask_layer.scale,
            properties={'label': labels},
            text={'string': '{label}', 'size': 10, 'color': '#40b5d8', 'translation': np.array([0, -5, 0])},
            blending='translucent_no_depth'
        )

        # Save points to disk and session
        if output_dir:
            try:
                seg_dir = Path(output_dir) / constants.SEGMENTATION_DIR
                ids_save_path = seg_dir / f"{ids_layer_name}.npy"
                dtype = [('coord', 'f4', 3), ('label', 'i4')]
                structured_pts = np.array([(p['coord'], p['label']) for p in pts1], dtype=dtype)
                np.save(ids_save_path, structured_pts)
                session.set_processed_file(ids_layer_name, str(ids_save_path), layer_type='points', metadata={'subtype': 'structured_ids'})
                print(f"Saved consensus IDs to {ids_save_path}")
            except Exception as e:
                print(f"Failed to save consensus IDs: {e}")

def add_or_update_label_ids(viewer: napari.Viewer, labels_layer: napari.layers.Labels):
    """
    Calculates centroids for a labels layer and displays them as text.

    This creates or updates a points layer named after the labels layer
    (e.g., 'MyMask_IDs') to show the ID number at the centroid of each label.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    labels_layer : napari.layers.Labels
        The labels layer to analyze.
    """
    if not labels_layer:
        return

    pts_data = segmentation.get_mask_centroids(labels_layer.data)
    
    name = f"{labels_layer.name}_IDs"
    coords = np.array([p['coord'] for p in pts_data]) if pts_data else np.empty((0, labels_layer.ndim))
    labels = np.array([p['label'] for p in pts_data]) if pts_data else np.empty(0)
    
    if name in viewer.layers:
        viewer.layers[name].data = coords
        viewer.layers[name].properties = {'label': labels}
    elif len(coords) > 0:
        viewer.add_points(
            coords, name=name, size=0, scale=labels_layer.scale,
            properties={'label': labels},
            text={'string': '{label}', 'size': 10, 'color': '#40b5d8', 'translation': np.array([0, -5, 0])},
            blending='translucent_no_depth'
        )
    viewer.status = f"Refreshed IDs for {labels_layer.name}"