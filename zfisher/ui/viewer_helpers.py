import logging
import napari
import numpy as np
import tifffile
from pathlib import Path
import pandas as pd
from packaging.version import parse as parse_version
from ..core import session, segmentation

from .. import constants
from . import style

logger = logging.getLogger(__name__)

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
    
    logger.info("Loaded %s: %d channels, %d Z-slices. Full shape: %s", prefix, data_swapped.shape[0], data_swapped.shape[1], data_swapped.shape)
    
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
        
        nuc_ch = session.get_nuclear_channel().upper()
        if nuc_ch not in layer.name.upper():
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
                viewer.text_overlay.position = "top_right"
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

def _load_points_layer(viewer, name, path, scale, file_info, translate):
    subtype = file_info.get('subtype')

    logger.debug("Loading points layer '%s' (subtype=%s, path=%s, scale=%s, translate=%s)", name, subtype, path, scale, translate)

    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path)
            coord_cols = ['Z', 'Y', 'X']
            if not all(col in df.columns for col in coord_cols):
                logger.warning("CSV for layer '%s' is missing coordinate columns.", name)
                return
            
            data = df[coord_cols].to_numpy()
            features = df.drop(columns=coord_cols, errors='ignore')
            
            # Ensure puncta_id exists for text display
            if 'puncta_id' not in features.columns:
                features['puncta_id'] = np.arange(len(data))
            
            text_params = {
                'string': '{puncta_id:.0f}', 'size': 8, 'color': 'white',
                'translation': np.array([0, 5, 5]),
            }
            viewer.add_points(
                data, name=name, size=3, face_color='yellow', scale=scale,
                features=features, text=text_params, translate=translate, visible=False
            )
        else:  # Handle .npy files and other binary formats
            data = np.load(path, allow_pickle=True)
            
            if subtype == 'structured_ids':
                coords = np.vstack(data['coord'])
                properties = {p_name: data[p_name] for p_name in data.dtype.names if p_name != 'coord'}
                text_params = {
                    'string': '{label}', 'size': 10, 'color': '#40b5d8',
                    'translation': np.array([0, -5, 0])
                }
                # Inherit visibility from parent mask and position above it
                parent_mask_name = name[:-4] if name.endswith("_IDs") else None
                parent_vis = False
                if parent_mask_name and parent_mask_name in viewer.layers:
                    parent_vis = viewer.layers[parent_mask_name].visible
                layer = viewer.add_points(
                    coords, name=name, size=9999, face_color='transparent',
                    border_color='transparent', border_width=0, scale=scale, properties=properties,
                    text=text_params, blending='translucent_no_depth', translate=translate, visible=parent_vis
                )
                layer.out_of_slice_display = True
                try:
                    layer.text.blending = 'translucent_no_depth'
                except Exception:
                    pass
                if parent_mask_name and parent_mask_name in viewer.layers:
                    parent = viewer.layers[parent_mask_name]
                    mask_idx = list(viewer.layers).index(parent)
                    viewer.layers.move(list(viewer.layers).index(layer), mask_idx + 1)
                    _attach_ids_visibility_sync(parent, layer)
            
            elif subtype == 'centroids':
                viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale, translate=translate, visible=False)

            else:  # Default for .npy, including old 'puncta' subtype
                features = pd.DataFrame({'puncta_id': np.arange(len(data))})
                text_params = {
                    'string': '{puncta_id:.0f}', 'size': 8, 'color': 'white',
                    'translation': np.array([0, 5, 5]),
                }
                viewer.add_points(
                    data, name=name, size=3, face_color='yellow', scale=scale,
                    features=features, text=text_params, translate=translate, visible=False
                )
    except Exception as e:
        # The calling function `restore_processed_layers` will print this
        raise e

def _load_labels_layer(viewer, name, path, scale, file_info, translate):
    data = tifffile.imread(path)
    layer = viewer.add_labels(data, name=name, opacity=0.3, visible=False, scale=scale, translate=translate)
    # Use iso_categorical for better 3D rendering of masks alongside points
    layer.rendering = 'iso_categorical'

def _load_image_layer(viewer, name, path, scale, file_info, translate):
    data = tifffile.imread(path)
    c_map = 'gray'
    for ch, color in constants.CHANNEL_COLORS.items():
        if ch.upper() in name.upper():
            c_map = color
            break
    viewer.add_image(data, name=name, blending='additive', scale=scale, colormap=c_map, translate=translate, visible=False)

def _load_vectors_layer(viewer, name, path, scale, file_info, translate):
    # Arrow annotations are handled by the ArrowOverlay, not a Vectors layer
    if file_info.get('subtype') == 'arrows':
        _sync_arrow_overlay(viewer)
        return

    data = np.load(path, allow_pickle=True)
    vector_params = {
        'data': data, 'name': name, 'opacity': 1.0, 'edge_width': 1,
        'length': 1, 'edge_color': 'white',
    }
    if parse_version(napari.__version__) >= parse_version("0.7.0"):
        vector_params['head_width'] = 4
        vector_params['head_length'] = 6
    vector_params['translate'] = translate
    vector_params['scale'] = scale
    vector_params['visible'] = False
    viewer.add_vectors(**vector_params)

def _load_shapes_layer(viewer, name, path, scale, file_info, translate):
    """Load a Shapes layer from a saved .npy file (e.g. arrow annotations)."""
    subtype = file_info.get('subtype')

    # Arrow annotations are now rendered by the ArrowOverlay
    if subtype == 'arrows':
        _sync_arrow_overlay(viewer)
        return

    logger.warning("Unknown shapes subtype '%s' for layer '%s'.", subtype, name)

def _sync_arrow_overlay(viewer):
    """Trigger the ArrowOverlay to reload endpoints from the session file."""
    overlay = getattr(viewer.window, 'arrow_overlay', None)
    if overlay is not None:
        overlay.sync_from_session()

def restore_processed_layers(viewer: napari.Viewer, processed_files: dict, default_scale: tuple, canvas_offset_pixels: list = None, progress_callback=None):
    """
    Loads processed files (masks, points, etc.) from a session into the viewer.
    """
    num_files = len(processed_files)
    if num_files == 0:
        return

    # --- NEW: Calculate world-space translation from pixel offset ---
    translate = [0.0, 0.0, 0.0]
    if canvas_offset_pixels:
        # Convert pixel offset to world coordinate translation
        translate = (np.array(canvas_offset_pixels) * np.array(default_scale)).tolist()
        logger.debug("Applying canvas translation: %s", translate)

    computed_ids_names = []  # deferred: must be rebuilt after all mask layers are loaded

    for i, (name, file_info) in enumerate(processed_files.items()):
        if progress_callback:
            progress = int(((i + 1) / num_files) * 100)
            progress_callback(progress, f"Loading: {name}")

        # Handle old format for backward compatibility, or just ignore non-dict values
        if not isinstance(file_info, dict):
            logger.warning("Skipping layer '%s' with old/invalid format in session file.", name)
            continue

        # computed_ids entries have no file — recompute from mask after all layers are loaded
        if file_info.get('subtype') == 'computed_ids':
            computed_ids_names.append(name)
            continue

        path_str = file_info.get('path')
        layer_type = file_info.get('type')

        if layer_type == 'report': # Don't load reports as layers
            continue

        if not path_str or not layer_type:
            logger.warning("Skipping layer '%s' due to missing path or type information.", name)
            continue

        path = Path(path_str)
        if not path.exists():
            logger.warning("File not found for layer '%s': %s", name, path_str)
            continue

        try:
            if layer_type == 'points':
                _load_points_layer(viewer, name, path, default_scale, file_info, translate)
            elif layer_type == 'labels':
                _load_labels_layer(viewer, name, path, default_scale, file_info, translate)
            elif layer_type == 'image':
                _load_image_layer(viewer, name, path, default_scale, file_info, translate)
            elif layer_type == 'vectors':
                _load_vectors_layer(viewer, name, path, default_scale, file_info, translate)
            elif layer_type == 'shapes':
                _load_shapes_layer(viewer, name, path, default_scale, file_info, translate)
            elif layer_type == 'arrows':
                _sync_arrow_overlay(viewer)
            else:
                logger.warning("Unknown layer type '%s' for layer '%s'.", layer_type, name)

            logger.debug("Restored layer: %s", name)
        except Exception as e:
            logger.error("Error restoring layer '%s': %s", name, e)

    # Post-pass: recompute any _IDs layers that were registered as computed (no file).
    # Parent mask must already be loaded before this runs.
    for ids_name in computed_ids_names:
        # ids_name is always "{mask_name}_IDs"
        mask_name = ids_name[:-4]  # strip "_IDs"
        if mask_name in viewer.layers and isinstance(viewer.layers[mask_name], napari.layers.Labels):
            try:
                add_or_update_label_ids(viewer, viewer.layers[mask_name])
            except Exception as e:
                logger.error("Error recomputing IDs for '%s': %s", mask_name, e)

def add_or_update_puncta_layer(viewer: napari.Viewer, source_layer: napari.layers.Image, puncta_data: np.ndarray):
    """
    Adds or updates a puncta layer with full feature support.

    Handles merging new data, creating unique IDs, and setting up text display.
    """
    if puncta_data is None or puncta_data.shape[0] == 0:
        return

    layer_name = f"{source_layer.name}{constants.PUNCTA_SUFFIX}"
    coords = puncta_data[:, :3]

    # Create a DataFrame for the new properties
    new_features = pd.DataFrame({
        'Nucleus_ID': puncta_data[:, 3],
        'Intensity': puncta_data[:, 4],
        'SNR': puncta_data[:, 5]
    })

    text_params = {
        'string': '{puncta_id:.0f}',  # Format as integer
        'size': 8,
        'color': 'white',
        'translation': np.array([0, 5, 5]),
    }

    if layer_name in viewer.layers:
        # Update existing layer
        pts_layer = viewer.layers[layer_name]

        # Get existing data and features
        existing_coords = pts_layer.data
        existing_features = pts_layer.features

        # Determine next ID based on existing points
        if not existing_features.empty and 'puncta_id' in existing_features:
            max_id = existing_features['puncta_id'].dropna().max()
            next_id = int(max_id) + 1 if pd.notna(max_id) else 0
        else:
            next_id = 0

        # Assign new unique IDs to the new data
        new_features['puncta_id'] = np.arange(next_id, next_id + len(new_features))

        # Merge data
        combined_coords = segmentation.merge_puncta(existing_coords, coords)
        combined_features = pd.concat([existing_features, new_features], ignore_index=True)

        # Remove and recreate the layer atomically to avoid vispy OpenGL
        # access violations caused by intermediate redraws between separate
        # .data / .features / .text assignments.
        layer_scale = pts_layer.scale
        layer_translate = pts_layer.translate
        viewer.layers.remove(pts_layer)
        viewer.add_points(
            combined_coords,
            name=layer_name,
            size=3,
            face_color="yellow",
            scale=layer_scale,
            translate=layer_translate,
            features=combined_features,
            text=text_params,
            visible=False,
        )

    else:
        # Create new layer
        # Assign initial IDs
        new_features['puncta_id'] = np.arange(len(new_features))

        viewer.add_points(
            coords,
            name=layer_name,
            size=3,
            face_color="yellow",
            scale=source_layer.scale,
            translate=source_layer.translate,
            features=new_features,
            text=text_params,
            visible=False,
        )

    # --- Save full features to CSV ---
    out_dir = session.get_data("output_dir")
    if out_dir:
        puncta_layer = viewer.layers[layer_name]
        
        # Combine coordinates and features for saving
        coords_df = pd.DataFrame(puncta_layer.data, columns=['Z', 'Y', 'X'])
        full_df_to_save = pd.concat([puncta_layer.features.reset_index(drop=True), coords_df.reset_index(drop=True)], axis=1)

        # Define path and save
        reports_dir = Path(out_dir) / constants.REPORTS_DIR
        reports_dir.mkdir(exist_ok=True)
        csv_path = reports_dir / f"{layer_name}.csv"
        
        full_df_to_save.to_csv(csv_path, index=False)
        
        # Update session file to point to this new CSV
        session.set_processed_file(layer_name, str(csv_path), layer_type='points', metadata={'subtype': 'puncta_csv'})

def _add_or_replace_ids_layer(viewer, name, coords, labels, scale, translate=None):
    """Create or replace a _IDs text-overlay Points layer from pre-computed centroids.

    The IDs layer is always positioned directly above its parent mask layer
    so that it renders on top in 3D (napari draws higher-index layers later).
    Visibility is synced to the parent mask via ``_attach_ids_visibility_sync``.
    """
    import zfisher.ui.events as _events_mod

    if translate is None:
        translate = (0,) * len(scale)

    # Determine the parent mask's visibility for new layers
    parent_mask_name = name[:-4] if name.endswith("_IDs") else None
    parent_visible = False
    if parent_mask_name and parent_mask_name in viewer.layers:
        parent_visible = viewer.layers[parent_mask_name].visible

    if name in viewer.layers:
        # UPDATE IN-PLACE — avoids destroying GL buffers which causes vispy crashes
        layer = viewer.layers[name]
        layer.data = coords
        layer.properties = {'label': labels}
        layer.text = {'string': '{label}', 'size': 12, 'color': '#40b5d8', 'translation': np.array([0, -5, 0])}
        layer.scale = scale
        layer.translate = translate
    else:
        # CREATE NEW — only when the layer doesn't exist yet
        target_idx = None
        if parent_mask_name and parent_mask_name in viewer.layers:
            target_idx = list(viewer.layers).index(viewer.layers[parent_mask_name]) + 1

        # Use a very large size so that out_of_slice_display keeps every point
        # in the visible set regardless of the current Z-slice (napari hides
        # points farther than size/2 from the slice).  The dot itself is
        # transparent — only the text label matters visually.
        layer = viewer.add_points(
            coords, name=name, size=9999, face_color='transparent',
            border_color='transparent', border_width=0, scale=scale,
            translate=translate, properties={'label': labels},
            text={'string': '{label}', 'size': 12, 'color': '#40b5d8', 'translation': np.array([0, -5, 0])},
            blending='translucent_no_depth', visible=parent_visible,
        )
        layer.out_of_slice_display = True
        # Disable depth testing on the text visual so ID numbers render
        # in front of iso_categorical mask surfaces in 3D.
        try:
            layer.text.blending = 'translucent_no_depth'
        except Exception:
            pass

        # Position directly above parent mask so it renders on top in 3D
        if target_idx is not None:
            viewer.layers.move(len(viewer.layers) - 1, target_idx)

        # Attach visibility sync (idempotent — skips if already connected)
        if parent_mask_name and parent_mask_name in viewer.layers:
            _attach_ids_visibility_sync(viewer.layers[parent_mask_name], layer)


# --- Mask ↔ IDs visibility syncing -------------------------------------------

# Track connected callbacks so we don't double-connect.
# Key: id(mask_layer), Value: callback
_visibility_sync_callbacks = {}


def _attach_ids_visibility_sync(mask_layer, ids_layer):
    """When *mask_layer* visibility toggles, mirror it on *ids_layer*."""
    key = id(mask_layer)

    # Disconnect any previous callback for this mask (e.g. after IDs replacement)
    if key in _visibility_sync_callbacks:
        try:
            mask_layer.events.visible.disconnect(_visibility_sync_callbacks[key])
        except (TypeError, RuntimeError):
            pass

    def _sync_visible(event=None):
        try:
            ids_layer.visible = mask_layer.visible
        except RuntimeError:
            pass  # layer was destroyed

    mask_layer.events.visible.connect(_sync_visible)
    _visibility_sync_callbacks[key] = _sync_visible


def add_segmentation_results_to_viewer(viewer: napari.Viewer, source_layer: napari.layers.Image, masks: np.ndarray, centroids: np.ndarray):
    """
    Adds nuclei segmentation results (masks and centroids) to the viewer.

    Handles layer creation, naming, styling, and saving to the session.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    source_layer : napari.layers.Image
        The nuclei image layer that was segmented.
    masks : np.ndarray
        The (Z, Y, X) labeled integer mask of the segmented nuclei.
    centroids : np.ndarray
        The (N, 3) array of centroid coordinates.
    """
    logger.debug("add_segmentation_results_to_viewer: source='%s'", source_layer.name)
    out_dir = session.get_data("output_dir")
    seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR if out_dir else None

    if masks is not None:
        mask_layer_name = f"{source_layer.name}{constants.MASKS_SUFFIX}"
        if mask_layer_name in viewer.layers:
            import zfisher.ui.events as _events_mod
            layer = viewer.layers[mask_layer_name]
            _events_mod._programmatic_removal = True
            try:
                layer.data = masks
            finally:
                _events_mod._programmatic_removal = False
            layer.scale = source_layer.scale
            layer.refresh()
        else:
            layer = viewer.add_labels(masks, name=mask_layer_name, opacity=0.3, visible=False, scale=source_layer.scale)
        # Use iso_categorical for better 3D rendering of masks alongside points
        layer.rendering = 'iso_categorical'
        if seg_dir:
            mask_path = seg_dir / f"{mask_layer_name}.tif"
            tifffile.imwrite(mask_path, masks)
            session.set_processed_file(mask_layer_name, str(mask_path), layer_type='labels', metadata={'subtype': 'mask'})

        # Build the _IDs text overlay directly from the centroids we already
        # have — no need to re-run regionprops via add_or_update_label_ids.
        if centroids is not None and len(centroids) > 0:
            ids_name = f"{mask_layer_name}_IDs"
            id_labels = np.arange(len(centroids)) + 1
            _add_or_replace_ids_layer(
                viewer, ids_name, centroids, id_labels,
                scale=source_layer.scale, translate=layer.translate,
            )
            if not session.is_loading():
                session.set_processed_file(ids_name, "", layer_type='points', metadata={'subtype': 'computed_ids'})

    if centroids is not None:
        centroid_layer_name = f"{source_layer.name}{constants.CENTROIDS_SUFFIX}"
        ids = np.arange(len(centroids)) + 1
        if centroid_layer_name in viewer.layers:
            # Remove and recreate to avoid vispy stale-buffer issues
            import zfisher.ui.events as _events_mod
            old = viewer.layers[centroid_layer_name]
            layer_idx = list(viewer.layers).index(old)
            _events_mod._programmatic_removal = True
            try:
                viewer.layers.remove(old)
            finally:
                _events_mod._programmatic_removal = False
        else:
            layer_idx = None
        new_pts = viewer.add_points(
            centroids,
            name=centroid_layer_name,
            size=5,
            face_color='orange',
            border_color='transparent',
            border_width=0,
            scale=source_layer.scale,
            properties={'id': ids},
            blending='translucent_no_depth',
            visible=False,
        )
        if layer_idx is not None:
            viewer.layers.move(len(viewer.layers) - 1, layer_idx)
        if seg_dir:
            cent_path = seg_dir / f"{centroid_layer_name}.npy"
            np.save(cent_path, centroids)
            session.set_processed_file(centroid_layer_name, str(cent_path), layer_type='points', metadata={'subtype': 'centroids'})

def add_consensus_nuclei_to_viewer(viewer: napari.Viewer, r1_mask_layer: napari.layers.Labels, merged_mask: np.ndarray, pts1: list):
    """
    Adds or updates consensus nuclei results (merged mask and ID points) in the viewer.
    If the layers already exist they are updated in-place rather than duplicated.

    Parameters
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    r1_mask_layer : napari.layers.Labels
        The reference mask layer, used for scale and translate information.
    merged_mask : np.ndarray
        The consensus mask data to be added as a new layer.
    pts1 : list
        A list of dictionaries [{'coord':..., 'label':...}] for the ID points.
    """
    layer_name = constants.CONSENSUS_MASKS_NAME
    ids_layer_name = f"{layer_name}{constants.CONSENSUS_IDS_SUFFIX}"

    # Add or update merged mask layer
    if layer_name in viewer.layers:
        viewer.layers[layer_name].data = merged_mask
    else:
        layer = viewer.add_labels(
            merged_mask,
            name=layer_name,
            scale=r1_mask_layer.scale,
            translate=r1_mask_layer.translate,
            opacity=0.5,
            visible=False,
        )
        layer.rendering = 'iso_categorical'

    # Add or update ID points layer
    if pts1:
        coords = np.array([p['coord'] for p in pts1])
        labels = np.array([p['label'] for p in pts1])
        _add_or_replace_ids_layer(
            viewer, ids_layer_name, coords, labels,
            scale=r1_mask_layer.scale, translate=r1_mask_layer.translate,
        )

def add_or_update_label_ids(viewer: napari.Viewer, labels_layer: napari.layers.Labels):
    """
    Calculates centroids for a labels layer and displays them as text.

    This creates or updates a points layer named after the labels layer
    (e.g., 'MyMask_IDs') to show the ID number at the centroid of each label.

    Used by the mask editor refresh button and debounced auto-refresh after
    edits.  During initial segmentation, add_segmentation_results_to_viewer
    bypasses this and builds the _IDs layer directly from pre-computed
    centroids to avoid a redundant regionprops pass.
    """
    if not labels_layer:
        return
    logger.debug("add_or_update_label_ids: layer='%s'", labels_layer.name)

    pts_data = segmentation.get_mask_centroids(labels_layer.data)

    name = f"{labels_layer.name}_IDs"
    coords = np.array([p['coord'] for p in pts_data]) if pts_data else np.empty((0, labels_layer.ndim))
    labels = np.array([p['label'] for p in pts_data]) if pts_data else np.empty(0)

    if len(coords) > 0:
        _add_or_replace_ids_layer(
            viewer, name, coords, labels,
            scale=labels_layer.scale, translate=labels_layer.translate,
        )

    # Register in session so this IDs layer is recreated on next session load.
    if not session.is_loading():
        session.set_processed_file(name, "", layer_type='points', metadata={'subtype': 'computed_ids'})