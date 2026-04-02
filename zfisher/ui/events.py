import logging
import napari
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtCore import QTimer
from ..core import session
from .. import constants

logger = logging.getLogger(__name__)

# When True, on_layer_removed skips cascade deletion (mask↔IDs).
# Set by programmatic helpers like add_or_update_label_ids that remove
# layers only to recreate them immediately.
_programmatic_removal = False


def _set_programmatic_removal(value):
    """Set the programmatic removal flag (prevents file deletion on layer remove)."""
    global _programmatic_removal
    _programmatic_removal = value


def is_layer_locked(layer):
    """Check whether a layer has been marked as non-deletable."""
    return getattr(layer, '_locked', False)


def lock_layer(layer):
    """Mark a layer as non-deletable."""
    layer._locked = True


def _should_lock(layer):
    """Determine if a newly added layer should be automatically locked.

    Lock everything except puncta Points layers (which the user edits).
    """
    # Puncta layers are the only user-editable layers — leave them unlocked.
    if isinstance(layer, napari.layers.Points):
        if constants.PUNCTA_SUFFIX in layer.name:
            return False

    # Lock all Images, Labels, Vectors, and non-puncta Points.
    return True


def install_layer_lock(viewer):
    """Monkey-patch viewer.layers to prevent deletion of locked layers."""
    original_remove = viewer.layers.remove
    original_clear = viewer.layers.clear

    def guarded_remove(layer):
        if is_layer_locked(layer):
            viewer.status = f"Cannot delete locked layer: {layer.name}"
            return
        return original_remove(layer)

    def guarded_clear():
        # Unlock all layers before clearing (used by Reset / New Session)
        for layer in list(viewer.layers):
            layer._locked = False
        return original_clear()

    viewer.layers.remove = guarded_remove
    viewer.layers.clear = guarded_clear

# Track layers that already have listeners to prevent duplicate attachment.
# Maps layer id -> {'layer': layer_ref, 'sync_data': callback, 'sync_color': callback}
_attached_listeners = {}


def reset_events_state():
    """Clear all module-level state. Called on session reset."""
    _attached_listeners.clear()


def attach_puncta_listener(layer, name):
    """Attaches listeners to a points layer for auto-saving and color syncing."""
    if id(layer) in _attached_listeners:
        return

    def sync_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir:
            reports_dir = Path(out_dir) / constants.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)
            csv_path = reports_dir / f"{name}.csv"
            try:
                coords_df = pd.DataFrame(layer.data, columns=['Z', 'Y', 'X'])
                features = layer.features.reset_index(drop=True) if hasattr(layer, 'features') and not layer.features.empty else pd.DataFrame()
                full_df = pd.concat([features, coords_df], axis=1)
                full_df.to_csv(csv_path, index=False)
            except Exception as e:
                # Fallback: save coordinates only
                logger.warning("Could not save full puncta CSV for '%s': %s. Saving coordinates only.", name, e)
                np.savetxt(csv_path, layer.data, delimiter=',', header='Z,Y,X', comments='')
            session.set_processed_file(name, str(csv_path), layer_type='points', metadata={'subtype': 'puncta_csv'})

    _sync_color_guard = {'active': False}

    def sync_color(event=None):
        # Guard against re-entrant calls: setting face_color can re-emit
        # current_face_color events, causing a feedback loop during 2D/3D
        # mode transitions.
        if _sync_color_guard['active']:
            return
        _sync_color_guard['active'] = True
        try:
            layer.face_color = layer.current_face_color
        finally:
            _sync_color_guard['active'] = False

    layer.events.data.connect(sync_data)
    layer.events.current_face_color.connect(sync_color)

    _attached_listeners[id(layer)] = {
        'layer': layer,
        'sync_data': sync_data,
        'sync_color': sync_color,
    }


def detach_puncta_listener(layer):
    """Disconnects listeners previously attached to a points layer."""
    info = _attached_listeners.pop(id(layer), None)
    if info is None:
        return
    try:
        layer.events.data.disconnect(info['sync_data'])
    except (TypeError, RuntimeError):
        pass
    try:
        layer.events.current_face_color.disconnect(info['sync_color'])
    except (TypeError, RuntimeError):
        pass

def _try_set(widgets, widget_key, *attr_chain):
    """Safely traverse a chain of attributes and set the final one.
    Usage: _try_set(widgets, 'key', 'child_widget', 'param', value)
    """
    try:
        obj = widgets[widget_key]
        for attr in attr_chain[:-2]:
            obj = getattr(obj, attr)
        setattr(getattr(obj, attr_chain[-2]), 'value', attr_chain[-1])
    except (KeyError, AttributeError, ValueError):
        pass

def on_layer_inserted(event, widgets):
    """
    Handles auto-selecting layers in widgets when a new layer is added.
    """
    layer = event.value

    # Auto-lock protected layer types
    if _should_lock(layer):
        lock_layer(layer)

    def update_widgets():
        # First, refresh all dropdowns to ensure the new layer is in the list
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                w.reset_choices()

        # Now, try to intelligently set the value based on layer type and name
        if isinstance(layer, napari.layers.Image):
            nuc_upper = session.get_nuclear_channel().upper()
            if nuc_upper in layer.name.upper():
                if "R1" in layer.name.upper():
                    _try_set(widgets, 'dapi_segmentation', '_dapi_segmentation_widget', 'r1_layer', layer)
                    _try_set(widgets, 'automated_preprocessing', '_automated_preprocessing_magic_widget', 'r1_dapi_layer', layer)
                elif "R2" in layer.name.upper():
                    _try_set(widgets, 'dapi_segmentation', '_dapi_segmentation_widget', 'r2_layer', layer)
                    _try_set(widgets, 'automated_preprocessing', '_automated_preprocessing_magic_widget', 'r2_dapi_layer', layer)
            else:
                # Auto-select non-nuclear image layers in puncta detection
                _try_set(widgets, 'puncta_detection', '_puncta_widget', 'image_layer', layer)

        elif isinstance(layer, napari.layers.Points):
            if "centroids" in layer.name.lower():
                if "R1" in layer.name.upper():
                    _try_set(widgets, 'registration', '_registration_widget', 'r1_points', layer)
                elif "R2" in layer.name.upper():
                    _try_set(widgets, 'registration', '_registration_widget', 'r2_points', layer)

            # Auto-select the new points layer in editors/analysis widgets
            _try_set(widgets, 'puncta_editor', '_puncta_editor_widget', 'points_layer', layer)
            _try_set(widgets, 'colocalization', '_rule_builder', 'source_layer', layer)

            # If it's a puncta layer, ensure its listener is attached.
            if "puncta" in layer.name.lower():
                 attach_puncta_listener(layer, layer.name)

        elif isinstance(layer, napari.layers.Labels):
            name = layer.name.upper()
            # Heuristic for matching aligned/warped nuclear masks
            nuc_mask_upper = session.get_nuclear_channel().upper()
            if nuc_mask_upper in name and ("ALIGNED" in name or "WARPED" in name):
                if "R1" in name:
                    _try_set(widgets, 'nuclei_matching', '_nuclei_matching_widget', 'r1_mask_layer', layer)
                elif "R2" in name:
                    _try_set(widgets, 'nuclei_matching', '_nuclei_matching_widget', 'r2_mask_layer', layer)

            # General purpose editors
            _try_set(widgets, 'mask_editor', '_mask_editor_widget', 'mask_layer', layer)
            # Auto-select a mask for puncta detection, but not other puncta layers
            if "puncta" not in name.lower():
                _try_set(widgets, 'puncta_detection', '_puncta_widget', 'nuclei_layer', layer)

    # Use a QTimer to delay execution slightly, ensuring the layer is fully added
    QTimer.singleShot(100, update_widgets)

def _remove_layer_and_file(layer_name):
    """Remove a layer's session entry and delete its file from disk."""
    from pathlib import Path
    processed = session.get_data("processed_files", default={})
    file_info = processed.get(layer_name)
    if isinstance(file_info, dict):
        path = file_info.get('path')
        if path:
            try:
                p = Path(path)
                if p.exists():
                    p.unlink()
                    logger.info("Deleted file on disk: %s", path)
            except Exception as e:
                logger.warning("Could not delete file for '%s': %s", layer_name, e)
    session.remove_processed_file(layer_name)

def on_layer_removed(event, widgets):
    """
    Refreshes all widget dropdowns when a layer is removed.
    For Labels layers, also removes the associated _IDs points layer
    and cleans up the session registry.

    Cascade deletion (mask ↔ _IDs) is skipped when _programmatic_removal
    is set, since programmatic helpers remove-then-recreate layers and
    don't want the partner destroyed.
    """
    # Suppress custom controls during cascade deletion to prevent popup flashing
    from .viewer import _suppress_custom_controls
    import zfisher.ui.viewer as _viewer_mod
    was_suppressed = _viewer_mod._suppress_custom_controls
    _viewer_mod._suppress_custom_controls = True

    try:
        layer = event.value
        viewer = napari.current_viewer()

        if isinstance(layer, napari.layers.Labels):
            # Cascade: also remove orphan _IDs and _centroids layers
            if not _programmatic_removal:
                for suffix in ("_IDs", "_centroids"):
                    child_name = f"{layer.name}{suffix}"
                    if viewer and child_name in viewer.layers:
                        child_layer = viewer.layers[child_name]
                        child_layer._locked = False
                        viewer.layers.remove(child_layer)
            _remove_layer_and_file(layer.name)
            _remove_layer_and_file(f"{layer.name}_IDs")
            _remove_layer_and_file(f"{layer.name}_centroids")

        elif isinstance(layer, napari.layers.Points):
            _remove_layer_and_file(layer.name)
            detach_puncta_listener(layer)
            # Cascade: also remove the parent mask
            if not _programmatic_removal and layer.name.endswith("_IDs"):
                mask_name = layer.name[:-4]  # strip "_IDs"
                if viewer and mask_name in viewer.layers:
                    mask_layer = viewer.layers[mask_name]
                    mask_layer._locked = False
                    viewer.layers.remove(mask_layer)
    finally:
        pass  # Don't restore yet — let QTimer do it

    def update_choices():
        for w in widgets.values():
            if hasattr(w, "reset_choices"):
                try:
                    w.reset_choices()
                except Exception as e:
                    logger.debug("Could not reset choices for widget: %s", e)
        # Restore suppression state after all updates
        _viewer_mod._suppress_custom_controls = was_suppressed

    QTimer.singleShot(100, update_choices)
