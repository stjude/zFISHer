import logging
import time
import weakref
import napari
import numpy as np
from collections import deque
from magicgui import magicgui, widgets
from pathlib import Path
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QApplication, QFrame

from ...core import session, timing
from ...core import io
from .. import popups, viewer_helpers
from ..decorators import require_active_session
from ...core import label_edits
from ... import constants
from ._shared import make_divider as _make_divider, make_section_header as _make_section_header

logger = logging.getLogger(__name__)


class _MaskUndoStack:
    """Undo records for mask edits. A record is a list of ``(indices, old_values)``
    pieces, applied in reverse order on undo.

    Single-label edits push the voxels they changed (``push``). A paint or erase
    session collects napari's own history atoms from ``events.paint``
    (``begin_session`` / ``add_atoms`` / ``end_session``), so neither needs a copy
    of the volume. Extrude and erase-all still snapshot and diff the whole volume
    (``begin`` / ``end``); their records are flagged whole-volume.
    """
    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)
        self._pre_edit = None
        self._session = None    # pieces of the open paint/erase session

    def begin(self, data):
        """Snapshot the current data before a whole-volume edit."""
        self._pre_edit = data.copy()

    def end(self, data):
        """Compute diff against the snapshot and store it."""
        if self._pre_edit is None:
            return
        diff_mask = self._pre_edit != data
        if np.any(diff_mask):
            indices = np.where(diff_mask)
            self._append([(indices, self._pre_edit[indices])], whole=True)
        self._pre_edit = None

    def push(self, indices, old_values):
        """Store one edit: the voxels at ``indices`` held ``old_values``."""
        self._append([(indices, old_values)])

    def _append(self, pieces, whole=False):
        # An edit made while a paint session is open goes after the strokes
        # already made, so those are closed into a record of their own first.
        if self._session:
            self._stack.append((self._session, False))
            self._session = []
        self._stack.append((pieces, whole))

    @property
    def in_session(self):
        return self._session is not None

    def begin_session(self):
        """Start collecting paint/erase strokes into one record."""
        if self._session is None:
            self._session = []

    def add_atoms(self, atoms):
        """Add napari history atoms ``(indices, old_values, new_value)``."""
        if self._session is not None:
            self._session.extend((indices, old) for indices, old, _new in atoms)

    def end_session(self):
        """Store the strokes of the open session as one record."""
        if self._session:
            self._stack.append((self._session, False))
        self._session = None

    def undo(self, data):
        """Apply the last record in reverse. Returns ``(pieces, whole, touched)``,
        or None if there was nothing to undo. ``touched`` holds every label the
        undo removed or put back, or is None for a whole-volume record."""
        if not self._stack:
            return None
        pieces, whole = self._stack.pop()
        touched = None if whole else set()
        for indices, old_values in reversed(pieces):
            if touched is not None:
                touched.update(np.unique(data[indices]).tolist())
                touched.update(np.unique(np.asarray(old_values)).tolist())
            data[indices] = old_values
        if touched is not None:
            touched.discard(0)
        return pieces, whole, touched

    def clear(self):
        self._stack.clear()
        self._pre_edit = None
        self._session = None

    def __len__(self):
        return len(self._stack)


# Undo stacks per mask layer, so an undo can only write into the array its
# records came from. Keyed like _label_boxes: id(layer) with a weak reference to
# the array. A stack whose array was replaced (re-segmentation, consensus
# rebuild) is stale and replaced by an empty one; one whose array was freed is
# dropped.
_mask_undo_stacks = {}


def _undo_for(layer):
    """The undo stack of ``layer.data``, created if missing or stale."""
    key = id(layer)
    entry = _mask_undo_stacks.get(key)
    if entry is not None and entry[0]() is layer.data:
        return entry[1]

    def _drop(ref):
        if _mask_undo_stacks.get(key, (None,))[0] is ref:
            _mask_undo_stacks.pop(key, None)

    stack = _MaskUndoStack()
    _mask_undo_stacks[key] = (weakref.ref(layer.data, _drop), stack)
    return stack

# Label bounding boxes per mask layer, so single-label edits touch only the
# label's box (core.label_edits). Built on first use; dropped when the array is
# replaced and after a whole-volume edit or undo. Keyed by id(layer) with a weak
# reference to the array the boxes describe, so nothing here keeps a volume alive.
_label_boxes = {}


def _boxes_for(layer):
    """The ``LabelBoxes`` of ``layer.data``, built if missing or stale."""
    entry = _label_boxes.get(id(layer))
    if entry is not None and entry[0]() is layer.data:
        return entry[1]
    boxes = label_edits.LabelBoxes(layer.data)
    # napari's own undo and redo write history values back without firing any
    # event, so every value in that history must already lie inside a box.
    for item in (*getattr(layer, '_undo_history', ()), *getattr(layer, '_redo_history', ())):
        for indices, old_values, new_values in item:
            boxes.extend_values(indices, old_values)
            boxes.extend_values(indices, new_values)
    _label_boxes[id(layer)] = (weakref.ref(layer.data), boxes)
    if _extend_boxes_on_paint not in layer.events.paint.callbacks:
        layer.events.paint.connect(_extend_boxes_on_paint)
    if _drop_boxes_on_data not in layer.events.data.callbacks:
        layer.events.data.connect(_drop_boxes_on_data)
    return boxes


def _cached_boxes(layer):
    """The ``LabelBoxes`` of ``layer.data`` if built and current, else None."""
    entry = _label_boxes.get(id(layer))
    if entry is not None and entry[0]() is layer.data:
        return entry[1]
    return None


def _extend_boxes_on_paint(event):
    """Brush, eraser and fill strokes, from napari's ``events.paint``."""
    boxes = _cached_boxes(event.source)
    if boxes is not None:
        for indices, _old, new_values in event.value:
            boxes.extend_values(indices, new_values)


def _drop_boxes_on_data(event):
    """The whole array was assigned: the boxes describe the old one."""
    _label_boxes.pop(id(event.source), None)


def _update_boxes_after_undo(layer, record):
    """Undo put old values back: grow their boxes, or drop the boxes if the
    record was a whole-volume diff."""
    pieces, whole, _touched = record
    if whole:
        _label_boxes.pop(id(layer), None)
        return
    boxes = _cached_boxes(layer)
    if boxes is not None:
        for indices, old_values in pieces:
            boxes.extend_values(indices, old_values)


# Compressed tiles of each saved mask (core.io.LabelTiles), so a save compresses
# again only the tiles that changed. Keyed like _label_boxes. Built at a mask's
# first save, which compresses every tile; dropped when the array is replaced.
# Every in-place edit must mark its region (_mark_tiles). Brush, eraser and fill
# strokes are marked from events.paint; napari's own undo and redo fire no event
# and are found at the next save in napari's history. A mask saved tile by tile
# is saved whole again at the flush points (flush_pending_mask_saves), so the
# file is exact whenever the session ends even if an edit went unmarked.
_label_tiles = {}


class _TileEntry:
    def __init__(self, layer):
        self.layer_ref = weakref.ref(layer)
        self.data_ref = weakref.ref(layer.data)
        self.tiles = io.LabelTiles(layer.data)
        self.seen = _napari_history_items(layer)
        self.partial = False        # saved tile by tile since its last whole save


def _napari_history_items(layer):
    """napari's undo and redo items, keyed by id. The items are held, so no
    other object can take one of these ids while the dict lives."""
    items = (*getattr(layer, '_undo_history', ()), *getattr(layer, '_redo_history', ()))
    return {id(item): item for item in items}


def _tile_entry(layer):
    """The tile entry of ``layer.data`` if built and current, else None."""
    entry = _label_tiles.get(id(layer))
    if entry is not None and entry.data_ref() is layer.data and entry.tiles.fits(layer.data):
        return entry
    return None


def _tiles_for_save(layer):
    """The tile entry for saving ``layer``: built if missing or stale, with the
    regions of napari undo and redo since the last save marked."""
    entry = _tile_entry(layer)
    if entry is None:
        entry = _TileEntry(layer)
        _label_tiles[id(layer)] = entry
        if _mark_tiles_on_paint not in layer.events.paint.callbacks:
            layer.events.paint.connect(_mark_tiles_on_paint)
        if _drop_tiles_on_data not in layer.events.data.callbacks:
            layer.events.data.connect(_drop_tiles_on_data)
        return entry
    current = _napari_history_items(layer)
    for key, item in current.items():
        if key not in entry.seen:           # napari undo or redo
            for atom in item:
                entry.tiles.mark(label_edits.index_extent(atom[0]))
    entry.seen = current
    return entry


def _mark_tiles(layer, indices=None, whole=False):
    """Record an in-place edit of ``layer.data`` at ``indices`` (a fancy-index
    tuple), or of the whole volume, for the next save."""
    entry = _tile_entry(layer)
    if entry is None:
        return                  # the next save compresses everything anyway
    if whole:
        entry.tiles.mark_all()
    elif indices is not None:
        entry.tiles.mark(label_edits.index_extent(indices))


def _mark_tiles_on_paint(event):
    """Brush, eraser and fill strokes, from napari's ``events.paint``. The
    stroke's history item is recorded as seen, so the next save does not take
    it for a napari undo or redo."""
    entry = _tile_entry(event.source)
    if entry is None:
        return
    entry.seen[id(event.value)] = event.value
    for indices, _old, _new in event.value:
        entry.tiles.mark(label_edits.index_extent(indices))


def _drop_tiles_on_data(event):
    """The whole array was assigned: the tiles describe the old one."""
    _label_tiles.pop(id(event.source), None)


def _refresh_ids(viewer, layer):
    """Update the ID overlay of ``layer`` for the labels edited since the last
    update; a full refresh when the overlay is not tracked yet."""
    viewer_helpers.refresh_label_ids(viewer, layer, lambda: _boxes_for(layer))



def _selected_mask_fields(_args=None):
    """Timing fields for handlers that act on the editor's selected layer."""
    layer = _mask_editor_widget.mask_layer.value
    return {"layer": layer.name} if layer is not None else {}


def _resync_puncta_for_layer(viewer, layer):
    """Cascade a mask edit on ``layer`` into the puncta layers' ``Nucleus_ID``.

    Only meaningful for the consensus mask — per-round masks are not the
    source of truth for puncta nucleus assignments. No-op otherwise.
    """
    if viewer is None or layer is None:
        return
    if constants.CONSENSUS_MASKS_NAME not in layer.name:
        return
    try:
        viewer_helpers.resync_puncta_nucleus_ids(
            viewer, layer, remove_extranuclear=False, save_csv=True,
        )
    except Exception:
        logger.exception("resync_puncta_nucleus_ids failed after mask edit")


class MaskHighlighter:
    """Highlights the hovered nucleus by setting its color to red.

    napari 0.6.x uses a CyclicLabelColormap with a fixed-size colors array
    (default 50).  Labels are coloured by ``label_id % num_colors``, so
    different IDs can share a slot.  On enable we expand the colormap so every
    label in the mask gets its own unique slot, eliminating collisions.
    """

    _RED = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def __init__(self, viewer):
        self.viewer = viewer
        self.last_layer = None
        self.last_id = None
        self._original_color = None
        self.active = False
        self._expanded_layer = None  # track which layer we expanded

    # ---- colormap expansion ------------------------------------------------

    @staticmethod
    def _ensure_unique_colormap(layer):
        """Expand the cyclic colormap so every label ID has a unique slot."""
        from napari.utils.colormaps.colormap import CyclicLabelColormap

        max_id = int(layer.data.max())
        num_colors = len(layer.colormap.colors)
        if num_colors > max_id:
            return  # already large enough

        old_colors = layer.colormap.colors.copy()
        num_old = len(old_colors)
        num_new = max_id + 1
        new_colors = np.empty((num_new, 4), dtype=np.float32)
        for i in range(num_new):
            new_colors[i] = old_colors[i % num_old]

        layer.colormap = CyclicLabelColormap(
            colors=new_colors,
            seed=layer.colormap.seed,
            background_value=layer.colormap.background_value,
        )

    # ---- enable / disable --------------------------------------------------

    def enable(self):
        if not self.active:
            self.viewer.mouse_move_callbacks.append(self.on_mouse_move)
            self.active = True

    def disable(self):
        if self.active:
            if self.on_mouse_move in self.viewer.mouse_move_callbacks:
                self.viewer.mouse_move_callbacks.remove(self.on_mouse_move)
            self.reset_highlight()
            self.active = False

    # ---- highlight / reset -------------------------------------------------

    def reset_highlight(self):
        """Restores the original color of the last highlighted label."""
        if self.last_layer is not None and self.last_id is not None:
            if self._original_color is not None:
                self._set_label_color(self.last_layer, self.last_id, self._original_color)
            self._original_color = None
        self.last_layer = None
        self.last_id = None

    @staticmethod
    def _set_label_color(layer, label_id, color):
        """Write into the colormap colors array and trigger a GPU refresh."""
        num_colors = len(layer.colormap.colors)
        idx = int(label_id) % num_colors
        layer.colormap.colors[idx] = np.asarray(color, dtype=np.float32)
        layer.events.colormap()

    def perform_highlight(self, layer, label_id):
        """Sets the hovered label to opaque red."""
        # Expand colormap if this is a new layer or IDs exceed its size
        if layer is not self._expanded_layer:
            self._ensure_unique_colormap(layer)
            self._expanded_layer = layer

        num_colors = len(layer.colormap.colors)
        idx = int(label_id) % num_colors

        # Restore previous highlight first
        if self.last_id is not None and self.last_id != label_id and self._original_color is not None:
            self._set_label_color(layer, self.last_id, self._original_color)

        # Store original color before overriding
        self._original_color = layer.colormap.colors[idx].copy()
        self._set_label_color(layer, label_id, self._RED)

        self.last_layer = layer
        self.last_id = label_id
        self.viewer.status = f"Hovering Nucleus ID: {label_id} (Press 'C' to delete)"

    # ---- cursor lookup -------------------------------------------------------

    @staticmethod
    def _label_at_position(layer, position):
        """Return the label ID at *position* (world coords), or None.

        Tries the full ``get_value`` API first (handles 3-D ray-casting).
        Falls back to a direct voxel lookup so 2-D mode always works.
        """
        # Attempt 1: napari's get_value with ray-casting
        try:
            viewer = napari.current_viewer()
            val = layer.get_value(
                position,
                view_direction=viewer.camera.view_direction,
                dims_displayed=list(viewer.dims.displayed),
                world=True,
            )
            if val is not None:
                return int(val)
        except Exception:
            pass

        # Attempt 2: direct voxel lookup (reliable in 2-D and when ray-cast fails)
        try:
            coords = layer.world_to_data(position)
            idx = tuple(int(round(float(c))) for c in coords)
            if all(0 <= i < s for i, s in zip(idx, layer.data.shape)):
                return int(layer.data[idx])
        except Exception:
            pass

        return None

    # ---- mouse callback ----------------------------------------------------

    def on_mouse_move(self, viewer, event):
        # Runs on every mouse move: timed in aggregate, one line per 5 s window
        # and only when a call in that window took over 20 ms.
        t0 = time.perf_counter()
        try:
            self._handle_mouse_move(viewer, event)
        finally:
            _hover_timing.add(time.perf_counter() - t0)

    def _handle_mouse_move(self, viewer, event):
        if not self.active:
            return

        layer = _mask_editor_widget.mask_layer.value
        if not isinstance(layer, napari.layers.Labels):
            if self.last_id is not None:
                self.reset_highlight()
            return

        # Use event args directly for speed (called on every pixel of mouse movement)
        try:
            id_under_cursor = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=list(event.dims_displayed),
                world=True,
            )
        except Exception:
            return

        if id_under_cursor == self.last_id:
            return

        if id_under_cursor is None or id_under_cursor == 0:
            if self.last_id is not None:
                self.reset_highlight()
        else:
            self.perform_highlight(layer, id_under_cursor)

# Global instance
_highlighter = None
_hover_timing = timing.HoverAggregate("mask_hover")

@magicgui(
    call_button="Merge Nuclei",
    mask_layer={"label": "Layer to Edit", "tooltip": "The nuclei mask layer to edit."},
    source_id={"label": "Source ID", "tooltip": "ID of the nucleus to act on (merge source or delete target)."},
    target_id={"label": "Target ID", "tooltip": "ID of the nucleus to merge into. Source will be absorbed into Target."}
)
@require_active_session("Please start or load a session before editing masks.")
@timing.timed_action("mask_merge", fields=lambda a: {
    "layer": getattr(a["mask_layer"], "name", None), "source": a["source_id"], "target": a["target_id"]})
def _mask_editor_widget(
    mask_layer: "napari.layers.Labels",
    source_id: int = 0,
    target_id: int = 0
):
    """Merges two labels in the selected mask layer."""
    viewer = napari.current_viewer()

    if mask_layer is None:
        viewer.status = "No mask layer selected."
        return

    if source_id == target_id:
        viewer.status = "Source and Target IDs must be different."
        return
        
    indices = label_edits.replace_label(mask_layer.data, _boxes_for(mask_layer), source_id, target_id)
    if indices is None:
        viewer.status = f"ID {source_id} not found."
        return
    count = indices[0].size
    _undo_for(mask_layer).push(indices, mask_layer.data.dtype.type(source_id))
    _mark_tiles(mask_layer, indices)
    viewer_helpers.mark_label_ids_dirty(mask_layer, (source_id, target_id))

    logger.info("MASK EDIT: Merge ID %d into %d (%d pixels) on layer '%s'", source_id, target_id, count, mask_layer.name)
    viewer_helpers.redraw_labels_voxels(mask_layer, [indices])
    _schedule_save(mask_layer)
    QTimer.singleShot(50, lambda: _refresh_ids(viewer, mask_layer))
    _resync_puncta_for_layer(viewer, mask_layer)

    viewer.status = f"Merged ID {source_id} into {target_id} ({count} pixels)."

@timing.timed_action("mask_delete", fields=lambda a: {"layer": a["layer"].name, "label": a["label_id"]})
def _delete_label_inplace(layer, label_id, reset_mode=False):
    """Delete a label in-place and refresh without triggering a full data reassignment."""
    viewer = napari.current_viewer()

    # Reset mode before data mutation
    if reset_mode and hasattr(layer, 'mode'):
        layer.mode = 'pan_zoom'

    indices = label_edits.replace_label(layer.data, _boxes_for(layer), label_id, 0)
    if indices is not None:
        _undo_for(layer).push(indices, layer.data.dtype.type(label_id))
        _mark_tiles(layer, indices)
        viewer_helpers.mark_label_ids_dirty(layer, (label_id,))

    # Defer all visual updates to next event loop iteration so vispy
    # finishes any in-progress GL draw before buffers are modified.
    ids_name = f"{layer.name}_IDs"
    centroids_name = layer.name.replace(constants.MASKS_SUFFIX, constants.CENTROIDS_SUFFIX)

    @timing.timed_action("mask_deferred_refresh", fields=lambda _a: {"trigger": "delete"})
    def _deferred_updates():
        viewer_helpers.redraw_labels_voxels(layer, [indices])
        _refresh_ids(viewer, layer)
        # Re-show the IDs layer (look up fresh reference in case it was recreated)
        if ids_name in viewer.layers:
            viewer.layers[ids_name].visible = True
        # Remove the deleted ID from the centroids layer
        if centroids_name in viewer.layers:
            pts = viewer.layers[centroids_name]
            ids_arr = np.asarray(pts.properties.get('id', np.empty(0)))
            if len(ids_arr) > 0:
                keep = ids_arr != label_id
                viewer_helpers.set_points_data(pts, pts.data[keep])
                pts.properties = {'id': ids_arr[keep]}
                pts.refresh()
    QTimer.singleShot(50, _deferred_updates)
    # Debounce save to disk
    _schedule_save(layer)
    _resync_puncta_for_layer(viewer, layer)

_save_timer = QTimer()
_save_timer.setSingleShot(True)
_save_pending_layer = None

def _schedule_save(layer):
    """Debounced save — writes mask to disk 500ms after the last edit."""
    global _save_pending_layer
    _save_pending_layer = layer
    _save_timer.start(500)

_last_save = {}     # tile counts of the latest save, for its mask_save line


@timing.timed_action(
    "mask_save", fields=lambda a: {"layer": getattr(a["layer"], "name", None)},
    result_fields=lambda path: ({"bytes": path.stat().st_size, **_last_save} if path else {"saved": False}))
def _write_mask_to_disk(layer):
    """Write ``layer.data`` to ``segmentation/<name>.ome.tif`` and register it
    in the session. The one mask writer in this module.

    Compresses only the tiles marked since the last save.
    Never raises. A failed save puts the layer back into ``_dirty_masks`` so the
    next flush retries it (every save writes the whole file from the current
    tiles), and is logged and shown in the status bar instead of vanishing."""
    out_dir = session.get_data("output_dir")
    if not (out_dir and layer is not None and layer.name):
        return None
    seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
    mask_path = io.mask_path(seg_dir, layer.name)
    try:
        seg_dir.mkdir(exist_ok=True, parents=True)
        entry = _tiles_for_save(layer)
        full = entry.tiles.all_pending
        _last_save.clear()
        _last_save.update(tiles=entry.tiles.pending, all_tiles=full)
        io.write_label_tif(mask_path, layer.data, voxel_size=tuple(layer.scale), tiles=entry.tiles)
        entry.partial = not full
    except Exception as exc:
        _dirty_masks[id(layer)] = layer
        logger.error("Could not save mask '%s': %s", layer.name, exc, exc_info=True)
        viewer = napari.current_viewer()
        if viewer is not None:
            viewer.status = (f"Could not save mask '{layer.name}': {str(exc).rstrip('.')}. "
                             f"It stays unsaved and will be retried on the next edit.")
        return None
    session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
    return mask_path

def _do_save():
    global _save_pending_layer
    layer = _save_pending_layer
    _save_pending_layer = None
    _write_mask_to_disk(layer)

_save_timer.timeout.connect(_do_save)


# --- After-edit cascade -----------------------------------------------------
#
# napari fires ``layer.events.data`` only when the whole array is assigned.
# Brush, eraser and fill strokes commit through the layer's undo history and
# fire ``layer.events.paint`` instead. Both must end in the same three steps:
# persist the mask, refresh its ID overlay, and re-derive puncta Nucleus_IDs
# (consensus mask only). Strokes are frequent and the overlay refresh scans the
# whole volume, so strokes only mark the layer dirty; the flush runs after an
# idle period or when the tool is released.

_dirty_masks = {}           # id(layer) -> layer with unsaved voxel edits
_flush_timer = QTimer()
_flush_timer.setSingleShot(True)
_EDIT_FLUSH_MS = 800        # after a whole-array assignment (events.data)
_STROKE_FLUSH_MS = 3000     # idle time after the last brush/erase/fill stroke

_TOOL_MODES = ('paint', 'erase', 'fill')


@timing.timed_action("mask_after_edit", fields=lambda a: {
    "layer": getattr(a["layer"], "name", None), "refresh_ids": a["refresh_ids"]})
def _after_mask_edit(layer, viewer=None, refresh_ids=True):
    """Persist ``layer``, optionally refresh its ID overlay, and cascade the edit
    into puncta ``Nucleus_ID`` (no-op unless ``layer`` is the consensus mask)."""
    if layer is None:
        return
    _dirty_masks.pop(id(layer), None)
    _write_mask_to_disk(layer)
    viewer = viewer or napari.current_viewer()
    if viewer is None:
        return
    if refresh_ids:
        ids_name = f"{layer.name}_IDs"
        if ids_name in viewer.layers:
            _refresh_ids(viewer, layer)
    _resync_puncta_for_layer(viewer, layer)


def _mark_mask_dirty(layer, delay_ms=_EDIT_FLUSH_MS):
    """Record an unsaved edit on ``layer`` and (re)start the flush timer."""
    if layer is None:
        return
    _dirty_masks[id(layer)] = layer
    _flush_timer.start(delay_ms)


def _flush_mask_edits(layer=None, refresh_ids=None):
    """Run the after-edit cascade for ``layer`` (or every dirty layer) if it has
    unsaved edits. ``refresh_ids=None`` refreshes the overlay unless the layer is
    still in a painting tool, in which case the refresh waits for tool release."""
    targets = [layer] if layer is not None else list(_dirty_masks.values())
    for lyr in targets:
        if id(lyr) not in _dirty_masks:
            continue
        do_refresh = refresh_ids
        if do_refresh is None:
            do_refresh = str(getattr(lyr, 'mode', '')) not in _TOOL_MODES
        _after_mask_edit(lyr, refresh_ids=do_refresh)


_flush_timer.timeout.connect(lambda: _flush_mask_edits())


def flush_pending_mask_saves():
    """Write every mask edit that is not on disk yet, now, and drop the timers.

    Saves are debounced (``_schedule_save``, ``_mark_mask_dirty``), so the last
    edits sit in memory for up to a few seconds. Call this before the session
    changes (new, load, reset) and when the app quits: otherwise they are lost
    at quit, or the timer fires after the switch and writes the old session's
    mask into the new session's folder. A save that fails here is logged and
    dropped, since its layer is about to go."""
    global _save_pending_layer
    _save_timer.stop()
    _flush_timer.stop()
    pending, _save_pending_layer = _save_pending_layer, None
    dirty = list(_dirty_masks.values())
    _dirty_masks.clear()
    # Masks saved tile by tile are compressed whole once more, so the file is
    # exact even if some edit went unmarked.
    to_save = [pending] if pending is not None else []
    for entry in list(_label_tiles.values()):
        layer = entry.layer_ref()
        if entry.partial and layer is not None and _tile_entry(layer) is entry:
            entry.tiles.mark_all()
            to_save.append(layer)
    for layer in dirty:
        _after_mask_edit(layer, refresh_ids=False)
    for i, layer in enumerate(to_save):
        if all(layer is not other for other in (*dirty, *to_save[:i])):
            _write_mask_to_disk(layer)
    for layer in list(_dirty_masks.values()):
        logger.error("Mask '%s' could not be saved before the session changed; "
                     "its latest edits are not on disk.", layer.name)
    _dirty_masks.clear()


_app = QApplication.instance()
if _app is not None:
    _app.aboutToQuit.connect(flush_pending_mask_saves)

def delete_mask_under_mouse(viewer):
    """Deletes the mask label currently under the mouse cursor."""
    global _highlighter

    # 1. If hover-edit has a cached ID, use it directly (fastest path).
    if _highlighter and _highlighter.active and _highlighter.last_id is not None:
        layer = _highlighter.last_layer
        id_to_delete = _highlighter.last_id

        if layer and id_to_delete:
            _highlighter.reset_highlight()
            _delete_label_inplace(layer, id_to_delete, reset_mode=True)
            logger.info("MASK EDIT: Deleted nucleus ID %d (hover mode) on layer '%s'", id_to_delete, layer.name)
            viewer.status = f"Deleted Nucleus ID {id_to_delete}"
            return

    # 2. Fallback: query the label under the cursor right now.
    #    Prefer the widget's selected mask layer over the viewer's active layer
    #    so the user doesn't have to manually select the Labels layer first.
    layer = _mask_editor_widget.mask_layer.value
    if not isinstance(layer, napari.layers.Labels):
        layer = viewer.layers.selection.active
    if isinstance(layer, napari.layers.Labels):
        val = MaskHighlighter._label_at_position(layer, viewer.cursor.position)
        if val is not None and val > 0:
            if _highlighter and _highlighter.active:
                _highlighter.reset_highlight()
            _delete_label_inplace(layer, val, reset_mode=True)
            logger.info("MASK EDIT: Deleted nucleus ID %d (cursor) on layer '%s'", val, layer.name)
            viewer.status = f"Deleted Nucleus ID {val}"

from qtpy.QtWidgets import QPushButton, QLabel as _QLabel

# --- Merge Section ---
merge_label = _make_section_header("Merge Nuclei")
delete_btn = widgets.PushButton(text="Delete Source ID", tooltip="Delete the nucleus with the Source ID from the mask.")

# --- Paint Section ---
paint_label = _make_section_header("Paint New Mask")
paint_chk = widgets.CheckBox(text="Paint (New ID)", tooltip="Enable paint mode to draw a new mask region with the Source ID.")
from qtpy.QtWidgets import QHBoxLayout as _QHBoxLayout, QWidget as _QWidget

# Extrude ID — label + spinbox + button in one row
_extrude_spinbox = widgets.SpinBox(label="", min=1, max=99999, value=1, tooltip="Nucleus ID to fill through all Z slices using its largest XY cross-section.")
_extrude_btn = QPushButton("Extrude (Fill Z)")
_extrude_btn.setToolTip("Fill the specified nucleus through all Z slices using its largest XY cross-section.")
_extrude_row = _QWidget()
_extrude_row_layout = _QHBoxLayout(_extrude_row)
_extrude_row_layout.setContentsMargins(0, 2, 0, 2)
_extrude_row_layout.setSpacing(4)
_extrude_row_layout.addWidget(_QLabel("Nucleus ID:"))
_extrude_row_layout.addWidget(_extrude_spinbox.native, 1)
_extrude_row_layout.addWidget(_extrude_btn)

# --- Erase Section ---
erase_label = _make_section_header("Erase")
erase_chk = widgets.CheckBox(text="Erase", tooltip="Enable eraser to remove mask pixels in a brush radius.")
brush_size_slider = widgets.Slider(label="", min=1, max=40, value=10, tooltip="Brush size for paint and erase tools. Syncs with layer controls.")

# Delete by ID — label + spinbox + button in one row
_delete_id_spinbox = widgets.SpinBox(label="", min=1, max=99999, value=1, tooltip="Nucleus ID to remove completely from the mask and ID layers.")
_delete_id_btn = QPushButton("Delete ID")
_delete_id_btn.setToolTip("Remove this nucleus from the mask and ID layers. All voxels with this label will be erased.")
_delete_id_row = _QWidget()
_delete_id_row_layout = _QHBoxLayout(_delete_id_row)
_delete_id_row_layout.setContentsMargins(0, 2, 0, 2)
_delete_id_row_layout.setSpacing(4)
_delete_id_row_layout.addWidget(_QLabel("Nucleus ID:"))
_delete_id_row_layout.addWidget(_delete_id_spinbox.native, 1)
_delete_id_row_layout.addWidget(_delete_id_btn)

hover_chk = widgets.CheckBox(text="Hover Edit Mode (Red + 'C' to Del)", tooltip="Highlight nuclei under the cursor in red. Press the 'C' key to delete the highlighted nucleus.")

# --- Utilities ---
undo_btn = widgets.PushButton(text="Undo", tooltip="Revert the last mask edit operation.")

# --- Rebuild layout from scratch using a fresh QVBoxLayout ---
# The magicgui QFormLayout leaves ghost rows when widgets are reparented,
# so we replace it entirely.
from qtpy.QtWidgets import QVBoxLayout as _QVBoxLayout

# Detach all children from the old magicgui layout
_old_layout = _mask_editor_widget.native.layout()
while _old_layout.count():
    _item = _old_layout.takeAt(0)
    _w = _item.widget()
    if _w:
        _w.setParent(None)

# Install a fresh vertical layout
from qtpy.QtWidgets import QSizePolicy as _QSizePolicy
try:
    from shiboken6 import delete as _sip_delete
except ImportError:
    from sip import delete as _sip_delete
_sip_delete(_old_layout)
_layout = _QVBoxLayout(_mask_editor_widget.native)
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
# Match the size policy of class-based Containers so the widget shrinks properly
# inside the nested QToolBox (QToolBox wraps pages in QScrollArea).
_mask_editor_widget.native.setSizePolicy(_QSizePolicy.Preferred, _QSizePolicy.Preferred)
_mask_editor_widget.native.setMinimumWidth(0)
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox, QLabel
for child in _mask_editor_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
for child in _mask_editor_widget.native.findChildren(QAbstractSpinBox) + _mask_editor_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)

# --- Header / info / description — added directly to avoid double-nesting ---
_hdr = widgets.Label(value="Mask Editor")
_hdr.native.setObjectName("widgetHeader")
_inf = widgets.Label(value="<i>Merge, paint, erase, and delete nuclei masks. Changes auto-save to disk.</i>")
_inf.native.setObjectName("widgetInfo")
_layout.addWidget(_hdr.native)
_layout.addWidget(_inf.native)
_layout.addWidget(_make_divider())

# --- Target Layer section ---
_target_label = _make_section_header("Target Layer")
_layout.addWidget(_target_label)
_target_desc = _QLabel("Select the nuclei mask layer to edit.")
_target_desc.setWordWrap(True)
_target_desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
_layout.addWidget(_target_desc)

_mask_editor_widget.mask_layer.label = "Layer to Edit:"
_layer_form = widgets.Container(labels=True)
_layer_form.extend([_mask_editor_widget.mask_layer])
_layer_form.native.layout().setContentsMargins(0, 4, 0, 4)
_layout.addWidget(_layer_form.native)

from qtpy.QtWidgets import QSpacerItem as _QSpacerItem, QSizePolicy as _QSizePolicy
_spacer = lambda: _QSpacerItem(0, 20, _QSizePolicy.Minimum, _QSizePolicy.Fixed)

# --- Merge Nuclei section ---
_layout.addSpacerItem(_spacer())
_layout.addWidget(_make_divider())
_layout.addWidget(merge_label)
_merge_desc = _QLabel("Combine two nuclei into one. Nucleus A is absorbed into Nucleus B.")
_merge_desc.setWordWrap(True)
_merge_desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
_layout.addWidget(_merge_desc)

# Nucleus A & B — in the same Container so QFormLayout aligns label columns
_mask_editor_widget.source_id.label = "Nucleus A:"
_mask_editor_widget.target_id.label = "Nucleus B:"
_merge_form = widgets.Container(labels=True)
_merge_form.extend([_mask_editor_widget.source_id, _mask_editor_widget.target_id])
_merge_form.native.layout().setContentsMargins(0, 2, 0, 2)
_layout.addWidget(_merge_form.native)

_layout.addWidget(_mask_editor_widget.call_button.native)

# --- Paint section ---
_layout.addSpacerItem(_spacer())
_layout.addWidget(_make_divider())
_layout.addWidget(paint_label)
_paint_desc = _QLabel("Draw new mask regions or extend existing nuclei. Use Paint New ID to auto-assign the next available label.")
_paint_desc.setWordWrap(True)
_paint_desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
_layout.addWidget(_paint_desc)

# Paint ID — magicgui SpinBox in Container(labels=True) for proper resize
_paint_id_spinbox = widgets.SpinBox(label="Nucleus ID:", min=1, max=99999, value=1, tooltip="Enter the nucleus label ID to paint with.")
_paint_id_form = widgets.Container(labels=True)
_paint_id_form.extend([_paint_id_spinbox])
_paint_id_form.native.layout().setContentsMargins(0, 2, 0, 2)
# Icon imports for paint/erase buttons
from pathlib import Path as _Path
from qtpy.QtGui import QIcon as _QIcon, QPixmap as _QPixmap, QPainter as _QPainter
from qtpy.QtCore import QByteArray as _QByteArray
from qtpy.QtSvg import QSvgRenderer as _QSvgRenderer

# Full-width paint toggle button with napari paint icon
_paint_toggle_btn = QPushButton()
_paint_toggle_btn.setCheckable(True)
_paint_toggle_btn.setToolTip("Toggle paint mode using the specified nucleus ID.")
_paint_icon_path = _Path(napari.__file__).parent / "resources" / "icons" / "paint.svg"
if _paint_icon_path.exists():
    _paint_svg = _paint_icon_path.read_text()
    _paint_svg_white = _paint_svg.replace('viewBox=', 'fill="white" viewBox=')
    _p_renderer = _QSvgRenderer(_QByteArray(_paint_svg_white.encode()))
    _p_pixmap = _QPixmap(24, 24)
    _p_pixmap.fill(Qt.transparent)
    _p_painter = _QPainter(_p_pixmap)
    _p_renderer.render(_p_painter)
    _p_painter.end()
    _paint_toggle_btn.setIcon(_QIcon(_p_pixmap))
else:
    _paint_toggle_btn.setText("Paint")

@timing.timed_action("mask_paint_toggle", fields=lambda a: {**_selected_mask_fields(), "on": a["checked"]})
def _on_paint_toggle(checked):
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        _paint_toggle_btn.setChecked(False)
        return
    if checked:
        erase_chk.value = False
        hover_chk.value = False
        _erase_toggle_btn.blockSignals(True)
        _erase_toggle_btn.setChecked(False)
        _erase_toggle_btn.blockSignals(False)
        viewer = napari.current_viewer()
        if viewer:
            viewer.layers.selection.active = layer
        layer.selected_label = _paint_id_spinbox.value
        _undo_for(layer).begin_session()  # collect the strokes of this paint session
        layer.mode = 'paint'
        if viewer:
            viewer.status = f"Painting with ID {_paint_id_spinbox.value}"
    else:
        _undo_for(layer).end_session()  # store the strokes of the paint session
        layer.mode = 'pan_zoom'
        # Persist the strokes and cascade into puncta now; the mode-change
        # handler does the same on the way out, and the second call is a no-op.
        _flush_mask_edits(layer, refresh_ids=False)
        # Refresh IDs after painting so new/modified labels get their centroid label
        viewer = napari.current_viewer()
        if viewer:
            ids_name = f"{layer.name}_IDs"
            if ids_name in viewer.layers:
                QTimer.singleShot(100, lambda: _refresh_ids(viewer, layer))
        _resync_puncta_for_layer(viewer, layer)

# PySide6 passes a slot as many arguments as its own code object declares, so
# the timing wrapper (*args) would receive no `checked`; the lambda forwards it.
_paint_toggle_btn.clicked.connect(lambda checked: _on_paint_toggle(checked))

# Paint eyedropper button — pick mode to select a nucleus ID
_paint_pick_btn = QPushButton()
_paint_pick_btn.setCheckable(True)
_paint_pick_btn.setToolTip("Pick a nucleus ID from the canvas (eyedropper).")
_pick_icon_path = _Path(napari.__file__).parent / "resources" / "icons" / "picker.svg"
if _pick_icon_path.exists():
    _pick_svg = _pick_icon_path.read_text()
    _pick_svg_white = _pick_svg.replace('viewBox=', 'fill="white" viewBox=')
    _pick_r = _QSvgRenderer(_QByteArray(_pick_svg_white.encode()))
    _pick_px = _QPixmap(24, 24)
    _pick_px.fill(Qt.transparent)
    _pick_pa = _QPainter(_pick_px)
    _pick_r.render(_pick_pa)
    _pick_pa.end()
    _paint_pick_btn.setIcon(_QIcon(_pick_px))
else:
    _paint_pick_btn.setText("Pick")

def _on_paint_pick_toggle(checked):
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        _paint_pick_btn.setChecked(False)
        return
    if checked:
        # Uncheck paint and erase
        _paint_toggle_btn.blockSignals(True)
        _paint_toggle_btn.setChecked(False)
        _paint_toggle_btn.blockSignals(False)
        _erase_toggle_btn.blockSignals(True)
        _erase_toggle_btn.setChecked(False)
        _erase_toggle_btn.blockSignals(False)
        erase_chk.value = False
        hover_chk.value = False
        viewer = napari.current_viewer()
        if viewer:
            viewer.layers.selection.active = layer
        layer.mode = 'pick'
        if viewer:
            viewer.status = "Click a nucleus to select its ID."
    else:
        layer.mode = 'pan_zoom'

_paint_pick_btn.clicked.connect(_on_paint_pick_toggle)

# Row with paint + pick buttons side by side
_paint_btn_row = _QWidget()
_paint_btn_layout = _QHBoxLayout(_paint_btn_row)
_paint_btn_layout.setContentsMargins(0, 0, 0, 0)
_paint_btn_layout.setSpacing(4)
_paint_btn_layout.addWidget(_paint_toggle_btn, 1)
_paint_btn_layout.addWidget(_paint_pick_btn, 0)

_layout.addWidget(_paint_id_form.native)
_layout.addWidget(_paint_btn_row)

# Paint New ID button — auto-assigns max+1
_paint_new_btn = QPushButton("Paint New ID")
_paint_new_btn.setToolTip("Start painting with the next available nucleus ID (max + 1).")

@timing.timed_action("mask_paint_new", fields=_selected_mask_fields)
def _on_paint_new(_checked=False):
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        return
    new_id = int(layer.data.max()) + 1
    _paint_id_spinbox.value = new_id
    _extrude_spinbox.value = new_id
    erase_chk.value = False
    hover_chk.value = False
    _erase_toggle_btn.blockSignals(True)
    _erase_toggle_btn.setChecked(False)
    _erase_toggle_btn.blockSignals(False)
    viewer = napari.current_viewer()
    if viewer:
        viewer.layers.selection.active = layer
    layer.selected_label = new_id
    _undo_for(layer).begin_session()  # collect the strokes of this paint session
    layer.mode = 'paint'
    _paint_toggle_btn.blockSignals(True)
    _paint_toggle_btn.setChecked(True)
    _paint_toggle_btn.blockSignals(False)
    if viewer:
        viewer.status = f"Painting new nucleus with ID {new_id}"

_paint_new_btn.clicked.connect(_on_paint_new)
_layout.addWidget(_paint_new_btn)

# Paint brush size row — independent from erase brush size
_paint_brush_slider = widgets.Slider(label="", min=1, max=40, value=10, tooltip="Brush size for painting (1-40 pixels). Syncs with layer brush size controls.")
_paint_brush_slider.label = "Brush Size:"
_paint_brush_form = widgets.Container(labels=True)
_paint_brush_form.extend([_paint_brush_slider])
_paint_brush_form.native.layout().setContentsMargins(0, 2, 0, 2)
_layout.addWidget(_paint_brush_form.native)

# Paint slider → layer.brush_size (only when paint mode is active)
def _on_paint_brush_changed(val):
    global _syncing_brush
    if _syncing_brush:
        return
    layer = _mask_editor_widget.mask_layer.value
    if layer and hasattr(layer, 'mode') and 'paint' in str(layer.mode).lower():
        _syncing_brush = True
        layer.brush_size = val
        _syncing_brush = False

_paint_brush_slider.changed.connect(_on_paint_brush_changed)

# Refresh IDs button
_refresh_ids_btn = QPushButton("Refresh IDs")
_refresh_ids_btn.setToolTip(
    "Recompute centroids, refresh the ID labels overlay, and re-sync puncta "
    "Nucleus_IDs from the current mask. Edits do this automatically; use this "
    "to force it."
)

@timing.timed_action("mask_refresh_ids", fields=_selected_mask_fields)
def _on_refresh_ids(_checked=False):
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not viewer or not layer:
        return
    viewer_helpers.add_or_update_label_ids(viewer, layer)
    # Also resync puncta Nucleus_IDs (no-op for non-consensus mask layers)
    _resync_puncta_for_layer(viewer, layer)
    viewer.status = "IDs refreshed."

_refresh_ids_btn.clicked.connect(_on_refresh_ids)
_layout.addWidget(_refresh_ids_btn)

# --- Extrude Mask section ---
_layout.addSpacerItem(_spacer())
_layout.addWidget(_make_divider())
_extrude_label = _make_section_header("Extrude Mask")
_layout.addWidget(_extrude_label)
_extrude_desc = _QLabel("Fills the specified nucleus through all Z slices using the largest XY footprint across the stack.")
_extrude_desc.setWordWrap(True)
_extrude_desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
_layout.addWidget(_extrude_desc)
_layout.addWidget(_extrude_row)

# --- Erase section ---
_layout.addSpacerItem(_spacer())
_layout.addWidget(_make_divider())
_layout.addWidget(erase_label)
_erase_desc = _QLabel("Remove mask pixels with a brush, delete a nucleus by ID, or hover over nuclei and press C to delete.")
_erase_desc.setWordWrap(True)
_erase_desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
_layout.addWidget(_erase_desc)

_erase_toggle_btn = QPushButton()
_erase_toggle_btn.setCheckable(True)
_erase_toggle_btn.setToolTip("Toggle erase mode on the selected mask layer.")
# Use napari's erase icon
_erase_icon_path = _Path(napari.__file__).parent / "resources" / "icons" / "erase.svg"
if _erase_icon_path.exists():
    # Recolor the SVG to white for dark theme visibility
    _erase_svg = _erase_icon_path.read_text()
    _erase_svg_white = _erase_svg.replace('viewBox=', 'fill="white" viewBox=')
    _renderer = _QSvgRenderer(_QByteArray(_erase_svg_white.encode()))
    _pixmap = _QPixmap(24, 24)
    _pixmap.fill(Qt.transparent)
    _painter = _QPainter(_pixmap)
    _renderer.render(_painter)
    _painter.end()
    _erase_toggle_btn.setIcon(_QIcon(_pixmap))
else:
    _erase_toggle_btn.setText("Erase")

@timing.timed_action("mask_erase_toggle", fields=lambda a: {**_selected_mask_fields(), "on": a["checked"]})
def _on_erase_toggle(checked):
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        _erase_toggle_btn.setChecked(False)
        return
    if checked:
        _paint_toggle_btn.blockSignals(True)
        _paint_toggle_btn.setChecked(False)
        _paint_toggle_btn.blockSignals(False)
        hover_chk.value = False
        viewer = napari.current_viewer()
        if viewer:
            viewer.layers.selection.active = layer
        _undo_for(layer).begin_session()  # collect the strokes of this erase session
        layer.mode = 'erase'
    else:
        _undo_for(layer).end_session()  # store the strokes of the erase session
        layer.mode = 'pan_zoom'
        viewer = napari.current_viewer()
        # Persist the erased voxels (strokes never fire events.data) and cascade.
        _flush_mask_edits(layer, refresh_ids=False)
        _resync_puncta_for_layer(viewer, layer)

_erase_toggle_btn.clicked.connect(lambda checked: _on_erase_toggle(checked))  # see paint toggle

# Erase eyedropper button — pick mode to select ID for delete
_erase_pick_btn = QPushButton()
_erase_pick_btn.setCheckable(True)
_erase_pick_btn.setToolTip("Pick a nucleus ID from the canvas to set the delete ID.")
if _pick_icon_path.exists():
    _erase_pick_btn.setIcon(_QIcon(_pick_px))
else:
    _erase_pick_btn.setText("Pick")

def _on_erase_pick_toggle(checked):
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        _erase_pick_btn.setChecked(False)
        return
    if checked:
        # Uncheck paint and erase
        _paint_toggle_btn.blockSignals(True)
        _paint_toggle_btn.setChecked(False)
        _paint_toggle_btn.blockSignals(False)
        _erase_toggle_btn.blockSignals(True)
        _erase_toggle_btn.setChecked(False)
        _erase_toggle_btn.blockSignals(False)
        _paint_pick_btn.blockSignals(True)
        _paint_pick_btn.setChecked(False)
        _paint_pick_btn.blockSignals(False)
        erase_chk.value = False
        hover_chk.value = False
        viewer = napari.current_viewer()
        if viewer:
            viewer.layers.selection.active = layer
        # Use pick mode; the selected_label sync will update the erase delete spinbox
        layer.mode = 'pick'
        if viewer:
            viewer.status = "Click a nucleus to set the delete ID."
    else:
        layer.mode = 'pan_zoom'

_erase_pick_btn.clicked.connect(_on_erase_pick_toggle)

# Row with erase + pick buttons side by side
_erase_btn_row = _QWidget()
_erase_btn_layout = _QHBoxLayout(_erase_btn_row)
_erase_btn_layout.setContentsMargins(0, 0, 0, 0)
_erase_btn_layout.setSpacing(4)
_erase_btn_layout.addWidget(_erase_toggle_btn, 1)
_erase_btn_layout.addWidget(_erase_pick_btn, 0)

_layout.addWidget(_erase_btn_row)

# Brush size row with label
brush_size_slider.label = "Brush Size:"
_brush_form = widgets.Container(labels=True)
_brush_form.extend([brush_size_slider])
_brush_form.native.layout().setContentsMargins(0, 2, 0, 2)
_layout.addWidget(_brush_form.native)

_layout.addWidget(_delete_id_row)

_erase_all_btn = QPushButton("Erase All Masks")
_erase_all_btn.setToolTip("Remove ALL nuclei from this mask layer. This cannot be undone for large masks.")
_layout.addWidget(_erase_all_btn)

_layout.addWidget(hover_chk.native)

# --- Undo ---
_layout.addSpacerItem(_spacer())
_layout.addWidget(_make_divider())
_layout.addSpacerItem(_QSpacerItem(0, 40, _QSizePolicy.Minimum, _QSizePolicy.Fixed))
_layout.addWidget(undo_btn.native)

_layout.addStretch(1)

@require_active_session("Please start or load a session before editing masks.")
def _on_paint(value: bool):
    if not session.get_data("output_dir"):
        _paint_toggle_btn.blockSignals(True)
        _paint_toggle_btn.setChecked(False)
        _paint_toggle_btn.blockSignals(False)
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            erase_chk.value = False
            hover_chk.value = False  # hover refresh conflicts with paint interaction
            viewer.layers.selection.active = layer
            layer.mode = 'paint'
            layer.n_edit_dimensions = 2
            new_id = int(layer.data.max()) + 1
            layer.selected_label = new_id
            logger.info("MASK EDIT: Paint mode ON, new ID=%d, layer='%s'", new_id, layer.name)
            viewer.status = f"Painting Mode. New ID: {new_id}"
        elif layer.mode == 'paint':
            layer.mode = 'pan_zoom'
            logger.info("MASK EDIT: Paint mode OFF")
            viewer.status = "Painting Mode Off."

@require_active_session("Please start or load a session before editing masks.")
def _on_erase(value: bool):
    if not session.get_data("output_dir"):
        erase_chk.value = False
        return
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if layer:
        if value:
            _paint_toggle_btn.blockSignals(True)
            _paint_toggle_btn.setChecked(False)
            _paint_toggle_btn.blockSignals(False)
            hover_chk.value = False
            viewer.layers.selection.active = layer
            layer.mode = 'erase'
            logger.info("MASK EDIT: Erase mode ON, layer='%s'", layer.name)
            viewer.status = "Erase Mode ON. Use brush to erase mask pixels."
        else:
            layer.mode = 'pan_zoom'
            logger.info("MASK EDIT: Erase mode OFF")
            viewer.status = "Erase Mode Off."

@require_active_session("Please start or load a session before editing masks.")
@timing.timed_action("mask_extrude", fields=_selected_mask_fields)
def _on_extrude(_checked=False):
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer: return
    
    label_id = _extrude_spinbox.value
    if label_id == 0:
        viewer.status = "Select a label to extrude (cannot extrude 0)."
        return
        
    if layer.ndim != 3:
        viewer.status = "Extrusion only works on 3D layers."
        return

    if not np.any(layer.data == label_id):
        viewer.status = f"Label {label_id} not found in mask."
        return

    logger.info("MASK EDIT: Extrude ID %d, layer='%s'", label_id, layer.name)

    # Switch to pan_zoom before data mutation to avoid brush cursor repaint
    if hasattr(layer, 'mode') and layer.mode != 'pan_zoom':
        layer.mode = 'pan_zoom'

    undo = _undo_for(layer)
    undo.begin(layer.data)
    # Compute union XY footprint across all Z slices
    union_2d = np.any(layer.data == label_id, axis=0)
    # Only fill where the voxel is background (0) or already this label —
    # never overwrite other nuclei that overlap the XY footprint
    fill_mask = union_2d[np.newaxis, :, :] & ((layer.data == 0) | (layer.data == label_id))
    layer.data[fill_mask] = label_id
    undo.end(layer.data)
    _label_boxes.pop(id(layer), None)  # the label grew outside its box
    _mark_tiles(layer, whole=True)

    # Refresh mask visual and recompute IDs (in-place update, no hide needed)
    layer.refresh()
    viewer_helpers.add_or_update_label_ids(viewer, layer)

    _schedule_save(layer)
    _resync_puncta_for_layer(viewer, layer)
    viewer.status = f"Extruded ID {label_id} through all Z slices."

@require_active_session("Please start or load a session before editing masks.")
@timing.timed_action("mask_delete_button", fields=_selected_mask_fields)
def _on_delete():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    src = _mask_editor_widget.source_id.value
    if layer and src > 0:
        if label_edits.count_label(layer.data, _boxes_for(layer), src) > 0:
            logger.info("MASK EDIT: Delete ID %d on layer '%s'", src, layer.name)
            _delete_label_inplace(layer, src)
            viewer.status = f"Deleted ID {src}."
        else:
            viewer.status = f"ID {src} not found."

@hover_chk.changed.connect
def _on_hover_mode(value: bool):
    viewer = napari.current_viewer()
    global _highlighter
    if _highlighter is None:
        _highlighter = MaskHighlighter(viewer)

    if value:
        _highlighter.enable()
        logger.info("MASK EDIT: Hover edit mode ON")
        viewer.status = "Hover Edit Mode ON. Nuclei turn red. Press 'C' to delete."
    else:
        _highlighter.disable()
        logger.info("MASK EDIT: Hover edit mode OFF")
        viewer.status = "Hover Edit Mode OFF."

@require_active_session("Please start or load a session before refreshing IDs.")
@timing.timed_action("mask_undo", fields=_selected_mask_fields)
def _on_mask_undo():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        viewer.status = "No mask layer selected."
        return
    undo = _undo_for(layer)  # only this layer's own edits can be undone into it
    # If currently in paint/erase mode, finalize the current session first
    in_edit_mode = False
    if hasattr(layer, 'mode'):
        mode = str(layer.mode).lower()
        if 'paint' in mode or 'erase' in mode:
            in_edit_mode = True
            undo.end_session()
    record = undo.undo(layer.data)
    if record:
        _update_boxes_after_undo(layer, record)
        if record[1]:
            _mark_tiles(layer, whole=True)
        else:
            for indices, _old in record[0]:
                _mark_tiles(layer, indices)
        touched = record[2]
        if touched is None:
            viewer_helpers.invalidate_label_ids(layer)
        else:
            viewer_helpers.mark_label_ids_dirty(layer, touched)
        if touched is None:
            layer.refresh()
        else:
            viewer_helpers.redraw_labels_voxels(layer, [ix for ix, _old in record[0]])
        # Restart a session if still in edit mode so future strokes are tracked
        if in_edit_mode:
            undo.begin_session()
        # Refresh IDs and centroids to match reverted mask data
        centroids_name = layer.name.replace(constants.MASKS_SUFFIX, constants.CENTROIDS_SUFFIX)
        @timing.timed_action("mask_deferred_refresh", fields=lambda _a: {"trigger": "undo"})
        def _deferred_undo_refresh():
            _refresh_ids(viewer, layer)
            # Update the centroids of the labels the undo touched; after a
            # whole-volume undo, rebuild them all from the reverted mask
            if centroids_name in viewer.layers:
                pts = viewer.layers[centroids_name]
                if touched is None:
                    from ...core.segmentation import get_mask_centroids
                    pts_data = get_mask_centroids(layer.data)
                    coords = np.array([p['coord'] for p in pts_data]) if pts_data else np.empty((0, layer.ndim))
                    ids = np.array([p['label'] for p in pts_data]) if pts_data else np.empty(0)
                else:
                    changed = label_edits.label_centroids(layer.data, _boxes_for(layer), touched)
                    coords, ids = label_edits.update_centroid_table(
                        pts.data, pts.properties.get('id', ()), changed, layer.ndim)
                viewer_helpers.set_points_data(pts, coords)
                pts.properties = {'id': ids}
                pts.refresh()
        QTimer.singleShot(50, _deferred_undo_refresh)
        _schedule_save(layer)
        _resync_puncta_for_layer(viewer, layer)
        logger.info("MASK EDIT: Undo on layer '%s' (%d remaining)", layer.name, len(undo))
        viewer.status = f"Undo ({len(undo)} remaining)."
    else:
        # Restart the session if in edit mode so we don't lose future strokes
        if in_edit_mode:
            undo.begin_session()
        viewer.status = "Nothing to undo."

@timing.timed_action("mask_delete_id", fields=_selected_mask_fields)
def _on_delete_id():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        if viewer:
            viewer.status = "No mask layer selected."
        return
    label_id = _delete_id_spinbox.value
    if label_id == 0:
        viewer.status = "Cannot delete background (ID 0)."
        return
    if label_edits.count_label(layer.data, _boxes_for(layer), label_id) == 0:
        viewer.status = f"ID {label_id} not found in mask."
        return
    _delete_label_inplace(layer, label_id, reset_mode=True)
    logger.info("MASK EDIT: Deleted nucleus ID %d on layer '%s'", label_id, layer.name)
    viewer.status = f"Deleted Nucleus ID {label_id}."

@timing.timed_action("mask_erase_all", fields=_selected_mask_fields)
def _on_erase_all():
    from qtpy.QtWidgets import QMessageBox
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        if viewer:
            viewer.status = "No mask layer selected."
        return
    if not np.any(layer.data):
        if viewer:
            viewer.status = "Mask is already empty."
        return
    n_ids = int(layer.data.max())
    with timing.user_wait():
        reply = QMessageBox.warning(
            _mask_editor_widget.native,
            "Erase All Masks",
            f"This will erase all {n_ids} nuclei from '{layer.name}'.\n\nAre you sure?",
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
    if reply != QMessageBox.Yes:
        return
    if hasattr(layer, 'mode'):
        layer.mode = 'pan_zoom'
    undo = _undo_for(layer)
    undo.begin(layer.data)
    layer.data[:] = 0
    undo.end(layer.data)
    _mark_tiles(layer, whole=True)
    ids_name = f"{layer.name}_IDs"
    centroids_name = layer.name.replace(constants.MASKS_SUFFIX, constants.CENTROIDS_SUFFIX)

    @timing.timed_action("mask_deferred_refresh", fields=lambda _a: {"trigger": "erase_all"})
    def _deferred_updates():
        layer.refresh()
        viewer_helpers.add_or_update_label_ids(viewer, layer)
        if ids_name in viewer.layers:
            viewer.layers[ids_name].visible = True
        # Clear centroids layer if it exists
        if centroids_name in viewer.layers:
            pts = viewer.layers[centroids_name]
            viewer_helpers.set_points_data(pts, np.empty((0, layer.ndim)))
            pts.properties = {'id': np.empty(0)}
            pts.refresh()
    QTimer.singleShot(50, _deferred_updates)
    _schedule_save(layer)
    _resync_puncta_for_layer(viewer, layer)
    logger.info("MASK EDIT: Erased all masks on layer '%s' (%d IDs)", layer.name, n_ids)
    if viewer:
        viewer.status = f"Erased all {n_ids} nuclei from {layer.name}."

# Connect signals after defining functions
_erase_all_btn.clicked.connect(_on_erase_all)
_delete_id_btn.clicked.connect(_on_delete_id)
paint_chk.changed.connect(_on_paint)
erase_chk.changed.connect(_on_erase)
_extrude_btn.clicked.connect(_on_extrude)
delete_btn.clicked.connect(_on_delete)
undo_btn.clicked.connect(_on_mask_undo)
# Note: hover_chk is already connected via @hover_chk.changed.connect decorator on _on_hover_mode

# --- Sync erase checkbox with layer mode changes from layer controls ---
_mask_editor_widget._mode_connection = None

_syncing_brush = False  # guard against recursive sync

_was_painting = False  # track paint mode to refresh IDs on exit
_was_erasing = False   # track erase mode for undo snapshots

@timing.timed_action("mask_mode_change", fields=lambda a: {
    "mode": getattr(a["event"], "mode", getattr(a["event"], "value", None))})
def _sync_erase_from_layer_mode(event):
    """Keep paint/erase buttons and checkbox in sync when mode changes via layer controls."""
    global _was_painting, _was_erasing
    mode = str(event.mode) if hasattr(event, 'mode') else str(event.value)
    mode_lower = mode.lower()
    is_erase = 'erase' in mode_lower
    is_paint = 'paint' in mode_lower

    layer = _mask_editor_widget.mask_layer.value

    # If we just left paint or erase mode, store the session's strokes
    if (_was_painting and not is_paint) or (_was_erasing and not is_erase):
        if layer:
            _undo_for(layer).end_session()

    # If we just entered paint or erase mode (via layer controls), start a session
    if layer:
        if (is_paint and not _was_painting) or (is_erase and not _was_erasing):
            _undo_for(layer).begin_session()

    # If we just left paint or erase mode (widget toggle or napari's own
    # controls): persist the strokes and cascade into puncta now, then refresh
    # IDs and ensure visibility. Brush/erase strokes never fire events.data, so
    # this is where their edits reach disk.
    if (_was_painting and not is_paint) or (_was_erasing and not is_erase):
        if layer:
            _flush_mask_edits(layer, refresh_ids=False)
            viewer = napari.current_viewer()
            if viewer:
                ids_name = f"{layer.name}_IDs"
                def _refresh_and_show(v=viewer, l=layer, n=ids_name):
                    _refresh_ids(v, l)
                    if n in v.layers:
                        v.layers[n].visible = True
                QTimer.singleShot(100, _refresh_and_show)
    _was_painting = is_paint
    _was_erasing = is_erase

    is_pick = 'pick' in mode_lower

    # Apply the correct brush size for the new mode
    layer = _mask_editor_widget.mask_layer.value
    if layer and hasattr(layer, 'brush_size'):
        if is_paint:
            layer.brush_size = _paint_brush_slider.value
        elif is_erase:
            layer.brush_size = brush_size_slider.value

    # Sync paint toggle button
    if _paint_toggle_btn.isChecked() != is_paint:
        _paint_toggle_btn.blockSignals(True)
        _paint_toggle_btn.setChecked(is_paint)
        _paint_toggle_btn.blockSignals(False)
    # Sync erase toggle button
    if _erase_toggle_btn.isChecked() != is_erase:
        _erase_toggle_btn.blockSignals(True)
        _erase_toggle_btn.setChecked(is_erase)
        _erase_toggle_btn.blockSignals(False)
    # Sync pick/eyedropper buttons — uncheck if mode is no longer pick
    if not is_pick:
        if _paint_pick_btn.isChecked():
            _paint_pick_btn.blockSignals(True)
            _paint_pick_btn.setChecked(False)
            _paint_pick_btn.blockSignals(False)
        if _erase_pick_btn.isChecked():
            _erase_pick_btn.blockSignals(True)
            _erase_pick_btn.setChecked(False)
            _erase_pick_btn.blockSignals(False)
    # Sync legacy checkbox
    if erase_chk.value != is_erase:
        erase_chk.changed.disconnect(_on_erase)
        erase_chk.value = is_erase
        erase_chk.changed.connect(_on_erase)

def _sync_brush_from_layer(event=None):
    """Keep the active slider in sync when brush size changes via layer controls."""
    global _syncing_brush
    if _syncing_brush:
        return
    layer = _mask_editor_widget.mask_layer.value
    if layer and hasattr(layer, 'brush_size'):
        val = int(layer.brush_size)
        mode = str(layer.mode).lower()
        _syncing_brush = True
        if 'erase' in mode:
            if brush_size_slider.value != val:
                brush_size_slider.value = max(brush_size_slider.min, min(brush_size_slider.max, val))
        elif 'paint' in mode:
            if _paint_brush_slider.value != val:
                _paint_brush_slider.value = max(_paint_brush_slider.min, min(_paint_brush_slider.max, val))
        _syncing_brush = False

def _on_erase_brush_slider_changed(val):
    """Push erase slider value to layer.brush_size (only when erase mode is active)."""
    global _syncing_brush
    if _syncing_brush:
        return
    layer = _mask_editor_widget.mask_layer.value
    if layer and hasattr(layer, 'mode') and 'erase' in str(layer.mode).lower():
        _syncing_brush = True
        layer.brush_size = val
        _syncing_brush = False

brush_size_slider.changed.connect(_on_erase_brush_slider_changed)

_mask_editor_widget._brush_connection = None
_mask_editor_widget._selected_label_connection = None

# --- selected_label ↔ spinbox sync (Option C) ---
_syncing_selected_label = False

def _sync_spinboxes_from_layer(event=None):
    """When pick mode or layer controls change selected_label, update our spinboxes."""
    global _syncing_selected_label
    if _syncing_selected_label:
        return
    layer = _mask_editor_widget.mask_layer.value
    if not layer or not hasattr(layer, 'selected_label'):
        return
    val = int(layer.selected_label)
    if val < 1:
        return
    _syncing_selected_label = True
    try:
        if _erase_pick_btn.isChecked():
            # Erase eyedropper: only update the delete spinbox
            if _delete_id_spinbox.value != val:
                _delete_id_spinbox.value = max(_delete_id_spinbox.min, min(_delete_id_spinbox.max, val))
        elif _paint_pick_btn.isChecked():
            # Paint eyedropper: update paint and extrude spinboxes only
            if _paint_id_spinbox.value != val:
                _paint_id_spinbox.value = max(_paint_id_spinbox.min, min(_paint_id_spinbox.max, val))
            if _extrude_spinbox.value != val:
                _extrude_spinbox.value = max(_extrude_spinbox.min, min(_extrude_spinbox.max, val))
        else:
            # No eyedropper active (e.g. layer controls pick button): update paint and extrude
            if _paint_id_spinbox.value != val:
                _paint_id_spinbox.value = max(_paint_id_spinbox.min, min(_paint_id_spinbox.max, val))
            if _extrude_spinbox.value != val:
                _extrude_spinbox.value = max(_extrude_spinbox.min, min(_extrude_spinbox.max, val))
    finally:
        _syncing_selected_label = False

def _on_paint_spinbox_changed(val):
    """Push paint spinbox value to layer.selected_label and sync extrude spinbox."""
    global _syncing_selected_label
    if _syncing_selected_label:
        return
    layer = _mask_editor_widget.mask_layer.value
    if not layer or not hasattr(layer, 'selected_label'):
        return
    _syncing_selected_label = True
    try:
        layer.selected_label = val
        if _extrude_spinbox.value != val:
            _extrude_spinbox.value = max(_extrude_spinbox.min, min(_extrude_spinbox.max, val))
    finally:
        _syncing_selected_label = False

_paint_id_spinbox.changed.connect(_on_paint_spinbox_changed)

def _on_mask_layer_changed(event=None):
    """Connect/disconnect mode, brush size, and selected_label sync when the selected mask layer changes."""
    # Disconnect previous mode sync
    if _mask_editor_widget._mode_connection is not None:
        old_layer, old_cb = _mask_editor_widget._mode_connection
        try:
            old_layer.events.mode.disconnect(old_cb)
        except (RuntimeError, TypeError):
            pass
        _mask_editor_widget._mode_connection = None
    # Disconnect previous brush size sync
    if _mask_editor_widget._brush_connection is not None:
        old_layer, old_cb = _mask_editor_widget._brush_connection
        try:
            old_layer.events.brush_size.disconnect(old_cb)
        except (RuntimeError, TypeError):
            pass
        _mask_editor_widget._brush_connection = None
    # Disconnect previous selected_label sync
    if _mask_editor_widget._selected_label_connection is not None:
        old_layer, old_cb = _mask_editor_widget._selected_label_connection
        try:
            old_layer.events.selected_label.disconnect(old_cb)
        except (RuntimeError, TypeError):
            pass
        _mask_editor_widget._selected_label_connection = None

    layer = _mask_editor_widget.mask_layer.value
    if isinstance(layer, napari.layers.Labels):
        layer.events.mode.connect(_sync_erase_from_layer_mode)
        _mask_editor_widget._mode_connection = (layer, _sync_erase_from_layer_mode)
        layer.events.brush_size.connect(_sync_brush_from_layer)
        _mask_editor_widget._brush_connection = (layer, _sync_brush_from_layer)
        layer.events.selected_label.connect(_sync_spinboxes_from_layer)
        _mask_editor_widget._selected_label_connection = (layer, _sync_spinboxes_from_layer)
        # Initialize sliders/spinboxes to current layer values
        _sync_brush_from_layer()
        _sync_spinboxes_from_layer()

_mask_editor_widget.mask_layer.changed.connect(_on_mask_layer_changed)

# --- Visibility & selection sync between dropdown and layer list ---

_syncing_layer_selection = False  # guard against recursive sync

def _on_mask_layer_dropdown_changed(event=None):
    """When Layer to Edit dropdown changes, show mask + DAPI, hide others, set active."""
    global _syncing_layer_selection
    if _syncing_layer_selection:
        return
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        return
    viewer = napari.current_viewer()
    if not viewer:
        return

    _syncing_layer_selection = True
    try:
        # Determine the related DAPI layer name (e.g. "R2 - DAPI_masks" → "R2 - DAPI")
        # Strip "_masks" suffix to get the base DAPI channel name
        base_name = layer.name.replace("_masks", "")

        for l in viewer.layers:
            # Show: the mask layer, its IDs, its centroids, and the base DAPI image
            if l is layer:
                l.visible = True
            elif l.name == f"{layer.name}_IDs":
                l.visible = True
            elif l.name == f"{layer.name}_centroids":
                l.visible = True
            elif l.name == base_name and isinstance(l, napari.layers.Image):
                l.visible = True
            else:
                l.visible = False

        viewer.layers.selection.active = layer
    finally:
        _syncing_layer_selection = False

_mask_editor_widget.mask_layer.changed.connect(_on_mask_layer_dropdown_changed)

def _on_viewer_layer_selection_changed(event=None):
    """When a mask or DAPI layer is selected in the layer list, set the dropdown."""
    global _syncing_layer_selection
    if _syncing_layer_selection:
        return
    # Skip during batch loading / segmentation
    from .. import viewer as _viewer_mod
    if getattr(_viewer_mod, '_suppress_custom_controls', False):
        return
    try:
        viewer = napari.current_viewer()
        if not viewer:
            return
        active = viewer.layers.selection.active
        if active is None:
            return

        # Find the relevant mask layer name
        mask_name = None
        if active.name.endswith("_masks"):
            mask_name = active.name
        elif active.name.endswith("_masks_IDs"):
            mask_name = active.name.replace("_IDs", "")
        elif active.name.endswith("_masks_centroids"):
            mask_name = active.name.replace("_centroids", "")
        else:
            # Check if it's a DAPI image layer with a matching mask
            candidate = f"{active.name}_masks"
            if candidate in viewer.layers:
                mask_name = candidate

        if mask_name and mask_name in viewer.layers:
            mask_layer = viewer.layers[mask_name]
            if isinstance(mask_layer, napari.layers.Labels):
                current = _mask_editor_widget.mask_layer.value
                if current is not mask_layer:
                    _syncing_layer_selection = True
                    try:
                        _mask_editor_widget.reset_choices()
                        _mask_editor_widget.mask_layer.value = mask_layer
                    finally:
                        _syncing_layer_selection = False
        else:
            # Selected a non-mask layer — disable hover edit mode
            if hover_chk.value:
                hover_chk.value = False
    except Exception:
        pass  # Silently skip — layer may not be fully initialized yet


def deactivate_hover_edit():
    """Disable hover edit mode. Called when switching away from the mask editor widget."""
    if hover_chk.value:
        hover_chk.value = False


def reset_mask_editor_state():
    """Save pending edits, then clear all module-level state. Called on
    session reset, while the session being reset is still the active one."""
    global _syncing_brush, _syncing_layer_selection, _highlighter
    flush_pending_mask_saves()
    _mask_undo_stacks.clear()
    _label_boxes.clear()
    _label_tiles.clear()
    viewer_helpers.reset_label_ids_tracking()
    _syncing_brush = False
    _syncing_layer_selection = False
    if _highlighter is not None:
        _highlighter.disable()
        _highlighter = None
    hover_chk.value = False
    _paint_toggle_btn.setChecked(False)
    _erase_toggle_btn.setChecked(False)

# Connect after a short delay so the viewer is fully initialized
def _connect_layer_selection_sync():
    viewer = napari.current_viewer()
    if viewer:
        viewer.layers.selection.events.changed.connect(_on_viewer_layer_selection_changed)

QTimer.singleShot(500, _connect_layer_selection_sync)

# --- Auto-saving for selected mask layer ---

# Store a reference to the layer and its callbacks to allow disconnection
_mask_editor_widget._current_layer = None
_mask_editor_widget._current_callbacks = None

def _create_edit_callbacks(layer):
    """Return ``(on_data, on_paint)`` callbacks that mark ``layer`` dirty.

    ``events.data`` fires on whole-array assignment (merge, delete, extrude,
    undo): flush soon. ``events.paint`` fires once per committed brush, eraser
    or fill stroke: flush after an idle period, or on tool release via
    ``_sync_erase_from_layer_mode``.
    """
    def _on_data(event=None):
        _mark_mask_dirty(layer, _EDIT_FLUSH_MS)

    def _on_paint(event=None):
        _undo_for(layer).add_atoms(getattr(event, 'value', None) or ())
        _mark_mask_dirty(layer, _STROKE_FLUSH_MS)

    return _on_data, _on_paint

@_mask_editor_widget.mask_layer.changed.connect
def _on_mask_layer_changed(new_layer: "napari.layers.Labels"):
    """Disconnects the old listeners and connects new ones to the selected layer."""
    global _was_painting, _was_erasing
    old_layer = _mask_editor_widget._current_layer
    old_callbacks = _mask_editor_widget._current_callbacks

    # A paint or erase session belongs to the layer it was started on: store
    # its strokes now. The new layer gets a session if it is already in paint
    # or erase mode, so its strokes can be undone too.
    if old_layer is not None and old_layer is not new_layer:
        _undo_for(old_layer).end_session()
    mode = str(getattr(new_layer, 'mode', '')).lower() if new_layer else ''
    _was_painting = 'paint' in mode
    _was_erasing = 'erase' in mode
    if _was_painting or _was_erasing:
        _undo_for(new_layer).begin_session()

    if old_layer is not None and old_callbacks:
        # Do not leave the previous layer's edits unsaved.
        _flush_mask_edits(old_layer)
        for emitter, cb in zip((old_layer.events.data, old_layer.events.paint), old_callbacks):
            if cb in emitter.callbacks:
                emitter.disconnect(cb)

    if new_layer:
        on_data, on_paint = _create_edit_callbacks(new_layer)
        new_layer.events.data.connect(on_data)
        new_layer.events.paint.connect(on_paint)
        _mask_editor_widget._current_layer = new_layer
        _mask_editor_widget._current_callbacks = (on_data, on_paint)
    else:
        _mask_editor_widget._current_layer = None
        _mask_editor_widget._current_callbacks = None

# --- Public API ---
# Expose the magicgui widget directly as mask_editor_widget (no wrapper Container).
# This gives the same single-level nesting as dapi_segmentation_widget and
# new_session_widget, which prevents right-edge clipping on panel resize.
mask_editor_widget = _mask_editor_widget
mask_editor_widget._mask_editor_widget = _mask_editor_widget
