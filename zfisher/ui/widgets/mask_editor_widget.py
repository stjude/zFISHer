import logging
import napari
import numpy as np
from collections import deque
from magicgui import magicgui, widgets
import tifffile
from pathlib import Path
from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QFrame

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session
from ...core import segmentation
from ... import constants

logger = logging.getLogger(__name__)


class _MaskUndoStack:
    """Stores diffs (changed indices + old values) to keep memory usage low."""
    def __init__(self, maxlen=10):
        self._stack = deque(maxlen=maxlen)
        self._pre_edit = None

    def begin(self, data):
        """Snapshot the current data before an edit."""
        self._pre_edit = data.copy()

    def end(self, data):
        """Compute diff against the snapshot and store it."""
        if self._pre_edit is None:
            return
        diff_mask = self._pre_edit != data
        if np.any(diff_mask):
            indices = np.where(diff_mask)
            self._stack.append((indices, self._pre_edit[indices]))
        self._pre_edit = None

    def undo(self, data):
        """Apply the last stored diff in reverse. Returns True if undo was performed."""
        if not self._stack:
            return False
        indices, old_values = self._stack.pop()
        data[indices] = old_values
        return True

    def clear(self):
        self._stack.clear()
        self._pre_edit = None

    def __len__(self):
        return len(self._stack)


_mask_undo = _MaskUndoStack()

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

@magicgui(
    call_button="Merge IDs",
    mask_layer={"label": "Layer to Edit", "tooltip": "The nuclei mask layer to edit."},
    source_id={"label": "Source ID", "tooltip": "ID of the nucleus to act on (merge source or delete target)."},
    target_id={"label": "Target ID", "tooltip": "ID of the nucleus to merge into. Source will be absorbed into Target."}
)
@require_active_session("Please start or load a session before editing masks.")
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
        
    count = np.sum(mask_layer.data == source_id)
    
    if count == 0:
        viewer.status = f"ID {source_id} not found."
        return
        
    logger.info("MASK EDIT: Merge ID %d into %d (%d pixels) on layer '%s'", source_id, target_id, count, mask_layer.name)
    _mask_undo.begin(mask_layer.data)
    mask_layer.data[mask_layer.data == source_id] = target_id
    _mask_undo.end(mask_layer.data)
    mask_layer.refresh()
    _remove_id_from_points_layer(mask_layer, source_id)
    _schedule_save(mask_layer)

    viewer.status = f"Merged ID {source_id} into {target_id} ({count} pixels)."

def _delete_label_inplace(layer, label_id, reset_mode=False):
    """Delete a label in-place and refresh without triggering a full data reassignment."""
    viewer = napari.current_viewer()

    # Hide the IDs points layer before any mutation to prevent vispy
    # from drawing stale GL buffers during intermediate repaints.
    ids_name = f"{layer.name}_IDs"
    ids_layer = viewer.layers[ids_name] if viewer and ids_name in viewer.layers else None
    if ids_layer is not None:
        ids_layer.visible = False

    # Reset mode before data mutation
    if reset_mode and hasattr(layer, 'mode'):
        layer.mode = 'pan_zoom'

    _mask_undo.begin(layer.data)
    layer.data[layer.data == label_id] = 0
    _mask_undo.end(layer.data)

    # Defer all visual updates to next event loop iteration so vispy
    # finishes any in-progress GL draw before buffers are modified.
    def _deferred_updates():
        layer.refresh()
        _remove_id_from_points_layer(layer, label_id)
        # Re-show the IDs layer after buffers are updated
        if ids_layer is not None:
            ids_layer.visible = True
    QTimer.singleShot(0, _deferred_updates)
    # Debounce save to disk
    _schedule_save(layer)

def _remove_id_from_points_layer(mask_layer, deleted_id):
    """Remove a single ID from the IDs points layer using napari's native API.

    Uses ``remove_selected()`` which handles vispy/GL buffer updates internally,
    avoiding the fragile remove-and-recreate pattern.
    """
    viewer = napari.current_viewer()
    if not viewer:
        return
    ids_name = f"{mask_layer.name}_IDs"
    if ids_name not in viewer.layers:
        return
    pts_layer = viewer.layers[ids_name]
    labels = np.asarray(pts_layer.properties.get('label', np.empty(0)))
    if len(labels) == 0:
        return
    to_remove = set(np.where(labels == deleted_id)[0])
    if not to_remove:
        return
    pts_layer.selected_data = to_remove
    pts_layer.remove_selected()

_save_timer = QTimer()
_save_timer.setSingleShot(True)
_save_pending_layer = None

def _schedule_save(layer):
    """Debounced save — writes mask to disk 500ms after the last edit."""
    global _save_pending_layer
    _save_pending_layer = layer
    _save_timer.start(500)

def _do_save():
    global _save_pending_layer
    layer = _save_pending_layer
    if layer is None:
        return
    out_dir = session.get_data("output_dir")
    if out_dir and layer.name:
        seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
        seg_dir.mkdir(exist_ok=True, parents=True)
        mask_path = seg_dir / f"{layer.name}.tif"
        tifffile.imwrite(mask_path, layer.data)
        session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
    _save_pending_layer = None

_save_timer.timeout.connect(_do_save)

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

def _make_divider():
    """Create a horizontal line divider using a native Qt QFrame."""
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet("background-color: #7a6b8a; border: none; margin: 8px 0px;")
    return line

def _make_section_header(text):
    """Create a left-aligned bold section header in light purple using plain QLabel."""
    label = _QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

from qtpy.QtWidgets import QPushButton, QLabel as _QLabel

# --- Merge Section ---
merge_label = _make_section_header("Merge Nuclei")
delete_btn = widgets.PushButton(text="Delete Source ID", tooltip="Delete the nucleus with the Source ID from the mask.")

# --- Paint Section ---
paint_label = _make_section_header("Paint New Mask")
paint_chk = widgets.CheckBox(text="Paint (New ID)", tooltip="Enable paint mode to draw a new mask region with the Source ID.")
from qtpy.QtWidgets import QHBoxLayout as _QHBoxLayout, QWidget as _QWidget

# Extrude ID — label + spinbox + button in one row
_extrude_spinbox = widgets.SpinBox(label="", min=1, max=99999, value=1, tooltip="Enter the nucleus label ID to extrude through all Z slices.")
_extrude_btn = QPushButton("Extrude (Fill Z)")
_extrude_btn.setToolTip("Extend the specified nucleus label through all Z slices.")
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
_delete_id_spinbox = widgets.SpinBox(label="", min=1, max=99999, value=1, tooltip="Enter the nucleus label ID to delete.")
_delete_id_btn = QPushButton("Delete ID")
_delete_id_btn.setToolTip("Delete the specified nucleus ID from the mask and ID layers.")
_delete_id_row = _QWidget()
_delete_id_row_layout = _QHBoxLayout(_delete_id_row)
_delete_id_row_layout.setContentsMargins(0, 2, 0, 2)
_delete_id_row_layout.setSpacing(4)
_delete_id_row_layout.addWidget(_QLabel("Nucleus ID:"))
_delete_id_row_layout.addWidget(_delete_id_spinbox.native, 1)
_delete_id_row_layout.addWidget(_delete_id_btn)

hover_chk = widgets.CheckBox(text="Hover Edit Mode (Red + 'C' to Del)", tooltip="Highlight nuclei under the cursor in red. Press C to delete the highlighted nucleus.")

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
from qtpy import sip
from qtpy.QtWidgets import QSizePolicy as _QSizePolicy
sip.delete(_old_layout)
_layout = _QVBoxLayout(_mask_editor_widget.native)
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
# Match the size policy of class-based Containers so the widget shrinks properly
# inside the nested QToolBox (QToolBox wraps pages in QScrollArea).
_mask_editor_widget.native.setSizePolicy(_QSizePolicy.Preferred, _QSizePolicy.Preferred)
_mask_editor_widget.native.setMinimumWidth(0)

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
        layer.mode = 'paint'
        if viewer:
            viewer.status = f"Painting with ID {_paint_id_spinbox.value}"
    else:
        layer.mode = 'pan_zoom'
        # Refresh IDs after painting so new/modified labels get their centroid label
        viewer = napari.current_viewer()
        if viewer:
            ids_name = f"{layer.name}_IDs"
            if ids_name in viewer.layers:
                QTimer.singleShot(100, lambda: viewer_helpers.add_or_update_label_ids(viewer, layer))

_paint_toggle_btn.clicked.connect(_on_paint_toggle)

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
    layer.mode = 'paint'
    _paint_toggle_btn.blockSignals(True)
    _paint_toggle_btn.setChecked(True)
    _paint_toggle_btn.blockSignals(False)
    if viewer:
        viewer.status = f"Painting new nucleus with ID {new_id}"

_paint_new_btn.clicked.connect(_on_paint_new)
_layout.addWidget(_paint_new_btn)

# Paint brush size row — independent from erase brush size
_paint_brush_slider = widgets.Slider(label="", min=1, max=40, value=10, tooltip="Brush size for painting.")
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
_refresh_ids_btn.setToolTip("Recompute centroids and refresh the ID labels overlay.")

def _on_refresh_ids(_checked=False):
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not viewer or not layer:
        return
    viewer_helpers.add_or_update_label_ids(viewer, layer)
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
        layer.mode = 'erase'
    else:
        layer.mode = 'pan_zoom'

_erase_toggle_btn.clicked.connect(_on_erase_toggle)

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

    _mask_undo.begin(layer.data)
    # Compute union XY footprint across all Z slices
    union_2d = np.any(layer.data == label_id, axis=0)
    # Only fill where the voxel is background (0) or already this label —
    # never overwrite other nuclei that overlap the XY footprint
    fill_mask = union_2d[np.newaxis, :, :] & ((layer.data == 0) | (layer.data == label_id))
    layer.data[fill_mask] = label_id
    _mask_undo.end(layer.data)

    # Refresh mask visual and recompute IDs (in-place update, no hide needed)
    layer.refresh()
    viewer_helpers.add_or_update_label_ids(viewer, layer)

    _schedule_save(layer)
    viewer.status = f"Extruded ID {label_id} through all Z slices."

@require_active_session("Please start or load a session before editing masks.")
def _on_delete():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    src = _mask_editor_widget.source_id.value
    if layer and src > 0:
        if np.sum(layer.data == src) > 0:
            logger.info("MASK EDIT: Delete ID %d on layer '%s'", src, layer.name)
            _mask_undo.begin(layer.data)
            layer.data = segmentation.delete_label(layer.data, src)
            _mask_undo.end(layer.data)
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

def _on_mask_undo():
    viewer = napari.current_viewer()
    layer = _mask_editor_widget.mask_layer.value
    if not layer:
        viewer.status = "No mask layer selected."
        return
    if _mask_undo.undo(layer.data):
        layer.refresh()
        logger.info("MASK EDIT: Undo on layer '%s' (%d remaining)", layer.name, len(_mask_undo))
        viewer.status = f"Undo ({len(_mask_undo)} remaining)."
    else:
        viewer.status = "Nothing to undo."

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
    if not np.any(layer.data == label_id):
        viewer.status = f"ID {label_id} not found in mask."
        return
    _delete_label_inplace(layer, label_id, reset_mode=True)
    logger.info("MASK EDIT: Deleted nucleus ID %d on layer '%s'", label_id, layer.name)
    viewer.status = f"Deleted Nucleus ID {label_id}."

# Connect signals after defining functions
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

def _sync_erase_from_layer_mode(event):
    """Keep paint/erase buttons and checkbox in sync when mode changes via layer controls."""
    global _was_painting
    mode = str(event.mode) if hasattr(event, 'mode') else str(event.value)
    mode_lower = mode.lower()
    is_erase = 'erase' in mode_lower
    is_paint = 'paint' in mode_lower

    # If we just left paint mode, refresh IDs so new labels get their text
    if _was_painting and not is_paint:
        layer = _mask_editor_widget.mask_layer.value
        if layer:
            viewer = napari.current_viewer()
            if viewer:
                ids_name = f"{layer.name}_IDs"
                if ids_name in viewer.layers:
                    QTimer.singleShot(100, lambda: viewer_helpers.add_or_update_label_ids(viewer, layer))
    _was_painting = is_paint

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

# Connect after a short delay so the viewer is fully initialized
def _connect_layer_selection_sync():
    viewer = napari.current_viewer()
    if viewer:
        viewer.layers.selection.events.changed.connect(_on_viewer_layer_selection_changed)

QTimer.singleShot(500, _connect_layer_selection_sync)

# --- Auto-saving for selected mask layer ---

# Store a reference to the layer and the callback to allow disconnection
_mask_editor_widget._current_layer = None
_mask_editor_widget._current_callback = None

def _create_save_callback(layer):
    """Factory to create a save callback for a specific layer."""
    _refresh_timer = QTimer()
    _refresh_timer.setSingleShot(True)

    def _refresh_ids():
        viewer = napari.current_viewer()
        if viewer:
            ids_name = f"{layer.name}_IDs"
            if ids_name in viewer.layers:
                viewer_helpers.add_or_update_label_ids(viewer, layer)

    _refresh_timer.timeout.connect(_refresh_ids)

    def _save_mask_data(event=None):
        out_dir = session.get_data("output_dir")
        if out_dir and layer and layer.name:
            seg_dir = Path(out_dir) / constants.SEGMENTATION_DIR
            seg_dir.mkdir(exist_ok=True, parents=True)
            mask_path = seg_dir / f"{layer.name}.tif"
            tifffile.imwrite(mask_path, layer.data)
            session.set_processed_file(layer.name, str(mask_path), layer_type='labels', metadata={'subtype': 'edited_mask'})
        # Debounced refresh: update existing ID layer 800 ms after the last edit
        _refresh_timer.start(800)

    return _save_mask_data

@_mask_editor_widget.mask_layer.changed.connect
def _on_mask_layer_changed(new_layer: "napari.layers.Labels"):
    """Disconnects the old listener and connects a new one to the selected layer."""
    old_layer = _mask_editor_widget._current_layer
    old_callback = _mask_editor_widget._current_callback

    if old_layer and old_callback and old_callback in old_layer.events.data.callbacks:
        old_layer.events.data.disconnect(old_callback)

    if new_layer:
        new_callback = _create_save_callback(new_layer)
        new_layer.events.data.connect(new_callback)
        _mask_editor_widget._current_layer = new_layer
        _mask_editor_widget._current_callback = new_callback
    else:
        _mask_editor_widget._current_layer = None
        _mask_editor_widget._current_callback = None

# --- Public API ---
# Expose the magicgui widget directly as mask_editor_widget (no wrapper Container).
# This gives the same single-level nesting as dapi_segmentation_widget and
# new_session_widget, which prevents right-edge clipping on panel resize.
mask_editor_widget = _mask_editor_widget
mask_editor_widget._mask_editor_widget = _mask_editor_widget
