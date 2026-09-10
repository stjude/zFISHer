import napari
import numpy as np
from collections import deque
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QFrame, QPushButton

from ..decorators import require_active_session, error_handler
from .. import style, viewer_helpers
from ... import constants
from ...core import puncta
from ._shared import (
    make_divider as _make_divider,
    make_section_header as _make_section_header,
    make_section_desc as _make_section_desc,
    make_spacer as _make_spacer,
)


class _RefilterUndoStack:
    """Stores snapshots of all affected layers before a refilter operation."""
    def __init__(self, maxlen=5):
        self._stack = deque(maxlen=maxlen)

    def push(self, layers):
        """Save (data, features) for each layer before refilter."""
        snapshot = []
        for layer in layers:
            feats = layer.features.copy() if layer.features is not None and len(layer.features) > 0 else None
            snapshot.append((layer, layer.data.copy(), feats))
        self._stack.append(snapshot)

    def undo(self):
        if not self._stack:
            return False
        snapshot = self._stack.pop()
        for layer, data, features in snapshot:
            layer.data = data
            if features is not None:
                layer.features = features
        return True

    def clear(self):
        self._stack.clear()

    def __len__(self):
        return len(self._stack)


_refilter_undo = _RefilterUndoStack()


def reset_refilter_state():
    """Clear all module-level state. Called on session reset."""
    _refilter_undo.clear()


def _get_puncta_channel_choices(viewer):
    """Build the channel filter choices from current puncta layers."""
    if viewer is None:
        return ["All (R1 + R2)"]

    puncta_layers = [
        l for l in viewer.layers
        if isinstance(l, napari.layers.Points) and constants.PUNCTA_SUFFIX in l.name
    ]

    # Filter to only aligned/warped puncta layers
    filtered = []
    for l in puncta_layers:
        name_upper = l.name.upper()
        if constants.ALIGNED_PREFIX.upper() in name_upper or constants.WARPED_PREFIX.upper() in name_upper:
            filtered.append(l)

    r1_layers = [l for l in filtered if "R1" in l.name.upper()]
    r2_layers = [l for l in filtered if "R2" in l.name.upper()]

    choices = ["All (R1 + R2)"]
    if r1_layers:
        choices.append("All R1")
    if r2_layers:
        choices.append("All R2")
    for l in filtered:
        choices.append(l.name)

    return choices


def _resolve_puncta_layers(viewer, channel_choice):
    """Return the list of puncta layers matching the channel choice."""
    puncta_layers = [
        l for l in viewer.layers
        if isinstance(l, napari.layers.Points) and constants.PUNCTA_SUFFIX in l.name
    ]

    # Filter to only aligned/warped
    filtered = [
        l for l in puncta_layers
        if constants.ALIGNED_PREFIX.upper() in l.name.upper()
        or constants.WARPED_PREFIX.upper() in l.name.upper()
    ]

    if channel_choice == "All (R1 + R2)":
        return filtered
    elif channel_choice == "All R1":
        return [l for l in filtered if "R1" in l.name.upper()]
    elif channel_choice == "All R2":
        return [l for l in filtered if "R2" in l.name.upper()]
    else:
        return [l for l in filtered if l.name == channel_choice]


@magicgui(
    call_button="Filter Puncta by Mask",
    mask_layer={"label": "Filter Mask", "tooltip": "Mask layer to filter against. Puncta outside this mask are removed."},
    channels={"label": "Channels", "choices": ["All (R1 + R2)"], "tooltip": "Select which puncta layers to re-filter against the mask. 'All (R1 + R2)' applies to all aligned/warped puncta."},
    auto_call=False,
)
@require_active_session()
@error_handler("Re-filter Failed")
def _refilter_widget(
    mask_layer: "napari.layers.Labels",
    channels: str = "All (R1 + R2)",
):
    """Remove puncta points that fall outside the selected mask."""
    viewer = napari.current_viewer()
    if mask_layer is None:
        viewer.status = "No mask layer selected."
        return

    mask_data = mask_layer.data
    target_layers = _resolve_puncta_layers(viewer, channels)

    if not target_layers:
        viewer.status = "No matching puncta layers found."
        return

    # Snapshot all affected layers before modification
    _refilter_undo.push(target_layers)

    total_removed = 0
    for layer in target_layers:
        coords = layer.data
        if len(coords) == 0:
            continue

        # Convert puncta coords -> world -> mask voxels so the lookup respects
        # both layers' scale/translate (a punctum and the mask need not share a
        # coordinate frame). Previously this indexed the mask directly with the
        # puncta data coords, which silently removed the wrong puncta whenever
        # the scales/translates differed.
        labels = puncta.lookup_label_ids(
            coords, layer.scale, layer.translate,
            mask_data, mask_layer.scale, mask_layer.translate,
        )
        inside = labels > 0
        n_before = len(coords)
        viewer_helpers.set_points_data(layer, coords[inside])
        if layer.features is not None and len(layer.features) == n_before:
            layer.features = layer.features.iloc[inside].reset_index(drop=True)
        total_removed += n_before - inside.sum()

    viewer.status = f"Removed {total_removed} extranuclear puncta across {len(target_layers)} layer(s)."


def _get_mask_layer_choices(widget=None):
    """Return only aligned, warped, or consensus Labels layers."""
    viewer = napari.current_viewer()
    if viewer is None:
        return []
    return [
        l for l in viewer.layers
        if isinstance(l, napari.layers.Labels) and (
            l.name.upper().startswith(constants.ALIGNED_PREFIX.upper())
            or l.name.upper().startswith(constants.WARPED_PREFIX.upper())
            or l.name == constants.CONSENSUS_MASKS_NAME
        )
    ]


def _refresh_channel_choices():
    """Update the channels and mask dropdowns based on current viewer layers."""
    viewer = napari.current_viewer()
    choices = _get_puncta_channel_choices(viewer)
    _refilter_widget.channels.choices = choices
    _refilter_widget.channels.value = choices[0] if choices else "All (R1 + R2)"

    # Filter mask dropdown to only show aligned/warped/consensus masks
    valid_masks = _get_mask_layer_choices()
    if valid_masks:
        _refilter_widget.mask_layer.choices = valid_masks
        if _refilter_widget.mask_layer.value not in valid_masks:
            _refilter_widget.mask_layer.value = valid_masks[0]


# --- UI Wrapper ---
class _RefilterWidgetContainer(widgets.Container):
    def reset_choices(self):
        _refilter_widget.reset_choices()
        _refresh_channel_choices()

refilter_puncta_widget = _RefilterWidgetContainer(labels=False)
refilter_puncta_widget._refilter_widget = _refilter_widget

header = widgets.Label(value="Puncta Cleanup")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Remove puncta that fall outside a nuclei mask.</i>")
info.native.setObjectName("widgetInfo")
from qtpy.QtWidgets import QSizePolicy as _QSP
info.native.setSizePolicy(_QSP.Expanding, _QSP.Preferred)

_layout = refilter_puncta_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())

# Insert section headers into inner form
_inner = _refilter_widget.native.layout()
_inner.insertWidget(0, _make_section_header("Filter Settings"))
_inner.insertWidget(1, _make_section_desc("Select a nuclei mask and channel(s) to remove extranuclear puncta."))
_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

_layout.addWidget(_refilter_widget.native)

# Undo button
_refilter_undo_btn = QPushButton("Undo")
_refilter_undo_btn.setToolTip("Undo the last refilter operation, restoring removed puncta.")

def _on_refilter_undo(_checked=False):
    viewer = napari.current_viewer()
    if _refilter_undo.undo():
        if viewer:
            viewer.status = f"Refilter undone ({len(_refilter_undo)} remaining)."
    else:
        if viewer:
            viewer.status = "Nothing to undo."

_refilter_undo_btn.clicked.connect(_on_refilter_undo)
_layout.addWidget(_refilter_undo_btn)

_layout.addStretch(1)

# Ensure widgets shrink with panel
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox, QLabel
_refilter_widget.native.setMinimumWidth(0)
for child in _refilter_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
for child in _refilter_widget.native.findChildren(QAbstractSpinBox) + _refilter_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)
