import napari
import numpy as np
from collections import deque
from magicgui import magicgui, widgets
from qtpy.QtWidgets import QFrame, QPushButton

from ..decorators import require_active_session, error_handler
from .. import style
from ... import constants


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


def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {style.COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line


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

        idx = np.round(coords).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, mask_data.shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, mask_data.shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, mask_data.shape[2] - 1)

        inside = mask_data[idx[:, 0], idx[:, 1], idx[:, 2]] > 0
        n_before = len(coords)
        layer.data = coords[inside]
        if layer.features is not None and len(layer.features) == n_before:
            layer.features = layer.features.iloc[inside].reset_index(drop=True)
        total_removed += n_before - inside.sum()

    viewer.status = f"Removed {total_removed} extranuclear puncta across {len(target_layers)} layer(s)."


def _refresh_channel_choices():
    """Update the channels dropdown based on current viewer layers."""
    viewer = napari.current_viewer()
    choices = _get_puncta_channel_choices(viewer)
    _refilter_widget.channels.choices = choices
    _refilter_widget.channels.value = choices[0] if choices else "All (R1 + R2)"


# --- UI Helpers ---
def _make_section_header(text):
    from qtpy.QtWidgets import QLabel
    label = QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

def _make_section_desc(text):
    from qtpy.QtWidgets import QLabel
    desc = QLabel(text)
    desc.setWordWrap(True)
    desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
    return desc

def _make_spacer():
    from qtpy.QtWidgets import QWidget as _W
    s = _W()
    s.setFixedHeight(20)
    return s

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
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox
_refilter_widget.native.setMinimumWidth(0)
for child in _refilter_widget.native.findChildren((QAbstractSpinBox, QComboBox)):
    child.setMinimumWidth(0)
