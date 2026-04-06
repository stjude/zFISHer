import napari
from magicgui import magicgui, widgets

from ...core import session, registration #
from .. import popups
from ..decorators import require_active_session, error_handler
from ._shared import make_header_divider

@magicgui(
    call_button="Calculate Alignment",
    r1_points={"label": "R1 Centroids", "tooltip": "Centroid points layer from Round 1 nuclei segmentation."},
    r2_points={"label": "R2 Centroids", "tooltip": "Centroid points layer from Round 2 nuclei segmentation."},
    max_distance={"label": "Max Pair Distance, px (0=auto)", "value": 0, "min": 0, "max": 100, "tooltip": "Maximum distance in pixels for matching centroid pairs between rounds. 0 = auto-detect."},
)
@require_active_session("Please start or load a session before running registration.")
@error_handler("Registration Failed")
def _registration_widget(
    r1_points: "napari.layers.Points",
    r2_points: "napari.layers.Points",
    max_distance: int = 0,
):
    """Calculates the XYZ shift between two point clouds via the core orchestrator."""
    viewer = napari.current_viewer()

    if r1_points is None or r2_points is None:
        viewer.status = "Please select both centroid layers."
        return

    # Warn if shift already exists
    existing_shift = session.get_data("shift")
    if existing_shift is not None:
        if not popups.show_yes_no_popup(
            viewer.window._qt_window,
            "Recalculate Alignment?",
            "An alignment has already been calculated.\n\n"
            "Recalculating will overwrite the existing shift. "
            "Any downstream results (canvas, warping, consensus) will need to be regenerated.\n\n"
            "Continue?",
        ):
            return

    # Use a progress dialog to wrap the core call
    with popups.ProgressDialog(viewer.window._qt_window, title="Calculating Registration (RANSAC)...") as dialog:
        
        # Call the Refactored Core Orchestrator
        # We pass .data (NumPy) so the core remains headless-compatible
        shift, rmsd = registration.calculate_session_registration(
            r1_points.data,
            r2_points.data,
            max_distance=max_distance,
            progress_callback=dialog.update_progress
        )
        
        # UI Feedback
        msg = f"Calculated Shift: Z={shift[0]:.2f}, Y={shift[1]:.2f}, X={shift[2]:.2f}"
        rmsd_msg = f"RMSD: {rmsd:.4f} px"
        
        viewer.status = f"{msg}, {rmsd_msg}"
        _registration_widget.result_label.value = f"<b>{msg}</b><br>{rmsd_msg}"
        
        dialog.update_progress(100, "Done.")

# Persistence for result display
_registration_widget.result_label = widgets.Label(value="")
_registration_widget.append(_registration_widget.result_label)

# --- UI Helpers ---
from qtpy.QtWidgets import QLabel, QFrame, QSizePolicy
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

def _make_section_header(text):
    label = QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

def _make_section_desc(text):
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
class _RegistrationContainer(widgets.Container):
    def reset_choices(self):
        _registration_widget.reset_choices()

registration_widget = _RegistrationContainer(labels=False)
registration_widget._registration_widget = _registration_widget
header = widgets.Label(value="Registration")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Calculates spatial alignment between Round 1 and Round 2.</i>")
info.native.setObjectName("widgetInfo")
info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

_layout = registration_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())

# Insert section headers into inner form
_inner = _registration_widget.native.layout()
_centroid_header = _make_section_header("Nuclei Centers")
_centroid_desc = _make_section_desc("Select the R1 and R2 centroid point layers for alignment calculation.")
_inner.insertWidget(0, _centroid_header)
_inner.insertWidget(1, _centroid_desc)
_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.insertWidget(_inner.count() - 1, _make_divider())
_params_header = _make_section_header("Parameters")
_params_desc = _make_section_desc("Adjust registration parameters. Use 0 for auto-detection.")
_inner.insertWidget(_inner.count() - 1, _params_header)
_inner.insertWidget(_inner.count() - 1, _params_desc)
_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

_registration_widget.native.setMinimumWidth(0)
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox, QLabel
for child in _registration_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
for child in _registration_widget.native.findChildren(QAbstractSpinBox) + _registration_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)

_layout.addWidget(_registration_widget.native)
_layout.addStretch(1)