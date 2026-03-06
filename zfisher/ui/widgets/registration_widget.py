import napari
from magicgui import magicgui, widgets

from ...core import session, registration #
from .. import popups
from ..decorators import require_active_session, error_handler
from ._shared import make_header_divider

@magicgui(
    call_button="Calculate Shift (RANSAC)",
    r1_points={"label": "R1 Centroids"},
    r2_points={"label": "R2 Centroids"}
)
@require_active_session("Please start or load a session before running registration.")
@error_handler("Registration Failed")
def _registration_widget(
    r1_points: "napari.layers.Points",
    r2_points: "napari.layers.Points"
):
    """Calculates the XYZ shift between two point clouds via the core orchestrator."""
    viewer = napari.current_viewer()

    if r1_points is None or r2_points is None:
        viewer.status = "Please select both centroid layers."
        return

    # Use a progress dialog to wrap the core call
    with popups.ProgressDialog(viewer.window._qt_window, title="Calculating Registration (RANSAC)...") as dialog:
        
        # Call the Refactored Core Orchestrator
        # We pass .data (NumPy) so the core remains headless-compatible
        shift, rmsd = registration.calculate_session_registration(
            r1_points.data, 
            r2_points.data, 
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

# --- UI Wrapper ---
class _RegistrationContainer(widgets.Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    def reset_choices(self):
        _registration_widget.reset_choices()

registration_widget = _RegistrationContainer(labels=False)
registration_widget._registration_widget = _registration_widget
header = widgets.Label(value="Registration")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Calculates shift between rounds.</i>")
info.native.setObjectName("widgetInfo")
registration_widget.extend([header, info, make_header_divider(), _registration_widget])