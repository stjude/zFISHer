import napari
import numpy as np
from magicgui import magicgui, widgets
from pathlib import Path

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.segmentation import detect_spots_3d
from ... import constants

@magicgui(
    call_button="Detect Puncta",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Optional)"},
    method={"label": "Algorithm", "choices": ["Local Maxima", "Laplacian of Gaussian"]},
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01, "value": constants.PUNCTA_THRESHOLD_REL},
    min_distance={"label": "Min Distance (px)", "min": 1, "max": 20, "step": 1, "value": constants.PUNCTA_MIN_DISTANCE},
    sigma={"label": "Spot Radius (Sigma)", "min": 0.0, "max": 5.0, "step": 0.1, "value": constants.PUNCTA_SIGMA},
    use_tophat={"label": "Subtract Background (Top-hat)"},
    tophat_radius={"label": "Top-hat Radius (px)", "min": 1, "max": 50, "value": constants.PUNCTA_TOPHAT_RADIUS}
)
@require_active_session("Please start or load a session before detecting puncta.")
@error_handler("Puncta Detection Failed")
def puncta_widget(
    image_layer: "napari.layers.Image",
    nuclei_layer: "napari.layers.Labels",
    method: str = "Local Maxima",
    threshold: float = constants.PUNCTA_THRESHOLD_REL,
    min_distance: int = constants.PUNCTA_MIN_DISTANCE,
    sigma: float = constants.PUNCTA_SIGMA,
    use_tophat: bool = False,
    tophat_radius: int = constants.PUNCTA_TOPHAT_RADIUS
):
    """Detects spots in the selected image layer."""
    viewer = napari.current_viewer()

    if image_layer is None:
        return
        
    with popups.ProgressDialog(viewer.window._qt_window, f"Detecting Puncta in {image_layer.name}...") as dialog:
        viewer.status = f"Detecting spots in {image_layer.name}..."
        # Run detection
        coords = detect_spots_3d(
            image_layer.data, 
            threshold_rel=threshold, 
            min_distance=min_distance, 
            sigma=sigma, 
            method=method,
            use_tophat=use_tophat,
            tophat_radius=tophat_radius
        )
        
        # Pass the results to the UI helper to handle layer creation/update
        viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, coords)

        msg = f"Found {len(coords) if coords is not None else 0} spots."
        print(msg)
        viewer.status = msg

# Add editing tools to the Puncta Widget
edit_chk = widgets.CheckBox(text="Edit Mode")
clear_btn = widgets.PushButton(text="Clear All Puncta")
puncta_widget.append(widgets.Label(value="<b>Editing Tools:</b>"))
puncta_widget.append(edit_chk)
puncta_widget.append(clear_btn)

@require_active_session("Please start or load a session before editing puncta.")
def _on_edit_puncta(state: bool):
    if not session.get_data("output_dir"): # Check again in case session was closed
        edit_chk.value = False
        return
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            layer = viewer.layers[p_name]
            viewer.layers.selection.active = layer
            if state:
                layer.mode = 'select'
                viewer.status = f"Editing {p_name}. Select points and press Backspace/Delete to remove."
            else:
                layer.mode = 'pan_zoom'
                viewer.status = f"Stopped editing {p_name}."
        else:
            viewer.status = f"Layer {p_name} not found. Run detection first."
            edit_chk.value = False

@require_active_session("Please start or load a session before clearing puncta.")
def _on_clear_puncta():
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            viewer.layers[p_name].data = np.empty((0, 3))
            viewer.status = f"Cleared all points in {p_name}."

edit_chk.changed.connect(_on_edit_puncta)
clear_btn.clicked.connect(_on_clear_puncta)
