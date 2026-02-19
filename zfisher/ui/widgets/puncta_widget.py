import napari
import numpy as np
from magicgui import magicgui, widgets
from pathlib import Path

from ...core import session, puncta  
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Detect Puncta",
    layout="vertical",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Consensus)"},
    # ADDED: "Difference of Gaussian" to choices
    method={
        "label": "Algorithm", 
        "choices": ["Local Maxima", "Laplacian of Gaussian", "Difference of Gaussian"]
    },
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
    """Refactored to delegate all logic to core.puncta."""
    viewer = napari.current_viewer()

    if image_layer is None:
        return
        
    with popups.ProgressDialog(viewer.window._qt_window, f"Detecting in {image_layer.name}...") as dialog:
        # 1. Package parameters
        params = {
            'threshold_rel': threshold, 
            'min_distance': min_distance, 
            'sigma': sigma, 
            'method': method,
            'use_tophat': use_tophat,
            'tophat_radius': tophat_radius
        }

        # 2. Determine Output Path
        output_dir = session.get_data("output_dir")
        csv_path = None
        if output_dir:
            csv_path = Path(output_dir) / constants.REPORTS_DIR / f"{image_layer.name}_puncta.csv"

        # 3. Call Core Orchestrator
        # This handles detection, nucleus mapping, and automated session logging.
        mask_data = nuclei_layer.data if nuclei_layer else None
        results = puncta.process_puncta_detection(
            image_data=image_layer.data,
            mask_data=mask_data,
            params=params,
            output_path=csv_path
        )
        
        # 4. Update Viewer
        # Pass ZYX coordinates (first 3 columns) to create/update the points layer
        viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, results[:, :3])

        viewer.status = f"Found {len(results)} spots in {image_layer.name}."