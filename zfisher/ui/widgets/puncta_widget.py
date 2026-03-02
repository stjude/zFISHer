import napari
from pathlib import Path
from magicgui import magicgui
from ...core import session, puncta
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants

import napari
from pathlib import Path
from magicgui import magicgui, widgets
from ...core import session, puncta
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Detect Puncta",
    layout="vertical",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Consensus)"},
    method={
        "label": "Algorithm", 
        "choices": ["Local Maxima", "Laplacian of Gaussian", "Difference of Gaussian", "Radial Symmetry"]
    },
    use_decon={"label": "Deconvolve (Crowded Fields)"},
    decon_iter={"label": "Iterations", "min": 1, "max": 50, "value": 10},
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01, "value": constants.PUNCTA_THRESHOLD_REL},
    min_distance={"label": "Min Distance (px)", "min": 1, "max": 20, "value": constants.PUNCTA_MIN_DISTANCE},
    sigma={"label": "Spot Radius (Sigma)", "min": 0.0, "max": 5.0, "step": 0.1, "value": constants.PUNCTA_SIGMA},
    z_scale={"label": "Z-Anisotropy Scale", "min": 0.01, "max": 20.0, "step": 0.01, "value": 1.0},
    use_tophat={"label": "Subtract Background (Top-hat)"},
    tophat_radius={"label": "Top-hat Radius (px)", "min": 1, "max": 50, "value": constants.PUNCTA_TOPHAT_RADIUS}
)
@require_active_session()
@error_handler("Puncta Detection Failed")
def _puncta_widget(
    image_layer: "napari.layers.Image", 
    nuclei_layer: "napari.layers.Labels", 
    method: str = "Local Maxima", 
    use_decon: bool = False, 
    decon_iter: int = 10,
    threshold: float = constants.PUNCTA_THRESHOLD_REL,
    min_distance: int = constants.PUNCTA_MIN_DISTANCE,
    sigma: float = constants.PUNCTA_SIGMA,
    z_scale: float = 1.0,
    use_tophat: bool = False,
    tophat_radius: int = constants.PUNCTA_TOPHAT_RADIUS
):
    """User interface for high-density puncta quantification."""
    viewer = napari.current_viewer()
    if not image_layer: 
        return

    with popups.ProgressDialog(viewer.window._qt_window, f"Processing {image_layer.name}...") as dialog:
        # Package parameters for the core orchestrator
        params = {
            'method': method, 
            'use_decon': use_decon, 
            'decon_iter': decon_iter,
            'threshold_rel': threshold, 
            'z_scale': z_scale,
            'min_distance': min_distance, 
            'sigma': sigma,
            'use_tophat': use_tophat, 
            'tophat_radius': tophat_radius
        }
        
        out_dir = session.get_data("output_dir")
        csv_path = Path(out_dir) / constants.REPORTS_DIR / f"{image_layer.name}_puncta.csv" if out_dir else None

        # Pass voxel scale for automated z_scale fallback
        results = puncta.process_puncta_detection(
            image_layer.data, 
            mask_data=nuclei_layer.data if nuclei_layer else None,
            voxels=getattr(image_layer, 'scale', (1,1,1)), 
            params=params, 
            output_path=csv_path
        )
        
        viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, results)
        viewer.status = f"Found {len(results)} spots in {image_layer.name}."

# --- Automated UI Event Listeners ---

@_puncta_widget.image_layer.changed.connect
def _on_image_change(new_layer: "napari.layers.Image"):
    """
    Automatically detects voxel metadata and sets the Z-Anisotropy slider.
    This ensures the 'Algorithmic' Hub is pre-configured for your specific .nd2 files.
    """
    if new_layer is not None:
        # napari scale is (z, y, x)
        scale = getattr(new_layer, 'scale', (1, 1, 1))
        
        # Calculate dz/dx ratio
        if len(scale) == 3 and scale[2] != 0:
            auto_z_scale = scale[0] / scale[2]
            
            # Update the slider value in the UI
            _puncta_widget.z_scale.value = round(float(auto_z_scale), 2)
            napari.current_viewer().status = f"Auto-configured Z-Anisotropy: {round(auto_z_scale, 2)}"

# --- UI Wrapper ---
puncta_widget = widgets.Container(labels=False)
header = widgets.Label(value="Puncta Detection")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Algorithmic detection of puncta.</i>")
puncta_widget.extend([header, info, _puncta_widget])