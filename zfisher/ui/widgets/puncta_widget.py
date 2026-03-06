import napari
from pathlib import Path
from magicgui import magicgui, widgets
from ...core import session, puncta
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_header_divider

@magicgui(
    call_button="Detect Puncta",
    layout="vertical",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Consensus)"},
    nuclei_only={"label": "Nuclei Only (discard extranuclear puncta)", "value": True},
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
    nuclei_only: bool = True,
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
            'tophat_radius': tophat_radius,
            'nuclei_only': nuclei_only
        }
        
        out_dir = session.get_data("output_dir")
        csv_path = Path(out_dir) / constants.REPORTS_DIR / f"{image_layer.name}_puncta.csv" if out_dir else None

        # Pass voxel scale for automated z_scale fallback
        results = puncta.process_puncta_detection(
            image_layer.data,
            mask_data=nuclei_layer.data if nuclei_layer else None,
            voxels=getattr(image_layer, 'scale', (1,1,1)),
            params=params,
            output_path=csv_path,
            progress_callback=lambda p, t: dialog.update_progress(p, t)
        )
        
        viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, results)
        viewer.status = f"Found {len(results)} spots in {image_layer.name}."

# --- Automated UI Event Listeners ---

_METHOD_INFO = {
    "Local Maxima": {
        "params": {"min_distance": True,  "z_scale": False, "sigma": True},
        "desc": (
            "Finds intensity peaks separated by a minimum distance. "
            "Fast and intuitive. Best for well-separated, bright puncta with low background. "
            "Struggles with crowded fields where spots overlap. "
            "Use <i>Sigma</i> to pre-blur noisy images before peak finding."
        ),
    },
    "Laplacian of Gaussian": {
        "params": {"min_distance": False, "z_scale": True,  "sigma": True},
        "desc": (
            "Scale-space blob detector that matches "
            "spot size to a Gaussian kernel. Accurate for varying spot sizes and handles "
            "anisotropic Z-spacing well. Slower than other methods, especially on large volumes. "
            "Best when spots vary in size or Z-resolution differs from XY."
        ),
    },
    "Difference of Gaussian": {
        "params": {"min_distance": False, "z_scale": True,  "sigma": True},
        "desc": (
            "Approximates LoG using bandpass filtering. "
            "Nearly as accurate as LoG but significantly faster. Good default for most FISH data. "
            "Handles anisotropic Z well. Slightly less precise than LoG for spots with highly "
            "variable sizes."
        ),
    },
    "Radial Symmetry": {
        "params": {"min_distance": False, "z_scale": False, "sigma": False},
        "desc": (
            "Finds all local maxima with minimal filtering "
            "(min distance = 1px). Designed for high-density transcript fields where spots "
            "are tightly packed. Very sensitive — will detect faint spots others miss, but may "
            "over-count in noisy images. Pair with <i>Deconvolve</i> or <i>Top-hat</i> to reduce false positives."
        ),
    },
}

from qtpy.QtWidgets import QLabel
_method_desc_qlabel = QLabel("")
_method_desc_qlabel.setWordWrap(True)

def _update_method_ui(method: str):
    info = _METHOD_INFO.get(method, {})
    for param_name, visible in info.get("params", {}).items():
        getattr(_puncta_widget, param_name).visible = visible
    _method_desc_qlabel.setText(f"<b>{method}</b><br>{info.get('desc', '')}")

@_puncta_widget.method.changed.connect
def _on_method_change(method: str):
    _update_method_ui(method)

# Set initial state
_update_method_ui(_puncta_widget.method.value)

@_puncta_widget.image_layer.changed.connect
def _on_image_change(new_layer: "napari.layers.Image"):
    """
    Automatically detects voxel metadata and sets the Z-Anisotropy slider.
    """
    if new_layer is not None:
        scale = getattr(new_layer, 'scale', (1, 1, 1))
        if len(scale) == 3 and scale[2] != 0:
            auto_z_scale = scale[0] / scale[2]
            _puncta_widget.z_scale.value = round(float(auto_z_scale), 2)
            napari.current_viewer().status = f"Auto-configured Z-Anisotropy: {round(auto_z_scale, 2)}"

# --- UI Wrapper (native layout like colocalization_widget) ---
from qtpy.QtWidgets import QFrame
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

puncta_widget = widgets.Container(labels=False)
header = widgets.Label(value="Puncta Detection")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Algorithmic detection of puncta.</i>")
info.native.setObjectName("widgetInfo")

# Insert section headers into the magicgui form's internal layout
# Original order: image_layer(0), nuclei_layer(1), nuclei_only(2), method(3),
#   use_decon(4), decon_iter(5), threshold(6), min_distance(7), sigma(8), z_scale(9),
#   use_tophat(10), tophat_radius(11), call_button(12)
_inner = _puncta_widget.native.layout()

# "Algorithm" header + description after nuclei_only (index 3 -> before method)
_algo_header = widgets.Label(value="<b>Algorithm:</b>")
_inner.insertWidget(3, _make_divider())
_inner.insertWidget(4, _algo_header.native)
# method is now at 5, description after it at 6
_inner.insertWidget(6, _method_desc_qlabel)

# "Preprocessing" header before use_decon (was 4, now shifted to 7)
_preproc_header = widgets.Label(value="<b>Preprocessing:</b>")
_inner.insertWidget(7, _make_divider())
_inner.insertWidget(8, _preproc_header.native)

# "Detection Parameters" header before threshold (was 6, now shifted to 11)
_params_header = widgets.Label(value="<b>Detection Parameters:</b>")
_inner.insertWidget(11, _make_divider())
_inner.insertWidget(12, _params_header.native)

# "Background Subtraction" header before use_tophat (was 10, now shifted to 17)
_bg_header = widgets.Label(value="<b>Background Subtraction:</b>")
_inner.insertWidget(17, _make_divider())
_inner.insertWidget(18, _bg_header.native)

# Outer layout: whole form as one block
_layout = puncta_widget.native.layout()
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_puncta_widget.native)