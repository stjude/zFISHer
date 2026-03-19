import logging
import napari
from pathlib import Path
from magicgui import magicgui, widgets
from ...core import session, puncta
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_header_divider

logger = logging.getLogger(__name__)

@magicgui(
    call_button="Detect Puncta",
    layout="vertical",
    image_layer={"label": "Target Channel", "tooltip": "The fluorescent channel image to detect puncta in."},
    nuclei_layer={"label": "Nuclei Masks", "tooltip": "Nuclei mask used to filter puncta. Only puncta inside nuclei are kept."},
    nuclei_only={"label": "Nuclei Only (discard extranuclear puncta)", "value": True, "tooltip": "If checked, discard puncta that fall outside the nuclei mask."},
    method={
        "label": "Algorithm",
        "choices": ["Local Maxima", "Laplacian of Gaussian", "Difference of Gaussian", "Radial Symmetry"],
        "tooltip": "Detection algorithm. Local Maxima is fastest. LoG and DoG detect blob-like structures. Radial Symmetry finds symmetric bright spots."
    },
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01, "value": constants.PUNCTA_THRESHOLD_REL, "tooltip": "Relative intensity threshold (0-1). Lower values detect dimmer spots but may increase false positives."},
    min_distance={"label": "Min Distance (px)", "min": 1, "max": 20, "value": constants.PUNCTA_MIN_DISTANCE, "tooltip": "Minimum separation between detected puncta in pixels. Prevents double-counting nearby spots."},
    sigma={"label": "Spot Radius (Sigma)", "min": 0.0, "max": 5.0, "step": 0.1, "value": constants.PUNCTA_SIGMA, "tooltip": "Gaussian sigma for spot detection. Match to the approximate radius of your puncta in pixels."},
    z_scale={"label": "Z-Anisotropy Scale", "min": 0.01, "max": 20.0, "step": 0.01, "value": 1.0, "tooltip": "Z-anisotropy correction factor. Adjusts for the difference between Z-step and XY pixel size."},
    use_tophat={"label": "Subtract Background (Top-hat)", "tooltip": "Apply white top-hat filter to subtract uneven background before detection."},
    tophat_radius={"label": "Top-hat Radius (px)", "min": 1, "max": 50, "value": constants.PUNCTA_TOPHAT_RADIUS, "tooltip": "Radius of the top-hat structuring element in pixels. Should be larger than puncta diameter."}
)
@require_active_session()
@error_handler("Puncta Detection Failed")
def _puncta_widget(
    image_layer: "napari.layers.Image",
    nuclei_layer: "napari.layers.Labels",
    nuclei_only: bool = True,
    method: str = "Local Maxima",
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

    logger.info("Puncta widget settings: layer=%s, nuclei=%s, nuclei_only=%s, method=%s, "
                "threshold=%.3f, min_distance=%d, sigma=%.1f, z_scale=%.2f, tophat=%s/%d",
                image_layer.name, nuclei_layer.name if nuclei_layer else None,
                nuclei_only, method, threshold, min_distance, sigma, z_scale, use_tophat, tophat_radius)

    with popups.ProgressDialog(viewer.window._qt_window, f"Processing {image_layer.name}...") as dialog:
        # Package parameters for the core orchestrator
        params = {
            'method': method,
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

        # Persist the detection parameters for this channel in the session
        puncta_params_all = session.get_data("puncta_params", default={})
        puncta_params_all[image_layer.name] = {
            'algorithm': method,
            'sensitivity': threshold,
            'min_distance': min_distance,
            'sigma': sigma,
            'z_scale': z_scale,
            'nuclei_only': nuclei_only,
            'tophat': use_tophat,
            'tophat_radius': tophat_radius,
            'num_puncta': len(results),
        }
        session.update_data("puncta_params", puncta_params_all)

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
            "over-count in noisy images. Pair with <i>Top-hat</i> to reduce false positives."
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
    Also auto-selects the appropriate nuclei mask based on the channel's round.
    """
    if new_layer is not None:
        scale = getattr(new_layer, 'scale', (1, 1, 1))
        if len(scale) == 3 and scale[2] != 0:
            auto_z_scale = scale[0] / scale[2]
            _puncta_widget.z_scale.value = round(float(auto_z_scale), 2)
            napari.current_viewer().status = f"Auto-configured Z-Anisotropy: {round(auto_z_scale, 2)}"

        # Auto-select the matching nuclei mask layer
        viewer = napari.current_viewer()
        if viewer is None:
            return
        name_upper = new_layer.name.upper()
        is_aligned = constants.ALIGNED_PREFIX.upper() in name_upper or constants.WARPED_PREFIX.upper() in name_upper

        best_mask = None
        if is_aligned:
            # Aligned/Warped channels → use consensus mask
            best_mask = next(
                (l for l in viewer.layers
                 if isinstance(l, napari.layers.Labels)
                 and constants.CONSENSUS_MASKS_NAME.upper() in l.name.upper()),
                None
            )
        else:
            # Raw channels → match per-round nuclei mask
            if "R1" in name_upper:
                target = "R1"
            elif "R2" in name_upper:
                target = "R2"
            else:
                target = None
            if target:
                best_mask = next(
                    (l for l in viewer.layers
                     if isinstance(l, napari.layers.Labels)
                     and target in l.name.upper()
                     and session.get_nuclear_channel().upper() in l.name.upper()
                     and constants.MASKS_SUFFIX.upper() in l.name.upper()
                     and constants.ALIGNED_PREFIX.upper() not in l.name.upper()
                     and constants.WARPED_PREFIX.upper() not in l.name.upper()),
                    None
                )

        if best_mask is not None:
            try:
                _puncta_widget.nuclei_layer.value = best_mask
            except (ValueError, AttributeError):
                pass

# --- UI Wrapper (native layout like colocalization_widget) ---
from qtpy.QtWidgets import QFrame
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

class _PunctaWidgetContainer(widgets.Container):
    """Wrapper that delegates reset_choices to the inner magicgui widget."""
    def reset_choices(self):
        _puncta_widget.reset_choices()

puncta_widget = _PunctaWidgetContainer(labels=False)
puncta_widget._puncta_widget = _puncta_widget
header = widgets.Label(value="Puncta Detection")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Algorithmic detection of puncta.</i>")
info.native.setObjectName("widgetInfo")

# Insert section headers into the magicgui form's internal layout
# Widget order: image_layer(0), nuclei_layer(1), nuclei_only(2), method(3),
#   threshold(4), min_distance(5), sigma(6), z_scale(7),
#   use_tophat(8), tophat_radius(9), call_button(10)
_inner = _puncta_widget.native.layout()

# "Algorithm" header + description after nuclei_only (index 3 -> before method)
_algo_header = widgets.Label(value="<b>Algorithm:</b>")
_inner.insertWidget(3, _make_divider())
_inner.insertWidget(4, _algo_header.native)
# method is now at 5, description after it at 6
_inner.insertWidget(6, _method_desc_qlabel)

# "Detection Parameters" header before threshold (was 4, now shifted to 7)
_params_header = widgets.Label(value="<b>Detection Parameters:</b>")
_inner.insertWidget(7, _make_divider())
_inner.insertWidget(8, _params_header.native)

# "Background Subtraction" header before use_tophat (was 8, now shifted to 13)
_bg_header = widgets.Label(value="<b>Background Subtraction:</b>")
_inner.insertWidget(13, _make_divider())
_inner.insertWidget(14, _bg_header.native)

# Tighten inner magicgui form
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

# Outer layout: whole form as one block
_layout = puncta_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_puncta_widget.native)
_layout.addStretch(1)