import logging
import napari
from pathlib import Path
from magicgui import magicgui, widgets
from ...core import session, puncta
from qtpy.QtWidgets import QApplication
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

    # Check if puncta layer already exists — ask before running detection
    _puncta_layer_name = f"{image_layer.name}{constants.PUNCTA_SUFFIX}"
    _existing_action = None
    if _puncta_layer_name in viewer.layers:
        _existing_action = viewer_helpers.ask_replace_or_merge(
            viewer.window._qt_window, _puncta_layer_name,
            len(viewer.layers[_puncta_layer_name].data)
        )
        if _existing_action == 'cancel' or _existing_action is None:
            return

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

        # Run detection in a background thread so the indeterminate progress
        # bar can animate during the blocking computation.
        # Progress callbacks from the thread are queued and applied on the
        # main thread to avoid Qt cross-thread crashes.
        import threading
        from collections import deque

        _result_holder = [None]
        _error_holder = [None]
        _progress_queue = deque()

        def _thread_progress(p, t):
            _progress_queue.append((p, t))

        def _run_detection():
            try:
                _result_holder[0] = puncta.process_puncta_detection(
                    image_layer.data,
                    mask_data=nuclei_layer.data if nuclei_layer else None,
                    voxels=getattr(image_layer, 'scale', (1,1,1)),
                    params=params,
                    output_path=csv_path,
                    progress_callback=_thread_progress
                )
            except Exception as e:
                _error_holder[0] = e

        thread = threading.Thread(target=_run_detection, daemon=True)
        dialog.update_progress(-1, f"Detecting spots ({method})...")
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.05)
            # Drain progress updates from background thread on main thread
            while _progress_queue:
                p, t = _progress_queue.popleft()
                dialog.update_progress(p, t)
            QApplication.processEvents()
        # Drain any remaining progress updates
        while _progress_queue:
            p, t = _progress_queue.popleft()
            dialog.update_progress(p, t)
        if _error_holder[0]:
            raise _error_holder[0]
        results = _result_holder[0]

        # Freeze canvas before adding/updating the points layer to prevent
        # vispy from drawing stale GL buffers during processEvents()
        dialog.freeze_canvas()
        from .. import viewer as _viewer_mod
        _viewer_mod._suppress_custom_controls = True
        try:
            viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, results,
                                                       existing_action=_existing_action)
        finally:
            _viewer_mod._suppress_custom_controls = False
            # Re-trigger custom controls now that suppression is lifted —
            # the callbacks were skipped when the layer was added.
            viewer.layers.selection.events.changed(added=set(), removed=set())

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

def _make_section_header(text):
    """Create a left-aligned bold section header in light purple using plain QLabel."""
    label = QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

def _make_section_desc(text):
    """Create a white description label with word wrap and bottom margin."""
    desc = QLabel(text)
    desc.setWordWrap(True)
    desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
    return desc

def _make_spacer():
    from qtpy.QtWidgets import QWidget as _W
    s = _W()
    s.setFixedHeight(20)
    return s

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

# --- Rearrange: move nuclei_only out of its original position ---
# Original order: image_layer(0), nuclei_layer(1), nuclei_only(2), method(3), ...
_nuclei_only_native = _puncta_widget.nuclei_only.native
_nuclei_only_native.setParent(None)  # detach from layout
# Now: image_layer(0), nuclei_layer(1), method(2), threshold(3), ...

# --- "Target" section: image_layer(0), nuclei_layer(1) ---
_target_header = _make_section_header("Target")
_target_desc = _make_section_desc("Select the fluorescent channel and nuclei mask for puncta detection.")
_inner.insertWidget(0, _target_header)
_inner.insertWidget(1, _target_desc)
# image_layer(2), nuclei_layer(3)

# --- "Algorithm" section: method(4) ---
_algo_header = _make_section_header("Algorithm")
_algo_desc = _make_section_desc("Choose the detection algorithm. Parameters adjust automatically per method.")
_inner.insertWidget(4, _make_spacer())
_inner.insertWidget(5, _make_divider())
_inner.insertWidget(6, _algo_header)
_inner.insertWidget(7, _algo_desc)
# method is now at 8, insert description after it at 9
_inner.insertWidget(9, _method_desc_qlabel)

# --- "Detection Parameters" section: nuclei_only, threshold, min_distance, sigma, z_scale ---
_params_header = _make_section_header("Detection Parameters")
_params_desc = _make_section_desc("Fine-tune sensitivity, spacing, and spot size for your data.")
_inner.insertWidget(10, _make_spacer())
_inner.insertWidget(11, _make_divider())
_inner.insertWidget(12, _params_header)
_inner.insertWidget(13, _params_desc)
_inner.insertWidget(14, _nuclei_only_native)
# threshold(15), min_distance(16), sigma(17), z_scale(18)

# --- "Background Subtraction" section: use_tophat, tophat_radius ---
_bg_header = _make_section_header("Background Subtraction")
_bg_desc = _make_section_desc("Apply top-hat filtering to remove uneven background before detection.")
_inner.insertWidget(19, _make_spacer())
_inner.insertWidget(20, _make_divider())
_inner.insertWidget(21, _bg_header)
_inner.insertWidget(22, _bg_desc)
# use_tophat(23), tophat_radius(24), call_button(25)

# Add spacer above the Detect Puncta button (last widget = call_button)
_inner.insertWidget(_inner.count() - 1, _make_spacer())

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