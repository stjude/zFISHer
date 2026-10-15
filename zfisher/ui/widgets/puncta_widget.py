import logging
import napari
from magicgui import magicgui, widgets
from ...core import session, puncta
from qtpy.QtWidgets import QApplication
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_divider as _make_divider, make_section_header as _make_section_header, make_section_desc as _make_section_desc, make_spacer as _make_spacer

logger = logging.getLogger(__name__)

@magicgui(
    call_button="Detect Puncta",
    layout="vertical",
    image_layer={"label": "Target Channel", "tooltip": "The image channel to scan for puncta."},
    nuclei_layer={"label": "Nuclei Masks", "tooltip": "Nuclei mask used to filter puncta. Only puncta inside nuclei are kept."},
    nuclei_only={"label": "Nuclei Only (discard outside nuclei)", "value": True, "tooltip": "Keep only puncta inside the nuclei mask. Uncheck to detect puncta everywhere."},
    method={
        "label": "Algorithm",
        "choices": ["Local Maxima", "Laplacian of Gaussian", "Difference of Gaussian", "Radial Symmetry"],
        "tooltip": "Detection algorithm: Local Maxima (fastest, well-separated puncta), Laplacian of Gaussian (blob-aware, crowded fields), Difference of Gaussian (fast blob approximation), Radial Symmetry (high-density fields)."
    },
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01, "value": constants.PUNCTA_THRESHOLD_REL, "tooltip": "Intensity threshold as a fraction (0-1) of the maximum. Lower values detect dimmer puncta but may increase false positives."},
    min_distance={"label": "Min Distance (px)", "min": 1, "max": 20, "value": constants.PUNCTA_MIN_DISTANCE, "tooltip": "Minimum separation between detected puncta in pixels. Prevents double-counting nearby puncta."},
    sigma={"label": "Spot Radius, px (Sigma)", "min": 0.0, "max": 5.0, "step": 0.1, "value": constants.PUNCTA_SIGMA, "tooltip": "Gaussian sigma for detection in pixels. Match to the approximate radius of your puncta. 0 = no smoothing."},
    z_scale={"label": "Z/XY Scale Ratio", "min": 0.01, "max": 20.0, "step": 0.01, "value": 1.0, "tooltip": "Ratio of Z-step to XY pixel size. Corrects for non-cubic voxels. Auto-computed from image metadata when available."},
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

    # Detecting on an aligned or warped R2 image is not batch's method: ask
    # that first, before what to do with the puncta already there.
    aligned = image_layer.name.startswith((constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))
    on_original = aligned and _detect_on_original.isEnabled() and _detect_on_original.isChecked()
    key = puncta.channel_key(image_layer.name)
    if aligned and not on_original and key and key.startswith("R2 - ") and not _confirm_aligned_r2(viewer, image_layer):
        return

    # Check if puncta layer already exists — ask before running detection
    _puncta_layer_name = f"{image_layer.name}{constants.PUNCTA_SUFFIX}"
    _existing_action = None
    if _puncta_layer_name in viewer.layers:
        _existing = viewer.layers[_puncta_layer_name]
        _existing_action = viewer_helpers.ask_replace_or_merge(
            viewer.window._qt_window, _puncta_layer_name, len(_existing.data),
            n_manual=int(viewer_helpers.manual_mask(_existing.features).sum()),
        )
        if _existing_action == 'cancel' or _existing_action is None:
            return

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

    if on_original:
        _redetect_on_original(viewer, image_layer, params, _existing_action)
        return

    with popups.ProgressDialog(viewer.window._qt_window, f"Processing {image_layer.name}...") as dialog:
        _raw_puncta_name = f"{image_layer.name}{constants.PUNCTA_SUFFIX}"

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
                    # No CSV here: the layer's CSV is written once below, from
                    # the layer itself. Writing the detection alone first put
                    # only the new puncta on disk, losing manual ones whenever
                    # the update below did not rewrite the file.
                    output_path=None,
                    layer_name=_raw_puncta_name,
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
        if _existing_action is None and len(results):
            # A new layer: ids the channel has never had (Replace and Merge
            # number their new rows themselves).
            results[:, 6] = puncta.allocate_puncta_ids(_puncta_layer_name, len(results))

        # Freeze canvas before adding/updating the points layer to prevent
        # vispy from drawing stale GL buffers during processEvents()
        dialog.freeze_canvas()
        from .. import viewer as _viewer_mod
        _viewer_mod._suppress_custom_controls = True
        try:
            viewer_helpers.add_or_update_puncta_layer(viewer, image_layer, results,
                                                       existing_action=_existing_action)
            viewer_helpers.record_detection_provenance(image_layer, nuclei_layer, params,
                                                       existing_action=_existing_action)
        finally:
            _viewer_mod._suppress_custom_controls = False
            # Re-trigger custom controls now that suppression is lifted —
            # the callbacks were skipped when the layer was added.
            viewer.layers.selection.events.changed(added=set(), removed=set())

        # Persist the detection parameters for this channel in the session
        aligned = image_layer.name.startswith((constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))
        puncta.record_detection_settings(image_layer.name, {
            'algorithm': method,
            'sensitivity': threshold,
            'min_distance': min_distance,
            'sigma': sigma,
            'z_scale': z_scale,
            'nuclei_only': nuclei_only,
            'tophat': use_tophat,
            'tophat_radius': tophat_radius,
            'num_puncta': len(results),
            'detected_in': 'aligned' if aligned else 'raw',
        })

        viewer.status = f"Found {len(results)} spots in {image_layer.name}."

def _confirm_aligned_r2(viewer, image_layer):
    """Ask before detecting on an aligned or warped R2 image, which is R2
    resampled into R1's space: not the method batch uses."""
    text = (f"{image_layer.name} is the R2 image after it was resampled into R1's space. Detecting on it "
            "is not the method batch uses (detect on the original R2 image, then warp the puncta), and "
            "the puncta can differ. The report will say this channel was detected on the aligned image.")
    if _detect_on_original.isEnabled():
        text += "\n\nTick \"Detect on original image, then warp\" to use batch's method instead."
    return popups.show_yes_no_popup(viewer.window._qt_window, "Detect on the aligned R2 image?",
                                    text + "\n\nDetect on the aligned image anyway?")


def _redetect_on_original(viewer, image_layer, params, existing_action):
    """Re-detect the channel of the aligned ``image_layer`` on its round's
    original image and warp the puncta onto it (``stages.redetect_channel``),
    replacing the aligned puncta layer in place."""
    import threading
    from collections import deque
    import pandas as pd
    from ...core import stages
    from .mask_editor_widget import flush_pending_mask_saves

    round_id, channel = puncta.channel_key(image_layer.name).split(" - ", 1)
    parent = viewer.window._qt_window
    # On this thread: finding a moved image can ask the user.
    image_path = popups.resolve_raw_path(parent, round_id)
    if image_path is None:
        viewer.status = f"{round_id} {channel} was not re-detected: its original image was not found."
        return
    flush_pending_mask_saves()                     # the consensus mask on disk is the one on screen
    consensus = viewer_helpers.find_consensus_layer(viewer)
    consensus_data = consensus.data if consensus is not None else None
    result, error, progress = [None], [None], deque()

    def run():
        try:
            result[0] = stages.redetect_channel(
                round_id, channel, params, existing=existing_action or "replace", image_path=image_path,
                consensus_mask=consensus_data, progress_callback=lambda p, t: progress.append((p, t)))
        except Exception as exc:
            error[0] = exc

    with popups.ProgressDialog(parent, f"Re-detecting {round_id} {channel}...") as dialog:
        dialog.update_progress(0, f"Reading the original {round_id} {channel} image...")
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=0.05)
            while progress:
                dialog.update_progress(*progress.popleft())
            QApplication.processEvents()
        while progress:
            dialog.update_progress(*progress.popleft())
        if error[0]:
            raise error[0]
        _raw_csv, aligned_csv = result[0]
        if aligned_csv is None:
            viewer.status = f"No {round_id} {channel} puncta were found or reached aligned space; nothing changed."
            return
        dialog.freeze_canvas()
        from .. import viewer as _viewer_mod
        _viewer_mod._suppress_custom_controls = True
        try:
            viewer_helpers.show_puncta_csv(viewer, f"{image_layer.name}{constants.PUNCTA_SUFFIX}",
                                           aligned_csv, image_layer)
        finally:
            _viewer_mod._suppress_custom_controls = False
            viewer.layers.selection.events.changed(added=set(), removed=set())

    table = pd.read_csv(aligned_csv)
    n_manual = int((table.get("Source") == constants.PUNCTA_SOURCE_MANUAL).sum()) if "Source" in table else 0
    note = stages.report_note()
    viewer.status = (f"Re-detected {round_id} {channel} on the original image and warped it: {len(table)} puncta"
                     + (f", {n_manual} placed by hand kept" if n_manual else "") + "."
                     + (f" {note}" if note else ""))
    from .colocalization_widget import refresh_report_note
    refresh_report_note()


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
    _update_detect_on_original(new_layer)
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

class _PunctaWidgetContainer(widgets.Container):
    """Wrapper that delegates reset_choices to the inner magicgui widget."""
    def reset_choices(self):
        _puncta_widget.reset_choices()

puncta_widget = _PunctaWidgetContainer(labels=False)
puncta_widget._puncta_widget = _puncta_widget
header = widgets.Label(value="Puncta Detection")
header.native.setObjectName("widgetHeader")
info = widgets.Label(value="<i>Automated puncta detection using intensity-based algorithms.</i>")
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
_target_header = _make_section_header("Input Layers")
_target_desc = _make_section_desc("Select the image channel to scan and the nuclei mask to filter against.")
_inner.insertWidget(0, _target_header)
_inner.insertWidget(1, _target_desc)
# image_layer(2), nuclei_layer(3)

# --- "Algorithm" section: method(4) ---
_algo_header = _make_section_header("Algorithm")
_algo_desc = _make_section_desc("Choose the detection algorithm. Available parameters change based on your selection.")
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

# --- "Detect on the original image" (aligned and warped image layers only) ---
from qtpy.QtWidgets import QCheckBox
_detect_on_original = QCheckBox("Detect on original image, then warp")
_DETECT_ON_ORIGINAL_TIP = (
    "Detect on the round's original image, then warp the puncta into aligned space with the session's "
    "saved registration: the method batch uses (detect, then warp). Puncta outside nuclei are dropped with the round's own "
    "nuclei mask, as in batch, not with the mask chosen above. Unticked, detection runs on the aligned "
    "image itself, which for R2 has been resampled.")
_detect_on_original.setChecked(True)
_detect_on_original.setToolTip(_DETECT_ON_ORIGINAL_TIP)
_detect_on_original.setVisible(False)
_inner.insertWidget(_inner.indexOf(_algo_header) - 2, _detect_on_original)


def _update_detect_on_original(layer):
    """Show the tick box for aligned and warped image layers; enable it when
    the session has a saved registration to warp with."""
    aligned = layer is not None and layer.name.startswith((constants.ALIGNED_PREFIX, constants.WARPED_PREFIX))
    _detect_on_original.setVisible(bool(aligned))
    if not aligned:
        return
    from ...core import stages
    ok, reason = stages.can_rewarp()
    if ok and not _detect_on_original.isEnabled():
        _detect_on_original.setChecked(True)       # the default again, after a session that could not
    _detect_on_original.setEnabled(ok)
    _detect_on_original.setToolTip(_DETECT_ON_ORIGINAL_TIP if ok else f"Not available: {reason}")
    if not ok:
        _detect_on_original.setChecked(False)


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
_puncta_widget.native.setMinimumWidth(0)
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox, QLabel
for child in _puncta_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
for child in _puncta_widget.native.findChildren(QAbstractSpinBox) + _puncta_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)

_layout.addWidget(_puncta_widget.native)
_layout.addStretch(1)