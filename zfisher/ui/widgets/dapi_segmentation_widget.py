import logging
import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler

logger = logging.getLogger(__name__)
from ...core.segmentation import (
    segment_nuclei_classical, segment_nuclei_cellpose,
    get_label_volumes, compute_min_volume_threshold, filter_small_labels,
)
from ... import constants


def _download_cellpose_model(dest_path, dialog):
    """Download the Cellpose cpsam model with progress updates."""
    import urllib.request
    import tempfile
    import shutil
    from cellpose.models import _CPSAM_MODEL_URL

    dialog.update_progress(5, "Downloading Cellpose model...")
    req = urllib.request.urlopen(urllib.request.Request(_CPSAM_MODEL_URL))
    file_size = int(req.headers.get("Content-Length", 0))

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        downloaded = 0
        while True:
            chunk = req.read(8192)
            if not chunk:
                break
            tmp.write(chunk)
            downloaded += len(chunk)
            if file_size > 0:
                pct = min(int((downloaded / file_size) * 95), 95)
                mb = downloaded / (1024 * 1024)
                total_mb = file_size / (1024 * 1024)
                dialog.update_progress(pct, f"Downloading... {mb:.1f} / {total_mb:.1f} MB")
        tmp_path = tmp.name

    shutil.move(tmp_path, dest_path)


@magicgui(
    call_button="Run Nuclei Mapping",
    r1_layer={"label": "Round 1 (Nuclei)", "tooltip": "Select the nuclear stain channel (e.g. DAPI) for Round 1."},
    r2_layer={"label": "Round 2 (Nuclei)", "tooltip": "Select the nuclear stain channel (e.g. DAPI) for Round 2."},
    method={
        "label": "Method",
        "choices": ["Classical (Fast)", "Deep Learning (Cellpose)"],
        "tooltip": "Classical: Fast, watershed-based. Cellpose: Deep learning, slower but more accurate.",
    },
    merge_splits={
        "label": "Merge Splits",
        "tooltip": "Merge over-segmented nuclei that share a large boundary surface.",
    },
    auto_call=False,
)
@require_active_session("Please start or load a session before running segmentation.")
@error_handler("Nuclei Segmentation Failed")
def _dapi_segmentation_widget(
    r1_layer: "napari.layers.Image",
    r2_layer: "napari.layers.Image",
    method: str = "Classical (Fast)",
    merge_splits: bool = True,
):
    """Runs segmentation on selected nuclei channels."""
    viewer = napari.current_viewer()
    logger.info("Nuclei segmentation settings: method=%s, merge_splits=%s, R1=%s, R2=%s",
                method, merge_splits,
                r1_layer.name if r1_layer else None,
                r2_layer.name if r2_layer else None)

    layers_to_process = [l for l in [r1_layer, r2_layer] if l is not None]

    if not layers_to_process:
        viewer.status = "No channels selected."
        return

    # If Cellpose selected, check if model is downloaded
    if method == "Deep Learning (Cellpose)":
        from cellpose.models import MODEL_DIR
        model_file = MODEL_DIR / "cpsam"
        if not model_file.exists():
            from qtpy.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                viewer.window._qt_window,
                "Cellpose Model Not Found",
                "The Cellpose model (cpsam) needs to be downloaded (~100 MB).\n\n"
                "This is a one-time download. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                viewer.status = "Cellpose download cancelled."
                return
            # Download with progress dialog
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            with popups.ProgressDialog(viewer.window._qt_window, title="Downloading Cellpose Model...") as dl_dialog:
                dl_dialog.update_progress(0, "Connecting to server...")
                _download_cellpose_model(str(model_file), dl_dialog)
                dl_dialog.update_progress(100, "Download complete.")

    viewer.status = f"Segmenting {len(layers_to_process)} layer(s)..."

    with popups.ProgressDialog(viewer.window._qt_window, title="Segmenting Nuclei...") as dialog:
        num_layers = len(layers_to_process)
        # Reserve 0-70% for segmentation, 70-85% for filtering, 85-100% for loading
        seg_pct = 70
        results = []

        # Pass 1: Segment all rounds
        for i, layer in enumerate(layers_to_process):
            dialog.update_progress(0, f"Starting segmentation for {layer.name}...")

            def on_progress(value, text, _i=i):
                base_progress = (_i / num_layers) * seg_pct
                scaled_value = base_progress + (value / num_layers) * (seg_pct / 100)
                dialog.update_progress(int(scaled_value), f"{layer.name}: {text}")

            voxel_spacing = tuple(layer.scale) if layer.scale is not None else None
            if method == "Deep Learning (Cellpose)":
                masks, centroids = segment_nuclei_cellpose(layer.data, merge_splits=merge_splits, progress_callback=on_progress)
            else:
                masks, centroids = segment_nuclei_classical(layer.data, voxel_spacing=voxel_spacing, merge_splits=merge_splits, progress_callback=on_progress)
            results.append((layer, masks, centroids))

        # Pass 2: Pool volumes from all rounds, compute shared threshold, filter
        dialog.update_progress(seg_pct, "Computing volume threshold across rounds...")
        all_volumes = np.concatenate([get_label_volumes(masks) for _, masks, _ in results if masks is not None])
        min_vol = compute_min_volume_threshold(all_volumes)

        filtered_results = []
        for i, (layer, masks, centroids) in enumerate(results):
            if masks is None:
                filtered_results.append((layer, masks, centroids))
                continue
            pct = seg_pct + int(((i + 1) / num_layers) * 15)
            dialog.update_progress(pct, f"Filtering {layer.name}...")
            masks, centroids = filter_small_labels(masks, min_vol)
            filtered_results.append((layer, masks.astype(np.uint32), centroids))

        # Freeze vispy canvas before adding layers to prevent GL access
        # violations from processEvents triggering draws mid-mutation.
        dialog.freeze_canvas()

        # Load results into viewer with progress feedback
        n_filtered = max(len(filtered_results), 1)
        for i, (layer, masks, centroids) in enumerate(filtered_results):
            pct = 85 + int((i / n_filtered) * 13)
            dialog.update_progress(pct, f"Loading masks & IDs: {layer.name}...")
            viewer_helpers.add_segmentation_results_to_viewer(viewer, layer, masks, centroids)

        dialog.update_progress(100, "Complete.")

# --- UI Wrapper ---
from qtpy.QtWidgets import QFrame, QLabel, QSpacerItem, QSizePolicy
from ..style import COLORS

class _DapiSegmentationContainer(Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    def reset_choices(self):
        _dapi_segmentation_widget.reset_choices()

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

dapi_segmentation_widget = _DapiSegmentationContainer(labels=False)
dapi_segmentation_widget._dapi_segmentation_widget = _dapi_segmentation_widget

_header = Label(value="Nuclei Mapping")
_header.native.setObjectName("widgetHeader")
_info = Label(value="<i>Segments nuclei in the nuclear stain channel.</i>")
_info.native.setObjectName("widgetInfo")

_desc = QLabel(
    "Select the nuclear stain channel for each round. "
    "Classical is fast watershed-based segmentation. "
    "Cellpose uses deep learning for higher accuracy but is slower."
)
_desc.setWordWrap(True)
_desc.setStyleSheet("color: white; margin: 4px 2px;")

_layout = dapi_segmentation_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(_header.native)
_layout.addWidget(_info.native)
_layout.addWidget(_make_divider())
_layout.addWidget(_desc)
_layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed))
_dapi_segmentation_widget.native.setMinimumWidth(0)
_layout.addWidget(_dapi_segmentation_widget.native)

_layout.addStretch(1)
