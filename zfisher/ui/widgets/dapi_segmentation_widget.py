import logging
import napari
import numpy as np
from magicgui.widgets import Container, PushButton, Label, ComboBox, create_widget
from qtpy.QtWidgets import QFrame, QLabel

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import error_handler
from ..style import COLORS
from ...core.segmentation import (
    segment_nuclei_classical, segment_nuclei_cellpose,
    get_label_volumes, compute_min_volume_threshold, filter_small_labels,
)
from ... import constants

logger = logging.getLogger(__name__)


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


class DapiSegmentationWidget(Container):
    """Class-based nuclei mapping widget with full layout control."""

    def __init__(self):
        super().__init__(labels=False)

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        self._header = Label(value="Nuclei Mapping")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Automatically segments nuclei from the nuclear stain channel in each round.</i>")
        self._info.native.setObjectName("widgetInfo")

        self.r1_layer = create_widget(
            annotation="napari.layers.Image", options={
                "label": "R1 (Nuclei Channel):",
                "tooltip": "Select the nuclear stain channel (e.g. DAPI) for Round 1.",
            }
        )
        self.r2_layer = create_widget(
            annotation="napari.layers.Image", options={
                "label": "R2 (Nuclei Channel):",
                "tooltip": "Select the nuclear stain channel (e.g. DAPI) for Round 2.",
            }
        )
        self.method = ComboBox(
            label="Method:",
            choices=["Classical (Fast)", "Deep Learning (Cellpose)"],
            value="Classical (Fast)",
            tooltip="Classical: Fast, watershed-based. Cellpose: Deep learning, slower but more accurate.",
        )
        self._run_btn = PushButton(text="Run Nuclei Mapping")
        self._run_btn.tooltip = "Run nuclei segmentation on the selected layers."

        self._cellpose_status = QLabel("")
        self._cellpose_status.setWordWrap(True)
        self._cellpose_status.setVisible(False)
        self._cellpose_status.setStyleSheet("margin: 4px 2px; font-size: 12px;")

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        self._desc = QLabel(
            "Select the nuclear stain channel for each round. "
            "Classical is fast watershed-based segmentation. "
            "Deep learning uses Cellpose for higher accuracy but is slower."
        )
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: white; margin: 4px 2px;")

        # Wrap layer selectors in labelled containers
        self._r1_form = Container(labels=True)
        self._r1_form.extend([self.r1_layer])
        self._r1_form.native.layout().setContentsMargins(0, 10, 0, 0)

        self._r2_form = Container(labels=True)
        self._r2_form.extend([self.r2_layer])
        self._r2_form.native.layout().setContentsMargins(0, 0, 0, 10)

        self._method_form = Container(labels=True)
        self._method_form.extend([self.method])
        self._method_form.native.layout().setContentsMargins(0, 10, 0, 0)

        _layout = self.native.layout()
        _layout.setSpacing(2)
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._desc)
        _layout.addWidget(self._r1_form.native)
        _layout.addWidget(self._r2_form.native)
        _layout.addWidget(self._method_form.native)
        _layout.addWidget(self._cellpose_status)
        self._run_btn.native.setStyleSheet("margin-top: 20px;")
        _layout.addWidget(self._run_btn.native)
        _layout.addStretch(1)

    def _connect_signals(self):
        self._run_btn.changed.connect(self._on_run)
        self.method.changed.connect(self._on_method_changed)

    def _on_method_changed(self, method):
        is_cellpose = method == "Deep Learning (Cellpose)"
        self._cellpose_status.setVisible(is_cellpose)
        if is_cellpose:
            self._check_cellpose_model()

    def _check_cellpose_model(self):
        from cellpose.models import MODEL_DIR
        model_file = MODEL_DIR / "cpsam"
        if model_file.exists():
            self._cellpose_status.setText("Cellpose model found.")
            self._cellpose_status.setStyleSheet("color: #a6e3a1; margin: 4px 2px; font-size: 12px;")
        else:
            self._cellpose_status.setText("Cellpose model not found. Download will begin on run.")
            self._cellpose_status.setStyleSheet("color: #f9e2af; margin: 4px 2px; font-size: 12px;")

    def reset_choices(self):
        self.r1_layer.reset_choices()
        self.r2_layer.reset_choices()

    @error_handler("Nuclei Segmentation Failed")
    def _on_run(self):
        if not session.get_data("output_dir"):
            popups.show_error_popup(
                None, "No Session",
                "Please start or load a session before running segmentation."
            )
            return

        viewer = napari.current_viewer()
        r1_layer = self.r1_layer.value
        r2_layer = self.r2_layer.value
        method = self.method.value
        merge_splits = True

        logger.info("Nuclei segmentation settings: method=%s, merge_splits=%s, R1=%s, R2=%s",
                     method, merge_splits,
                     r1_layer.name if r1_layer else None,
                     r2_layer.name if r2_layer else None)

        layers_to_process = [l for l in [r1_layer, r2_layer] if l is not None]

        if not layers_to_process:
            viewer.status = "No channels selected."
            return

        # Check if mask layers already exist — ask before overwriting
        existing = [
            f"{l.name}{constants.MASKS_SUFFIX}" for l in layers_to_process
            if f"{l.name}{constants.MASKS_SUFFIX}" in viewer.layers
        ]
        if existing:
            if not popups.show_yes_no_popup(
                viewer.window._qt_window,
                "Masks Already Exist",
                f"The following mask layers already exist:\n\n"
                + "\n".join(f"  • {n}" for n in existing)
                + "\n\nReplace them with new segmentation?",
            ):
                viewer.status = "Segmentation cancelled."
                return

        # If Cellpose selected, check if model is downloaded
        if method == "Deep Learning (Cellpose)":
            from cellpose.models import MODEL_DIR
            model_file = MODEL_DIR / "cpsam"
            if not model_file.exists():
                if not popups.show_yes_no_popup(
                    viewer.window._qt_window,
                    "Cellpose Model Not Found",
                    "The Cellpose model (cpsam) needs to be downloaded (~100 MB).\n\n"
                    "This is a one-time download. Continue?",
                ):
                    viewer.status = "Cellpose download cancelled."
                    return
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                with popups.ProgressDialog(viewer.window._qt_window, title="Downloading Cellpose Model...") as dl_dialog:
                    dl_dialog.update_progress(0, "Connecting to server...")
                    _download_cellpose_model(str(model_file), dl_dialog)
                    dl_dialog.update_progress(100, "Download complete.")

        viewer.status = f"Segmenting {len(layers_to_process)} layer(s)..."

        with popups.ProgressDialog(viewer.window._qt_window, title="Segmenting Nuclei...") as dialog:
            num_layers = len(layers_to_process)
            seg_pct = 70
            results = []

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

            dialog.freeze_canvas()

            # Suppress custom layer control callbacks during batch layer creation
            from .. import viewer as viewer_module
            viewer_module._suppress_custom_controls = True
            try:
                n_filtered = max(len(filtered_results), 1)
                for i, (layer, masks, centroids) in enumerate(filtered_results):
                    pct = 85 + int((i / n_filtered) * 13)
                    dialog.update_progress(pct, f"Loading masks & IDs: {layer.name}...")
                    viewer_helpers.add_segmentation_results_to_viewer(viewer, layer, masks, centroids)
            finally:
                viewer_module._suppress_custom_controls = False

            dialog.update_progress(100, "Complete.")


# Public instance — matches the old API expected by viewer.py, events.py, nuclei_segmentation_widget.py
dapi_segmentation_widget = DapiSegmentationWidget()
# Expose inner reference for events.py compatibility (_try_set accesses ._dapi_segmentation_widget.r1_layer)
dapi_segmentation_widget._dapi_segmentation_widget = dapi_segmentation_widget
