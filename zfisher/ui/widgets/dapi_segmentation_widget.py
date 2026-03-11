import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ...core.segmentation import segment_nuclei_classical
from ... import constants
from ._shared import make_header_divider

@magicgui(
    call_button="Run DAPI Mapping",
    r1_layer={"label": "Round 1 (DAPI)"},
    r2_layer={"label": "Round 2 (DAPI)"},
    auto_call=False,
)
@require_active_session("Please start or load a session before running segmentation.")
@error_handler("DAPI Segmentation Failed")
def _dapi_segmentation_widget(
    r1_layer: "napari.layers.Image",
    r2_layer: "napari.layers.Image"
):
    """Runs segmentation on selected DAPI channels."""
    viewer = napari.current_viewer()

    layers_to_process = [l for l in [r1_layer, r2_layer] if l is not None]
    
    if not layers_to_process:
        viewer.status = "No channels selected."
        return

    viewer.status = f"Segmenting {len(layers_to_process)} layer(s)..."
    
    with popups.ProgressDialog(viewer.window._qt_window, title="Segmenting Nuclei...") as dialog:
        num_layers = len(layers_to_process)
        # Reserve 0-85% for segmentation, 85-100% for loading layers
        seg_pct = 85
        results = []
        for i, layer in enumerate(layers_to_process):
            dialog.update_progress(0, f"Starting segmentation for {layer.name}...")

            def on_progress(value, text, _i=i):
                base_progress = (_i / num_layers) * seg_pct
                scaled_value = base_progress + (value / num_layers) * (seg_pct / 100)
                dialog.update_progress(int(scaled_value), f"{layer.name}: {text}")

            masks, centroids = segment_nuclei_classical(layer.data, progress_callback=on_progress)
            results.append((layer, masks, centroids))

        # Freeze vispy canvas before adding layers to prevent GL access
        # violations from processEvents triggering draws mid-mutation.
        dialog.freeze_canvas()

        # Load results into viewer with progress feedback
        for i, (layer, masks, centroids) in enumerate(results):
            pct = seg_pct + int(((i + 1) / len(results)) * (100 - seg_pct))
            dialog.update_progress(pct, f"Loading layers: {layer.name}...")
            viewer_helpers.add_segmentation_results_to_viewer(viewer, layer, masks, centroids)

        dialog.update_progress(100, "Complete.")
        viewer.status = "Segmentation complete."

# --- UI Wrapper ---
class _DapiSegmentationContainer(Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    def reset_choices(self):
        _dapi_segmentation_widget.reset_choices()

dapi_segmentation_widget = _DapiSegmentationContainer(labels=False)
dapi_segmentation_widget._dapi_segmentation_widget = _dapi_segmentation_widget
header = Label(value="DAPI Mapping")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Segments nuclei in DAPI channels.</i>")
info.native.setObjectName("widgetInfo")
dapi_segmentation_widget.extend([header, info, make_header_divider(), _dapi_segmentation_widget])
