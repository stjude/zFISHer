import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, registration, stages
from .. import popups, viewer_helpers, events
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_divider as _make_divider, make_section_header as _make_section_header, make_section_desc as _make_section_desc, make_spacer as _make_spacer

@magicgui(
    call_button="Generate Global Canvas",
    layout="vertical",
    apply_warp={"label": "Apply Deformable Warping?", "tooltip": "Apply elastic warping to correct tissue shape differences between rounds. Disable for rigid-only alignment."},
    show_checkerboard={"label": "Show Checkerboard", "value": True, "tooltip": "Show a checkerboard pattern to visually assess alignment quality."},
    show_deformation={"label": "Show Deformation Field", "value": True, "tooltip": "Show a vector field visualizing how the tissue was warped."},
    hide_raw={"label": "Hide Raw Layers?", "tooltip": "Hide raw input layers after processing, showing only aligned results."}
)
@require_active_session("Please start or load a session before generating the canvas.")
@error_handler("Canvas Generation Failed")
def _canvas_widget(
    apply_warp: bool = True,
    show_checkerboard: bool = True,
    show_deformation: bool = True,
    hide_raw: bool = False
):
    """
    Applies the calculated shift to all layers and creates a global canvas.
    Refactored to delegate all warping and saving logic to core.registration.
    """
    viewer = napari.current_viewer()

    # 1. Retrieve the calculated shift from the session
    shift_list = session.get_data("shift")
    shift = np.array(shift_list) if shift_list else None
    
    if shift is None:
        viewer.status = "No shift calculated. Run Registration first."
        return
        
    # 2. Setup the output directory
    base_output_dir = session.get_data("output_dir")
    output_dir = Path(base_output_dir) / constants.ALIGNED_DIR
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 3. OOM Protection: Sanity check the shift
    if abs(shift[0]) > 1000:
        popups.show_error_popup(
            viewer.window._qt_window, 
            "Shift Too Large", 
            f"ABORTING: Calculated Z-shift ({shift[0]:.2f}) would crash the application."
        )
        return

    # 4. Extract layer data to pass to the core engine
    r1_layers_data = []
    r2_layers_data = []
    
    for l in viewer.layers:
        if isinstance(l, (napari.layers.Image, napari.layers.Labels)):
            layer_info = {
                'name': l.name, 
                'data': l.data, 
                'colormap': l.colormap.name if hasattr(l, 'colormap') else 'gray', 
                'scale': l.scale,
                'is_label': isinstance(l, napari.layers.Labels)
            }
            if "R1" in l.name:
                r1_layers_data.append(layer_info)
            elif "R2" in l.name:
                r2_layers_data.append(layer_info)

    # 5. Execute Core Orchestration with UI Progress Feedback
    with popups.ProgressDialog(viewer.window._qt_window, title="Generating Global Canvas") as dialog:

        # Core computation uses 0-70%, layer loading uses 70-80%, puncta transform 80-100%
        results, bspline_transform, canvas_offset = registration.generate_global_canvas(
            r1_layers_data,
            r2_layers_data,
            shift,
            output_dir,
            apply_warp=apply_warp,
            progress_callback=lambda p, m: dialog.update_progress(int(p * 0.70), m),
            # Rounds whose nuclear channels are named differently (2.1.2's batch fix).
            r2_channel_names=stages.nuclear_channel_names(),
        )

        # 6. Remove existing aligned/warped layers to prevent duplicates on rerun
        existing_names = {layer_info['name'] for layer_info in results}
        for old_layer in list(viewer.layers):
            if old_layer.name in existing_names:
                old_layer._locked = False
                try:
                    viewer.layers.remove(old_layer)
                except ValueError:
                    pass

        # Add resulting layers back to napari
        n_results = max(len(results), 1)
        for i, layer_info in enumerate(results):
            pct = 70 + int(((i + 1) / n_results) * 10)
            dialog.update_progress(pct, f"Loading layer: {layer_info['name']}...")

            # Skip checkerboard/deformation if user disabled them
            if not show_checkerboard and constants.WARPED_CHECKERBOARD_NAME in layer_info['name']:
                continue
            if not show_deformation and constants.DEFORMATION_FIELD_NAME in layer_info['name']:
                continue

            layer_type = layer_info['type']
            meta = layer_info['meta']

            # Skip aligned/warped mask labels — they exist on disk for the
            # consensus step but adding them to the viewer causes duplicate IDs.
            if layer_type == 'labels' and constants.MASKS_SUFFIX in layer_info['name']:
                continue
            if layer_type == 'labels':
                layer = viewer.add_labels(
                    layer_info['data'],
                    name=layer_info['name'],
                    scale=meta['scale'],
                    opacity=0.6, visible=False,
                )
                # Use iso_categorical for better 3D rendering of masks alongside points
                layer.rendering = 'iso_categorical'
            elif layer_type == 'image':
                viewer.add_image(
                    layer_info['data'],
                    name=layer_info['name'],
                    colormap=meta.get('colormap', 'gray'),
                    scale=meta['scale'],
                    blending=meta.get('blending', 'additive'),
                    opacity=meta.get('opacity', 1.0), visible=False,
                )
            elif layer_type == 'vectors':
                viewer.add_vectors(
                    layer_info['data'],
                    name=layer_info['name'],
                    scale=meta['scale'],
                    edge_width=0.2,
                    length=2.5,
                    edge_color='cyan', visible=False,
                )

        # 7. Transform existing puncta layers into aligned/warped space
        # Lock originals first so they can't be deleted during transform
        for l in viewer.layers:
            if isinstance(l, napari.layers.Points) and constants.PUNCTA_SUFFIX in l.name:
                events.lock_layer(l)
        puncta_layers = [
            l for l in list(viewer.layers)
            if isinstance(l, napari.layers.Points)
            and constants.PUNCTA_SUFFIX in l.name
            and constants.ALIGNED_PREFIX not in l.name
            and constants.WARPED_PREFIX not in l.name
        ]
        if puncta_layers:
            voxels = session.get_data("canvas_scale")
            if not voxels:
                # Fallback: use scale from the first R1 image layer
                ref = next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and "R1" in l.name), None)
                voxels = tuple(ref.scale) if ref else (1, 1, 1)
            # The consensus mask does not exist yet in the manual workflow: no
            # nuclei now; Match Nuclei assigns them next.
            viewer_helpers.transform_raw_puncta_layers(
                viewer, puncta_layers, tuple(voxels), remove_extranuclear=False, assign_nuclei=False,
                progress=lambda pct, text: dialog.update_progress(80 + int(pct * 0.18), text))

        # Lock all puncta layers after warping — no further deletion allowed
        for layer in viewer.layers:
            if isinstance(layer, napari.layers.Points) and constants.PUNCTA_SUFFIX in layer.name:
                events.lock_layer(layer)

        dialog.update_progress(100, "Complete.")

    # 8. Final UI Tidy Up
    if hide_raw:
        nuc_ch = session.get_nuclear_channel()
        r1_nuc = f"{constants.ALIGNED_PREFIX} R1 - {nuc_ch}"
        # R2 is "Warped" when deformable warp applied, "Aligned" otherwise
        r2_warped = f"{constants.WARPED_PREFIX} R2 - {nuc_ch}"
        r2_aligned = f"{constants.ALIGNED_PREFIX} R2 - {nuc_ch}"
        show = {r1_nuc, r2_warped, r2_aligned}
        for layer in viewer.layers:
            layer.visible = layer.name in show

    viewer.status = "Global Canvas Generation Complete."

# --- UI Helpers ---
from qtpy.QtWidgets import QLabel, QSizePolicy

# --- UI Wrapper ---
class _CanvasContainer(Container):
    def reset_choices(self):
        _canvas_widget.reset_choices()

canvas_widget = _CanvasContainer(labels=False)
canvas_widget._canvas_widget = _canvas_widget
header = Label(value="Global Canvas")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Applies registration and creates aligned layers.</i>")
info.native.setObjectName("widgetInfo")
info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

_layout = canvas_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())

# Insert section headers into inner form
_inner = _canvas_widget.native.layout()
_options_header = _make_section_header("Options")
_options_desc = _make_section_desc("Configure warping and layer visibility after alignment.")
_inner.insertWidget(0, _options_header)
_inner.insertWidget(1, _options_desc)
_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

_canvas_widget.native.setMinimumWidth(0)
from qtpy.QtWidgets import QAbstractSpinBox, QComboBox
for child in _canvas_widget.native.findChildren(QLabel):
    child.setMinimumWidth(0)
for child in _canvas_widget.native.findChildren(QAbstractSpinBox) + _canvas_widget.native.findChildren(QComboBox):
    child.setMinimumWidth(0)

_layout.addWidget(_canvas_widget.native)
_layout.addStretch(1)