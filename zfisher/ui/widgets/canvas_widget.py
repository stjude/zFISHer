import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, registration, puncta # Importing from core
from .. import popups, viewer_helpers, events
from ..decorators import require_active_session, error_handler
from ... import constants
from ._shared import make_header_divider

@magicgui(
    call_button="Generate Global Canvas",
    layout="vertical",
    apply_warp={"label": "Apply Deformable Warping?", "tooltip": "Apply deformable B-spline warping to correct tissue deformation between rounds."},
    hide_raw={"label": "Hide Raw Layers?", "tooltip": "Hide raw input layers after processing, showing only aligned results."}
)
@require_active_session("Please start or load a session before generating the canvas.")
@error_handler("Canvas Generation Failed")
def _canvas_widget(
    apply_warp: bool = True,
    hide_raw: bool = True
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
            progress_callback=lambda p, m: dialog.update_progress(int(p * 0.70), m)
        )

        # 6. Add resulting layers back to napari
        n_results = max(len(results), 1)
        for i, layer_info in enumerate(results):
            pct = 70 + int(((i + 1) / n_results) * 10)
            dialog.update_progress(pct, f"Loading layer: {layer_info['name']}...")

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
            import pandas as pd
            base_output = Path(base_output_dir)
            reports_dir = base_output / constants.REPORTS_DIR
            reports_dir.mkdir(exist_ok=True, parents=True)
            voxels = session.get_data("canvas_scale")
            if not voxels:
                # Fallback: use scale from the first R1 image layer
                ref = next((l for l in viewer.layers if isinstance(l, napari.layers.Image) and "R1" in l.name), None)
                voxels = tuple(ref.scale) if ref else (1, 1, 1)
            else:
                voxels = tuple(voxels)

            n_puncta = max(len(puncta_layers), 1)
            for pi, pts_layer in enumerate(puncta_layers):
                pct = 80 + int(((pi + 1) / n_puncta) * 18)
                dialog.update_progress(pct, f"Transforming puncta: {pts_layer.name}...")

                name_upper = pts_layer.name.upper()
                if "R1" in name_upper:
                    round_id = "R1"
                elif "R2" in name_upper:
                    round_id = "R2"
                else:
                    continue

                coords = np.array(pts_layer.data)
                if len(coords) == 0:
                    continue
                feats = pts_layer.features if hasattr(pts_layer, 'features') and isinstance(pts_layer.features, pd.DataFrame) and not pts_layer.features.empty else None
                if feats is not None and len(feats) == len(coords):
                    nuc_ids = feats['Nucleus_ID'].values if 'Nucleus_ID' in feats.columns else np.zeros(len(coords))
                    intensity = feats['Intensity'].values if 'Intensity' in feats.columns else np.zeros(len(coords))
                    snr = feats['SNR'].values if 'SNR' in feats.columns else np.zeros(len(coords))
                else:
                    nuc_ids = np.zeros(len(coords))
                    intensity = np.zeros(len(coords))
                    snr = np.zeros(len(coords))
                raw_puncta = np.column_stack([coords, nuc_ids, intensity, snr])

                prefix_str = constants.ALIGNED_PREFIX if (round_id == "R1" or bspline_transform is None) else constants.WARPED_PREFIX
                base_name = pts_layer.name.replace(constants.PUNCTA_SUFFIX, "")
                aligned_layer_name = f"{prefix_str} {base_name.strip()}{constants.PUNCTA_SUFFIX}"
                csv_out = reports_dir / f"{aligned_layer_name.replace(' ', '_')}.csv"

                transformed = puncta.transform_puncta_to_aligned_space(
                    raw_puncta=raw_puncta,
                    round_id=round_id,
                    shift=shift,
                    canvas_offset=canvas_offset,
                    bspline_transform=bspline_transform if round_id == "R2" else None,
                    consensus_mask=None,
                    output_path=csv_out,
                    layer_name=aligned_layer_name,
                )

                if transformed is not None and len(transformed) > 0:
                    ref = type('_ref', (), {
                        'name': aligned_layer_name.replace(constants.PUNCTA_SUFFIX, ''),
                        'scale': voxels,
                        'translate': (0,) * len(voxels),
                    })()
                    viewer_helpers.add_or_update_puncta_layer(viewer, ref, transformed)

                try:
                    pts_layer._locked = False  # Unlock so it can be removed
                    viewer.layers.remove(pts_layer)
                except ValueError:
                    pass

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
from qtpy.QtWidgets import QLabel, QFrame, QSizePolicy
from ..style import COLORS

def _make_divider():
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line

def _make_section_header(text):
    label = QLabel(f"<b style='color: #7a6b8a;'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label

def _make_section_desc(text):
    desc = QLabel(text)
    desc.setWordWrap(True)
    desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
    return desc

def _make_spacer():
    from qtpy.QtWidgets import QWidget as _W
    s = _W()
    s.setFixedHeight(20)
    return s

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

_layout.addWidget(_canvas_widget.native)
_layout.addStretch(1)