import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui

import zfisher.core.session as session
from .. import popups
from ..decorators import require_active_session, error_handler
from zfisher.core.pipeline import generate_global_canvas

@magicgui(
    call_button="Generate Global Canvas",
    apply_warp={"label": "Apply Deformable Warping?"},
    hide_raw={"label": "Hide Raw Layers?"}
)
@require_active_session("Please start or load a session before generating the canvas.")
@error_handler("Canvas Generation Failed")
def canvas_widget(
    apply_warp: bool = True,
    hide_raw: bool = True
):
    """Applies the calculated shift to all layers and creates a global canvas."""
    viewer = napari.current_viewer()

    shift_list = session.get_data("shift")
    shift = np.array(shift_list) if shift_list else None
    
    base_output_dir = session.get_data("output_dir")
    output_dir = Path(base_output_dir) / "aligned"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if shift is None:
        viewer.status = "No shift calculated. Run Registration first."
        return
        
    estimated_z_expansion = abs(shift[0])
    if estimated_z_expansion > 1000:
        viewer.status = "Error: Shift too large (OOM Protection)"
        popups.show_error_popup(viewer.window._qt_window, "Shift Too Large", f"ABORTING: Calculated Z-shift ({shift[0]:.2f}) is dangerously large and would crash the application.")
        return

    viewer.status = f"Generating Canvas with Shift: {shift}"
    
    # Extract layer data to pass to pipeline
    r1_layers_data = []
    r2_layers_data = []
    for l in viewer.layers:
        if isinstance(l, (napari.layers.Image, napari.layers.Labels)):
            layer_info = {
                'name': l.name, 
                'data': l.data, 
                'colormap': l.colormap.name if hasattr(l, 'colormap') and l.colormap else None, 
                'scale': l.scale,
                'is_label': isinstance(l, napari.layers.Labels)
            }
            if "R1" in l.name:
                r1_layers_data.append(layer_info)
            elif "R2" in l.name:
                r2_layers_data.append(layer_info)

    # Run pipeline with the new progress dialog
    with popups.ProgressDialog(viewer.window._qt_window, title="Generating Global Canvas") as dialog:
        pipeline_generator = generate_global_canvas(r1_layers_data, r2_layers_data, shift, output_dir, apply_warp)

        for progress, message, result in pipeline_generator:
            dialog.update_progress(progress, message)
            
            if result:
                for layer_data in [result['r1'], result['r2']]:
                    if layer_data.get('is_label'):
                        data = layer_data['data']
                        if not np.issubdtype(data.dtype, np.integer):
                            data = data.astype(np.uint32)
                        viewer.add_labels(data, name=layer_data['name'], scale=layer_data['scale'], opacity=0.6)
                    else:
                        viewer.add_image(
                            layer_data['data'], 
                            name=layer_data['name'], 
                            colormap=layer_data['colormap'], 
                            scale=layer_data['scale'], 
                            blending='additive'
                        )

    if hide_raw:
        for layer in viewer.layers:
            if "Aligned" in layer.name or "Warped" in layer.name or "Consensus" in layer.name:
                layer.visible = True
            else:
                layer.visible = False

    viewer.status = "Global Canvas Generation Complete."
