import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui

from ...core import session, registration # Importing from core
from .. import popups
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Generate Global Canvas",
    layout="vertical",
    apply_warp={"label": "Apply Deformable Warping?"},
    hide_raw={"label": "Hide Raw Layers?"}
)
@require_active_session("Please start or load a session before generating the canvas.")
@error_handler("Canvas Generation Failed")
def canvas_widget(
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
        
        # Call the core logic. Note: No more 'yield' here; we use a callback.
        results = registration.generate_global_canvas(
            r1_layers_data, 
            r2_layers_data, 
            shift, 
            output_dir, 
            apply_warp=apply_warp,
            progress_callback=lambda p, m: dialog.update_progress(p, m)
        )

        # 6. Add resulting layers back to napari
        for layer_info in results:
            layer_type = layer_info['type']
            meta = layer_info['meta']
            
            if layer_type == 'labels':
                viewer.add_labels(
                    layer_info['data'], 
                    name=layer_info['name'], 
                    scale=meta['scale'], 
                    opacity=0.6
                )
            elif layer_type == 'image':
                viewer.add_image(
                    layer_info['data'], 
                    name=layer_info['name'], 
                    colormap=meta.get('colormap', 'gray'), 
                    scale=meta['scale'], 
                    blending='additive'
                )
            elif layer_type == 'vectors':
                viewer.add_vectors(
                    layer_info['data'],
                    name=layer_info['name'],
                    scale=meta['scale'],
                    edge_width=0.2,
                    length=2.5,
                    edge_color='cyan'
                )

    # 7. Final UI Tidy Up
    if hide_raw:
        for layer in viewer.layers:
            # Show the new results, hide the raw rounds
            layer.visible = any(x in layer.name for x in ["Aligned", "Warped", "Consensus"])

    viewer.status = "Global Canvas Generation Complete."