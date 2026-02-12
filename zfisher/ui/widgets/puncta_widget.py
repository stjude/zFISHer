import napari
import numpy as np
from magicgui import magicgui, widgets

from zfisher.core.segmentation import detect_spots_3d
from .. import popups

@magicgui(
    call_button="Detect Puncta",
    image_layer={"label": "Target Channel"},
    nuclei_layer={"label": "Nuclei Masks (Optional)"},
    method={"label": "Algorithm", "choices": ["Local Maxima", "Laplacian of Gaussian"]},
    threshold={"label": "Sensitivity (0-1)", "min": 0.01, "max": 1.0, "step": 0.01},
    min_distance={"label": "Min Distance (px)", "min": 1, "max": 20, "step": 1},
    sigma={"label": "Spot Radius (Sigma)", "min": 0.0, "max": 5.0, "step": 0.1}
)
def puncta_widget(
    image_layer: "napari.layers.Image",
    nuclei_layer: "napari.layers.Labels",
    method: str = "Local Maxima",
    threshold: float = 0.05,
    min_distance: int = 2,
    sigma: float = 0.0
):
    """Detects spots in the selected image layer."""
    viewer = napari.current_viewer()
    if image_layer is None:
        return
        
    viewer.status = f"Detecting spots in {image_layer.name}..."
    dialog = popups.ProgressDialog(viewer.window._qt_window, f"Detecting Puncta in {image_layer.name}...")
    
    try:
        # Run detection
        coords = detect_spots_3d(image_layer.data, threshold_rel=threshold, min_distance=min_distance, sigma=sigma, method=method)
        
        layer_name = f"{image_layer.name}_puncta"
        
        if layer_name in viewer.layers:
            pts_layer = viewer.layers[layer_name]
            if len(coords) > 0:
                if len(pts_layer.data) > 0:
                    combined = np.vstack((pts_layer.data, coords))
                    pts_layer.data = np.unique(combined, axis=0)
                else:
                    pts_layer.data = coords
            
            pts_layer.properties = {'id': np.arange(len(pts_layer.data)) + 1}
            pts_layer.text = {'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, 5, 5])}
        else:
            properties = {'id': np.arange(len(coords)) + 1}
            text_params = {'string': '{id}', 'size': 8, 'color': 'white', 'translation': np.array([0, 5, 5])}
            
            pts_layer = viewer.add_points(
                coords,
                name=layer_name,
                size=3,
                face_color="yellow",
                scale=image_layer.scale,
                properties=properties,
                text=text_params
            )
            # The main event handler in events.py will now attach the listener.
            pts_layer.events.data(value=pts_layer.data)

        msg = f"Found {len(coords)} spots."
        print(msg)
        viewer.status = msg
    finally:
        dialog.close()

# Add editing tools to the Puncta Widget
edit_chk = widgets.CheckBox(text="Edit Mode")
clear_btn = widgets.PushButton(text="Clear All Puncta")
puncta_widget.append(widgets.Label(value="<b>Editing Tools:</b>"))
puncta_widget.append(edit_chk)
puncta_widget.append(clear_btn)

@edit_chk.changed.connect
def _on_edit_puncta(state: bool):
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            layer = viewer.layers[p_name]
            viewer.layers.selection.active = layer
            if state:
                layer.mode = 'select'
                viewer.status = f"Editing {p_name}. Select points and press Backspace/Delete to remove."
            else:
                layer.mode = 'pan_zoom'
                viewer.status = f"Stopped editing {p_name}."
        else:
            viewer.status = f"Layer {p_name} not found. Run detection first."
            edit_chk.value = False

@clear_btn.clicked.connect
def _on_clear_puncta():
    viewer = napari.current_viewer()
    img_layer = puncta_widget.image_layer.value
    if img_layer:
        p_name = f"{img_layer.name}_puncta"
        if p_name in viewer.layers:
            viewer.layers[p_name].data = np.empty((0, 3))
            viewer.status = f"Cleared all points in {p_name}."
