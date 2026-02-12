import napari
import numpy as np
from pathlib import Path

from zfisher.core.io import load_nd2
from ..constants import CHANNEL_COLORS

# Global to track the zoom listener
_current_scale_updater = None

def load_raw_data_into_viewer(viewer, round1_path, round2_path, progress_callback=None):
    """Helper function to load and display the raw ND2 image data."""
    global _current_scale_updater

    all_paths = [
        (Path(round1_path), "R1"), 
        (Path(round2_path), "R2")
    ]
    
    total_steps = len(all_paths) * 2 + 1 # load + process for each

    for i, (path, prefix) in enumerate(all_paths):
        if not path.exists():
            print(f"Error: {path} not found.")
            if progress_callback:
                progress_callback(
                    int(((i * 2 + 2) / total_steps) * 100), 
                    f"Not found: {path.name}"
                )
            continue
        
        if progress_callback:
            progress_callback(
                int(((i * 2) / total_steps) * 100), 
                f"Loading {prefix}: {path.name}..."
            )
            
        nd2_session = load_nd2(str(path))
        
        # YOUR DATA SHAPE: (Z, C, Y, X) -> NAPARI EXPECTS (C, Z, Y, X)
        data_swapped = np.moveaxis(nd2_session.data, 1, 0)
        
        print(f"Loaded {prefix}: {data_swapped.shape[0]} channels, {data_swapped.shape[1]} Z-slices. Full shape: {data_swapped.shape}")
        
        if progress_callback:
            progress_callback(
                int(((i * 2 + 1) / total_steps) * 100), 
                f"Adding {prefix} layers to viewer..."
            )
        
        new_layers = viewer.add_image(
            data_swapped,
            name=[f"{prefix} - {ch}" for ch in nd2_session.channels],
            channel_axis=0,
            scale=nd2_session.voxels,
            blending="additive"
        )

        # Apply colors
        for layer in new_layers:
            for ch_name, color in CHANNEL_COLORS.items():
                if ch_name.upper() in layer.name.upper():
                    layer.colormap = color
            
            if "DAPI" not in layer.name.upper():
                layer.visible = False

        # Update Text Overlay with Pixel Info
        # voxels is (dz, dy, dx)
        dz, dy, dx = nd2_session.voxels
        
        def update_scale_text(event=None):
            try:
                # Calculate Field of View (FOV) width
                width = viewer.canvas.size[1]
                zoom = viewer.camera.zoom
                if zoom > 0:
                    fov_px = width / zoom
                    fov_um = fov_px * dx
                    px_per_um = 1.0 / dx if dx > 0 else 0
                    viewer.text_overlay.text = (
                        f"Voxel Size: {dx:.4f} x {dy:.4f} x {dz:.4f} um\n"
                        f"Scale: 1 um = {px_per_um:.2f} px\n"
                        f"FOV Width: {fov_um:.1f} um ({int(fov_px)} px)"
                    )
            except Exception:
                pass

        # Prevent duplicate listeners if loading multiple files
        if _current_scale_updater is not None:
            try: viewer.camera.events.zoom.disconnect(_current_scale_updater)
            except: pass
        _current_scale_updater = update_scale_text
        viewer.camera.events.zoom.connect(update_scale_text)
        
        viewer.text_overlay.visible = True
        viewer.text_overlay.color = "white"
        viewer.text_overlay.font_size = 10
        viewer.text_overlay.position = "top_right"
        
        update_scale_text()

    # Force the Z-slider to appear
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.reset_view()
