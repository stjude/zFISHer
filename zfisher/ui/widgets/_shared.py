import napari
import numpy as np
import tifffile
import json
from pathlib import Path

from zfisher.core.io import load_nd2
from ..constants import CHANNEL_COLORS

# Global to track the zoom listener
_current_scale_updater = None

class TiffSession:
    """Helper to wrap TIFF data with an interface similar to the ND2 loader."""
    def __init__(self, path):
        self.ome_metadata = None
        self.data = self._load_data(str(path))
        # Default metadata (generic names and 1.0 scale)
        self.channels = [f"Ch{i+1}" for i in range(self.data.shape[1])]
        self.voxels = (1.0, 1.0, 1.0) # (dz, dy, dx)

    def _load_data(self, path):
        with tifffile.TiffFile(path) as tif:
            data = tif.asarray()
            self.ome_metadata = tif.ome_metadata

        # Normalize to (Z, C, Y, X)
        if data.ndim == 2:   # (Y, X) -> (1, 1, Y, X)
            return data[np.newaxis, np.newaxis, :, :]
        elif data.ndim == 3: # (Z, Y, X) -> (Z, 1, Y, X)
            return data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            # Heuristic: If dim0 < dim1, assume (C, Z, Y, X) and swap to (Z, C, Y, X)
            # Otherwise assume it is already (Z, C, Y, X)
            if data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, 1)
            return data
        return data

def load_raw_data_into_viewer(viewer, round1_path, round2_path, output_dir=None, progress_callback=None):
    """Helper function to load and display the raw ND2 image data."""
    global _current_scale_updater

    # Setup input storage directory
    input_storage_dir = None
    if output_dir:
        input_storage_dir = Path(output_dir) / "input"
        input_storage_dir.mkdir(parents=True, exist_ok=True)

    all_paths = [
        (Path(round1_path), "R1"), 
        (Path(round2_path), "R2")
    ]
    
    # Increased granularity for progress bar:
    # 1. Load ND2
    # 2. Start Conversion
    # 3. Save OME-TIFF
    # 4. Save Metadata
    # 5. Add to Viewer
    steps_per_file = 5
    total_steps = len(all_paths) * steps_per_file

    for i, (path, prefix) in enumerate(all_paths):
        base_progress = i * steps_per_file
        
        if not path.exists():
            print(f"Error: {path} not found.")
            if progress_callback:
                progress_callback(
                    int(((base_progress + steps_per_file) / total_steps) * 100), 
                    f"Not found: {path.name}"
                )
            continue
        
        if progress_callback:
            progress_callback(
                int(((base_progress + 0) / total_steps) * 100), 
                f"Loading {prefix}: {path.name}..."
            )
            
        if path.suffix.lower() == '.nd2':
            nd2_session = load_nd2(str(path))
            
            # Convert to OME-TIFF and generate OME-XML
            if input_storage_dir:
                if progress_callback:
                    progress_callback(
                        int(((base_progress + 1) / total_steps) * 100), 
                        f"Converting {prefix} to OME-TIFF..."
                    )
                try:
                    ome_tif_path = input_storage_dir / f"{prefix}_converted.ome.tif"
                    print(f"Converting {path.name} to OME-TIFF...")
                    
                    # Construct basic OME metadata for the writer
                    # Data is (Z, C, Y, X), Voxels are (dz, dy, dx)
                    metadata = {
                        'axes': 'ZCYX',
                        'Channel': {'Name': nd2_session.channels},
                        'PhysicalSizeX': nd2_session.voxels[2],
                        'PhysicalSizeXUnit': 'µm',
                        'PhysicalSizeY': nd2_session.voxels[1],
                        'PhysicalSizeYUnit': 'µm',
                        'PhysicalSizeZ': nd2_session.voxels[0],
                        'PhysicalSizeZUnit': 'µm',
                    }
                    
                    if progress_callback:
                        progress_callback(
                            int(((base_progress + 2) / total_steps) * 100), 
                            f"Saving OME-TIFF for {prefix}..."
                        )
                    
                    tifffile.imwrite(ome_tif_path, nd2_session.data, metadata=metadata, photometric='minisblack')
                    print(f"Saved OME-TIFF to {ome_tif_path}")

                    if progress_callback:
                        progress_callback(
                            int(((base_progress + 3) / total_steps) * 100), 
                            f"Extracting OME-XML for {prefix}..."
                        )

                    # Extract the generated OME-XML
                    with tifffile.TiffFile(ome_tif_path) as tf:
                        if tf.ome_metadata:
                            xml_path = input_storage_dir / f"{prefix}_metadata.ome.xml"
                            with open(xml_path, "w", encoding="utf-8") as f:
                                f.write(tf.ome_metadata)
                            print(f"Exported OME-XML metadata to {xml_path}")
                except Exception as e:
                    print(f"Failed to convert ND2 to OME-TIFF: {e}")

            # Export ND2 metadata as JSON if available
            if input_storage_dir and hasattr(nd2_session, 'metadata') and nd2_session.metadata:
                try:
                    json_path = input_storage_dir / f"{prefix}_metadata.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(nd2_session.metadata, f, indent=4, default=str)
                    print(f"Exported ND2 metadata to {json_path}")
                except Exception as e:
                    print(f"Failed to export ND2 metadata for {path.name}: {e}")
        else:
            nd2_session = TiffSession(path)
            # Export OME-XML metadata if available and output_dir is set
            if input_storage_dir:
                if nd2_session.ome_metadata:
                    try:
                        xml_path = input_storage_dir / f"{prefix}_metadata.ome.xml"
                        with open(xml_path, "w", encoding="utf-8") as f:
                            f.write(nd2_session.ome_metadata)
                        print(f"Exported OME-XML metadata to {xml_path}")
                    except Exception as e:
                        print(f"Failed to export metadata for {path.name}: {e}")
                else:
                    print(f"No OME-XML metadata found in {path.name}")
        
        # YOUR DATA SHAPE: (Z, C, Y, X) -> NAPARI EXPECTS (C, Z, Y, X)
        data_swapped = np.moveaxis(nd2_session.data, 1, 0)
        
        print(f"Loaded {prefix}: {data_swapped.shape[0]} channels, {data_swapped.shape[1]} Z-slices. Full shape: {data_swapped.shape}")
        
        if progress_callback:
            progress_callback(
                int(((base_progress + 4) / total_steps) * 100), 
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
