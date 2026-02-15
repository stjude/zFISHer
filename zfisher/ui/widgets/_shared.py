import napari
from pathlib import Path

from ...core import io
from .. import viewer_helpers

def load_raw_data_into_viewer(viewer, round1_path, round2_path, output_dir=None, progress_callback=None):
    """
    Orchestrates loading raw image data (ND2/TIFF), converting it, and adding it to the viewer.
    """
    # Setup input storage directory
    input_storage_dir = None
    if output_dir:
        input_storage_dir = Path(output_dir) / "input"
        input_storage_dir.mkdir(parents=True, exist_ok=True)

    all_paths = [(Path(round1_path), "R1"), (Path(round2_path), "R2")]
    steps_per_file = 3 # Load, Convert, Add to viewer
    total_steps = len(all_paths) * steps_per_file

    for i, (path, prefix) in enumerate(all_paths):
        base_progress = i * steps_per_file

        if not path.exists():
            print(f"Error: {path} not found.")
            if progress_callback:
                progress_callback(int(((base_progress + steps_per_file) / total_steps) * 100), f"Not found: {path.name}")
            continue

        if progress_callback:
            progress_callback(int(((base_progress + 0) / total_steps) * 100), f"Loading {prefix}: {path.name}...")

        # 1. Core I/O Logic
        image_session = io.load_image_session(path)

        # 2. Conversion/Processing Logic
        if path.suffix.lower() == '.nd2' and input_storage_dir:
            def conversion_progress(msg):
                if progress_callback:
                    progress_callback(int(((base_progress + 1) / total_steps) * 100), msg)
            io.convert_nd2_to_ome(image_session, input_storage_dir, prefix, conversion_progress)

        # 3. UI Logic
        if progress_callback:
            progress_callback(int(((base_progress + 2) / total_steps) * 100), f"Adding {prefix} layers to viewer...")
        viewer_helpers.add_image_session_to_viewer(viewer, image_session, prefix)

    # Force the Z-slider to appear
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.reset_view()
