import napari
from pathlib import Path
from magicgui import magicgui
from zfisher.core import session
from .. import popups
from ._shared import load_raw_data_into_viewer

# Define your paths as constants at the top for easy editing later
DEFAULT_R1 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2")
DEFAULT_R2 = Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2")

@magicgui(
    call_button="Load Data",
    round1_path={"label": "Round 1 (.nd2)", "filter": "*.nd2"},
    round2_path={"label": "Round 2 (.nd2)", "filter": "*.nd2"},
    output_dir={"label": "Output Directory", "mode": "d"},
    auto_call=False,
)
def file_selector_widget(
    round1_path: Path = DEFAULT_R1,
    round2_path: Path = DEFAULT_R2,
    output_dir: Path = Path.home() / "zFISHer_Output",
):
    """Loads ND2 files, sets up output directories, and initializes a new session."""
    viewer = napari.current_viewer()

    # Prevent overwriting an existing session
    session_file = output_dir / "zfisher_session.json"
    if session_file.exists():
        popups.show_error_popup(
            viewer.window._qt_window,
            "Session Already Exists",
            f"A session already exists in this directory.\n\n"
            f"{output_dir}\n\n"
            f"Please choose a different output directory, or use the 'Load Session' "
            f"widget to continue your previous analysis."
        )
        return
    
    # 1. Setup Output Directories
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    (output_dir / "segmentation").mkdir(exist_ok=True)
    (output_dir / "aligned").mkdir(exist_ok=True)
    
    # This widget always starts a new session.
    viewer.layers.clear()
    session.clear_session()
    session.update_data("output_dir", str(output_dir))
    session.update_data("r1_path", str(round1_path))
    session.update_data("r2_path", str(round2_path))
    session.save_session()
    
    # Load the raw data into the viewer
    with popups.ProgressDialog(viewer.window._qt_window, title="Loading Data...") as dialog:
        load_raw_data_into_viewer(
            viewer, 
            round1_path, 
            round2_path,
            progress_callback=lambda p, t: dialog.update_progress(p, t)
        )
        dialog.update_progress(100, "Done.")
