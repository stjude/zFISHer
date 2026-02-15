from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari

from zfisher.core import session
from .. import popups, viewer_helpers
from ._shared import load_raw_data_into_viewer

class StartSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        # The main container uses a simple vertical layout (labels=False).
        # This allows headers to span the full width.
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the session starter."""
        # --- Headers & Separator ---
        self._new_session_header = Label(value="New Session")
        self._new_session_header.native.setObjectName("newSessionHeader")

        self._load_session_header = Label(value="Load Previous Session")
        self._load_session_header.native.setObjectName("loadSessionHeader")

        self._separator = Container(labels=False)
        self._separator.native.setObjectName("separator")

        # --- "New Session" Widgets ---
        self._new_session_container = Container()
        self._round1_path = FileEdit(label="Round 1", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2"))
        self._round2_path = FileEdit(label="Round 2", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2"))
        self._output_dir = FileEdit(label="Output Directory", mode="d", value=Path.home() / "zFISHer_Output")
        self._new_session_btn = PushButton(text="New Session")

        # --- "Load Session" Widgets ---
        self._load_session_container = Container()
        self._load_session_file = FileEdit(label="Session File (.json)", filter="*.json")
        self._load_session_btn = PushButton(text="Load Session")

    def _init_layout(self):
        """Arranges all widgets in the container."""
        self._new_session_container.extend([
            self._round1_path,
            self._round2_path,
            self._output_dir,
            self._new_session_btn,
        ])
        self._load_session_container.extend([
            self._load_session_file,
            self._load_session_btn,
        ])
        self.extend([
            self._new_session_header,
            self._new_session_container,
            self._separator,
            self._load_session_header,
            self._load_session_container,
        ])

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._new_session_btn.clicked.connect(self._on_new_session)
        self._load_session_btn.clicked.connect(self._on_load_session)

    def _on_new_session(self):
        # Get paths from widgets to initialize the session
        round1_path_val = self._round1_path.value
        round2_path_val = self._round2_path.value
        output_dir_val = self._output_dir.value

        if not session.initialize_new_session(output_dir_val, round1_path_val, round2_path_val):
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Session Already Exists",
                f"""A session already exists in this directory.

{output_dir_val}

Please choose a different output directory, or use the 'Load Session' button to continue your previous analysis."""
            )
            return

        self._viewer.layers.clear()

        # Now that session is initialized, get paths from the session (single source of truth)
        r1_path = session.get_data("r1_path")
        r2_path = session.get_data("r2_path")
        output_dir = session.get_data("output_dir")

        with popups.ProgressDialog(self._viewer.window._qt_window, title="Loading Data...") as dialog:
            load_raw_data_into_viewer(
                self._viewer,
                r1_path,
                r2_path,
                output_dir=output_dir,
                progress_callback=lambda p, t: dialog.update_progress(p, t)
            )
            dialog.update_progress(100, "Done.")

        if hasattr(self._viewer.window, 'custom_scale_bar'):
            self._viewer.window.custom_scale_bar.show()
            self._viewer.window.custom_scale_bar.move_to_bottom_right()

    def _on_load_session(self):
        session_file = self._load_session_file.value
        if not session_file.exists() or session_file.is_dir():
            if session_file.is_dir():
                self._viewer.status = "Error: Please select a session file, not a directory."
            return

        dialog = popups.ProgressDialog(self._viewer.window._qt_window, "Loading Session...")
        session.set_loading(True)
        try:
            self._viewer.layers.clear()
            
            dialog.update_progress(10, "Loading session file...")
            session.load_session_file(session_file)
            
            shift = session.get_data("shift")
            if shift:
                print(f"Restored Shift: {shift}")

            # --- Load Raw Data ---
            r1_path = session.get_data("r1_path")
            r2_path = session.get_data("r2_path")
            output_dir = session.get_data("output_dir")
            if r1_path and r2_path:
                def raw_progress(p, text):
                    scaled_progress = 10 + int(p * 0.6) 
                    dialog.update_progress(scaled_progress, text)

                load_raw_data_into_viewer(
                    self._viewer, 
                    r1_path, 
                    r2_path,
                    output_dir=output_dir,
                    progress_callback=raw_progress
                )

            # --- Restore Processed Layers ---
            processed_files = session.get_data("processed_files", {})
            if processed_files:
                scale = next((layer.scale for layer in self._viewer.layers if isinstance(layer, napari.layers.Image)), (1, 1, 1))
                def processed_progress(p, text):
                    scaled_progress = 70 + int(p * 0.25)
                    dialog.update_progress(scaled_progress, text)
                viewer_helpers.restore_processed_layers(self._viewer, processed_files, scale, progress_callback=processed_progress)
            
            dialog.update_progress(95, "Finalizing...")
            self._viewer.status = "Session Restored."

            if hasattr(self._viewer.window, 'custom_scale_bar'):
                self._viewer.window.custom_scale_bar.show()
                self._viewer.window.custom_scale_bar.move_to_bottom_right()
            
            dialog.update_progress(100, "Done.")

        finally:
            session.set_loading(False)
            dialog.close()
