from magicgui.widgets import Container, PushButton, FileEdit
import napari
import numpy as np

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import error_handler
from ._shared import load_raw_data_into_viewer

class LoadSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the load session functionality."""
        self._load_session_file = FileEdit(label="Session File (.json)", filter="*.json")
        self._load_session_btn = PushButton(text="Load Session")

    def _init_layout(self):
        """Arranges all widgets in the container."""
        self.extend([
            self._load_session_file,
            self._load_session_btn,
        ])

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._load_session_btn.clicked.connect(self._on_load_session)

    @error_handler("Load Session Failed")
    def _on_load_session(self):
        session_file = self._load_session_file.value
        if not session_file.exists() or session_file.is_dir():
            if session_file.is_dir():
                self._viewer.status = "Error: Please select a session file, not a directory."
            else:
                self._viewer.status = f"Error: Session file not found at {session_file}"
            return

        with popups.ProgressDialog(self._viewer.window._qt_window, "Loading Session...") as dialog:
            session.set_loading(True)
            try:
                self._viewer.layers.clear()
                
                dialog.update_progress(10, "Loading session file...")
                session.load_session_file(session_file)
                
                shift = session.get_data("shift")
                if shift:
                    print(f"Restored Shift: {shift}")

                r1_path = session.get_data("r1_path")
                r2_path = session.get_data("r2_path")
                output_dir = session.get_data("output_dir")
                if r1_path and r2_path:
                    def raw_progress(p, text):
                        scaled_progress = 10 + int(p * 0.6) 
                        dialog.update_progress(scaled_progress, text)
                    load_raw_data_into_viewer(self._viewer, r1_path, r2_path, output_dir=output_dir, progress_callback=raw_progress)

                processed_files = session.get_data("processed_files", default={})
                if processed_files:
                    raw_img_layer = next((l for l in self._viewer.layers if isinstance(l, napari.layers.Image) and "Aligned" not in l.name and "Warped" not in l.name), None)
                    scale = raw_img_layer.scale if raw_img_layer else (1.0, 1.0, 1.0)
                    sanitized_scale = tuple(s if isinstance(s, (int, float)) and s > 0 and not np.isnan(s) else 1.0 for s in scale)

                    def processed_progress(p, text):
                        scaled_progress = 70 + int(p * 0.25)
                        dialog.update_progress(scaled_progress, text)
                    viewer_helpers.restore_processed_layers(self._viewer, processed_files, sanitized_scale, progress_callback=processed_progress)
                
                dialog.update_progress(95, "Finalizing...")
                self._viewer.status = "Session Restored."

                if hasattr(self._viewer.window, 'custom_scale_bar'):
                    self._viewer.window.custom_scale_bar.show()
                    self._viewer.window.custom_scale_bar.move_to_bottom_right()
                
                dialog.update_progress(100, "Done.")
            finally:
                session.set_loading(False)