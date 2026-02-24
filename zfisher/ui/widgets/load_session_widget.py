from magicgui.widgets import Container, PushButton, FileEdit, Label
import napari
import numpy as np
from pathlib import Path

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import error_handler
from ._shared import load_raw_data_into_viewer

class LoadSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the load session functionality."""
        self._header = Label(value="Load Session")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Load a previously saved zFISHer session.</i>")
        self._load_session_file = FileEdit(label="Session File (.json)", filter="*.json")
        self._load_session_btn = PushButton(text="Load Session")

    def _init_layout(self):
        """Arranges all widgets in the container."""
        self.extend([
            self._header,
            self._info,
            self._load_session_file,
            self._load_session_btn,
        ])

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._load_session_btn.clicked.connect(self._on_load_session)

    @error_handler("Load Session Failed")
    def _on_load_session(self):
        session_file = self._load_session_file.value
        if not session_file or not Path(session_file).exists() or Path(session_file).is_dir():
            self._viewer.status = "Error: Invalid session file selected."
            return

        with popups.ProgressDialog(self._viewer.window._qt_window, "Loading Session...") as dialog:
            session.set_loading(True)
            try:
                self._viewer.layers.clear()
                
                dialog.update_progress(10, "Loading session file...")
                session.load_session_file(session_file)
                
                processed_files = session.get_data("processed_files", default={})
                
                # FIX: Check if we already have aligned/warped data to prevent Z-padding bloat
                has_aligned_data = any("Aligned" in key or "Warped" in key for key in processed_files.keys())

                r1_path = session.get_data("r1_path")
                r2_path = session.get_data("r2_path")
                output_dir = session.get_data("output_dir")

                # STEP 1: Load Raw Data ONLY if no aligned data exists
                if r1_path and r2_path and not has_aligned_data:
                    def raw_progress(p, text):
                        dialog.update_progress(10 + int(p * 0.5), f"Raw: {text}")
                    load_raw_data_into_viewer(self._viewer, r1_path, r2_path, output_dir=output_dir, progress_callback=raw_progress)
                else:
                    dialog.update_progress(60, "Skipping raw reload, jumping to aligned data...")

                # STEP 2: Restore Processed Layers (The Aligned 70-slice files)
                if processed_files:
                    # Get the scale from the first available image layer or default
                    raw_img_layer = next((l for l in self._viewer.layers if isinstance(l, napari.layers.Image)), None)
                    scale = raw_img_layer.scale if raw_img_layer else (1.0, 1.0, 1.0)
                    sanitized_scale = tuple(s if isinstance(s, (int, float)) and s > 0 and not np.isnan(s) else 1.0 for s in scale)

                    def processed_progress(p, text):
                        dialog.update_progress(60 + int(p * 0.35), f"Restoring: {text}")
                    
                    # This ensures your 5,492 spots and 73 nuclei load correctly
                    viewer_helpers.restore_processed_layers(
                        self._viewer, 
                        processed_files, 
                        sanitized_scale, 
                        progress_callback=processed_progress
                    )
                
                dialog.update_progress(95, "Finalizing UI...")
                self._viewer.status = "Session Restored."

                if hasattr(self._viewer.window, 'custom_scale_bar'):
                    self._viewer.window.custom_scale_bar.show()
                
                dialog.update_progress(100, "Done.")
            finally:
                session.set_loading(False)