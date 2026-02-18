from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari
import os
import logging
import numpy as np
from napari.qt.threading import thread_worker

from ...core import session, pipeline, io
from .. import popups, viewer_helpers, style
from ..decorators import error_handler
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

        # --- Separator 2 ---
        self._separator2 = Container(labels=False)
        self._separator2.native.setObjectName("separator")

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

        # --- "Batch Process" Widgets ---
        self._batch_header = Label(value="Batch Process Directory")
        self._batch_header.native.setObjectName("batchSessionHeader")
        self._batch_container = Container()
        self._batch_input_dir = FileEdit(label="Input Directory", mode='d')
        self._batch_output_dir = FileEdit(label="Base Output Dir", mode='d', value=Path.home() / "zFISHer_Batch_Output")
        self._batch_run_btn = PushButton(text="Run Batch Processing")

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
        self._batch_container.extend([
            self._batch_input_dir,
            self._batch_output_dir,
            self._batch_run_btn,
        ])
        self.extend([
            self._new_session_header,
            self._new_session_container,
            self._separator,
            self._load_session_header,
            self._load_session_container,
            self._separator2,
            self._batch_header,
            self._batch_container,
        ])
        # Apply stylesheet to separators
        self._separator.native.setStyleSheet(style.SEPARATOR_STYLESHEET)
        self._separator2.native.setStyleSheet(style.SEPARATOR_STYLESHEET)

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._new_session_btn.clicked.connect(self._on_new_session)
        self._load_session_btn.clicked.connect(self._on_load_session)
        self._batch_run_btn.clicked.connect(self._on_batch_run)

    def _validate_input_files(self, r1_path, r2_path):
        """Checks if files exist, are files, and are readable."""
        error_messages = []
        for path, name in [(r1_path, "Round 1"), (r2_path, "Round 2")]:
            if not path.is_file():
                error_messages.append(f"• {name} file does not exist or is a directory:\n  {path}")
            elif not os.access(path, os.R_OK):
                error_messages.append(f"• {name} file is not readable (check permissions):\n  {path}")

        if error_messages:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Invalid Input Files",
                "Please correct the following issues:\n\n" + "\n\n".join(error_messages)
            )
            return False
        return True

    @error_handler("New Session Failed")
    def _on_new_session(self):
        r1_val = self._round1_path.value
        r2_val = self._round2_path.value
        out_val = self._output_dir.value

        if not self._validate_input_files(r1_val, r2_val):
            return

        # Use a progress dialog to wrap the core call
        with popups.ProgressDialog(self._viewer.window._qt_window, title="Initializing...") as dialog:
            # Call the refactored core function
            success = session.initialize_new_session(
                out_val, r1_val, r2_val, 
                progress_callback=lambda p, t: dialog.update_progress(p, t)
            )
            
            if not success:
                popups.show_error_popup(self._viewer.window._qt_window, "Session Exists", "...")
                return

            # UI-Specific Task: Only load into viewer if we are in the UI
            self._viewer.layers.clear()
            load_raw_data_into_viewer(
                self._viewer, 
                session.get_data("r1_path"), 
                session.get_data("r2_path"),
                output_dir=session.get_data("output_dir"),
                progress_callback=lambda p, t: dialog.update_progress(p, t)
            )

    @error_handler("Load Session Failed")
    def _on_load_session(self):
        session_file = self._load_session_file.value
        if not session_file.exists() or session_file.is_dir():
            if session_file.is_dir():
                self._viewer.status = "Error: Please select a session file, not a directory."
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
                processed_files = session.get_data("processed_files", default={})
                if processed_files:
                    # Find a raw data layer to get the canonical scale.
                    # A raw layer does not have "Aligned" or "Warped" in its name.
                    raw_img_layer = next((
                        l for l in self._viewer.layers 
                        if isinstance(l, napari.layers.Image) and 
                        "Aligned" not in l.name and "Warped" not in l.name
                    ), None)
                    
                    scale = raw_img_layer.scale if raw_img_layer else (1.0, 1.0, 1.0)
                    
                    # Sanitize the scale one last time to be safe against zeros or NaNs
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

    @error_handler("Batch Processing Failed")
    def _on_batch_run(self):
        input_dir = self._batch_input_dir.value
        output_base_dir = self._batch_output_dir.value

        if not input_dir.is_dir() or not output_base_dir:
            popups.show_error_popup(self._viewer.window._qt_window, "Invalid Directories", "Please select valid input and output directories.")
            return

        output_base_dir.mkdir(parents=True, exist_ok=True)

        @thread_worker(connect={"returned": self._on_batch_finished, "yielded": self._on_batch_progress})
        def run_batch_pipeline():
            file_pairs = io.discover_nd2_pairs(input_dir)
            num_pairs = len(file_pairs)
            if num_pairs == 0:
                yield "No file pairs found to process."
                return "No pairs found."

            for i, pair in enumerate(file_pairs):
                fov_name = pair['name']
                fov_output = output_base_dir / fov_name
                
                yield f"Processing {i+1}/{num_pairs}: {fov_name}"
                
                try:
                    # For now, we pass empty params. This could be expanded later.
                    pipeline.run_full_zfisher_pipeline(
                        r1_path=pair['r1'],
                        r2_path=pair['r2'],
                        output_dir=fov_output,
                        params={}
                    )
                except Exception as e:
                    logging.error(f"Failed to process {fov_name}: {str(e)}")
                    yield f"ERROR processing {fov_name}: {e}"
            
            return f"Batch processing complete for {num_pairs} pairs."

        self._viewer.status = "Starting batch processing..."
        worker = run_batch_pipeline()
        worker.start()

    def _on_batch_progress(self, message: str):
        self._viewer.status = message

    def _on_batch_finished(self, result_message: str):
        self._viewer.status = result_message
        popups.show_info_popup(self._viewer.window._qt_window, "Batch Processing Complete", result_message)
