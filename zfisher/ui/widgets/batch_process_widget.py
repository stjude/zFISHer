from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari
import logging
from napari.qt.threading import thread_worker

from ...core import pipeline, io
from .. import popups
from ..decorators import error_handler

class BatchProcessWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the batch processing functionality."""
        self._header = Label(value="Batch Process")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Run the full pipeline on multiple fields of view.</i>")
        self._batch_input_dir = FileEdit(label="Input Directory", mode='d')
        self._batch_output_dir = FileEdit(label="Base Output Dir", mode='d', value=Path.home() / "zFISHer_Batch_Output")
        self._batch_run_btn = PushButton(text="Run Batch Processing")

    def _init_layout(self):
        """Arranges all widgets in the container."""
        self.extend([
            self._header,
            self._info,
            self._batch_input_dir,
            self._batch_output_dir,
            self._batch_run_btn,
        ])

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._batch_run_btn.clicked.connect(self._on_batch_run)

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
                    pipeline.run_full_zfisher_pipeline(r1_path=pair['r1'], r2_path=pair['r2'], output_dir=fov_output, params={})
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