from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari
import os

from qtpy.QtWidgets import QFrame

from ...core import session
from .. import popups
from ..decorators import error_handler
from ._shared import load_raw_data_into_viewer
from ..style import COLORS

class NewSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the new session functionality."""
        self._header = Label(value="New Session")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Start a new zFISHer session from raw data.</i>")
        self._info.native.setObjectName("widgetInfo")
        self._round1_path = FileEdit(label="R1:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path(r"..."))
        self._round2_path = FileEdit(label="R2:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path(r"..."))
     #   self._round1_path = FileEdit(label="Round 1", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R1_micro.tif"))
     #   self._round2_path = FileEdit(label="Round 2", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R2_micro.tif"))

        self._output_dir = FileEdit(label="Output:", mode="d", value=Path.home() / "zFISHer_Output")
        self._new_session_btn = PushButton(text="Start New Session")

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        """Arranges all widgets in the container using native layout."""
        # Form container with labels for the file inputs
        self._form = Container(labels=True)
        self._form.extend([self._round1_path, self._round2_path, self._output_dir])

        _layout = self.native.layout()
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._form.native)
        _layout.addWidget(self._new_session_btn.native)

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._new_session_btn.clicked.connect(self._on_new_session)

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

        with popups.ProgressDialog(self._viewer.window._qt_window, title="Initializing...") as dialog:
            success = session.initialize_new_session(out_val, r1_val, r2_val, progress_callback=dialog.update_progress)
            if not success:
                popups.show_error_popup(self._viewer.window._qt_window, "Session Exists", "A session already exists in the selected output directory. Please choose a different directory or load the existing session.")
                return

            self._viewer.layers.clear()
            load_raw_data_into_viewer(self._viewer, session.get_data("r1_path"), session.get_data("r2_path"), output_dir=session.get_data("output_dir"), progress_callback=dialog.update_progress)

