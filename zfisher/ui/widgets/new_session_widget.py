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
        self._info = Label(value="<i>Create a new session by loading Round 1 and Round 2 image files.</i>")
        self._info.native.setObjectName("widgetInfo")
        self._round1_path = FileEdit(label="R1:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path(r"..."), tooltip="Path to Round 1 .nd2 or OME-TIFF image file.")
        self._round2_path = FileEdit(label="R2:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path(r"..."), tooltip="Path to Round 2 .nd2 or OME-TIFF image file.")
     #   self._round1_path = FileEdit(label="Round 1", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R1_micro.tif"))
     #   self._round2_path = FileEdit(label="Round 2", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R2_micro.tif"))

        self._output_dir = FileEdit(label="Output:", mode="d", value=Path.home() / "zFISHer_Output", tooltip="Directory where all session files and results will be saved.")
        self._new_session_btn = PushButton(text="Start New Session")
        self._new_session_btn.tooltip = "Load the input files and initialize a new zFISHer session."

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        """Arranges all widgets in the container using native layout."""
        from qtpy.QtWidgets import QLabel, QSpacerItem, QSizePolicy

        # Wrap each FileEdit in its own labelled container
        self._r1_form = Container(labels=True)
        self._r1_form.extend([self._round1_path])
        self._r1_form.native.layout().setContentsMargins(0, 10, 0, 0)
        self._r2_form = Container(labels=True)
        self._r2_form.extend([self._round2_path])
        self._r2_form.native.layout().setContentsMargins(0, 0, 0, 0)
        self._out_form = Container(labels=True)
        self._out_form.extend([self._output_dir])
        self._out_form.native.layout().setContentsMargins(0, 10, 0, 20)

        self._desc = QLabel(
            "Define input R1 and R2 .nd2 or OME-TIFF files. "
            "R2 will be registered and warped to R1. "
            "All project files will be stored in defined output directory."
        )
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: white; margin: 4px 2px;")

        spacer = lambda: QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)

        _layout = self.native.layout()
        _layout.setSpacing(2)
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._desc)
        _layout.addSpacerItem(spacer())
        _layout.addWidget(self._r1_form.native)
        _layout.addWidget(self._r2_form.native)
        _layout.addSpacerItem(spacer())
        _layout.addWidget(self._out_form.native)
        _layout.addSpacerItem(spacer())
        _layout.addWidget(self._new_session_btn.native)
        _layout.addStretch(1)

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

            if hasattr(self._viewer.window, 'custom_scale_bar'):
                self._viewer.window.custom_scale_bar.show()

