import logging
from magicgui.widgets import Container, PushButton, FileEdit, Label
import napari
import numpy as np
from pathlib import Path
from qtpy.QtWidgets import QFrame

logger = logging.getLogger(__name__)

from ...core import session
from .. import popups, viewer_helpers
from ..decorators import error_handler
from ._shared import load_raw_data_into_viewer
from ..style import COLORS
from .colocalization_widget import refresh_rules_display

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
        self._info.native.setObjectName("widgetInfo")
        self._load_session_file = FileEdit(label="Session File:", filter="*.json", tooltip="Path to a zfisher_session_x.json file from a previous project.")
        self._load_session_btn = PushButton(text="Load Session")
        self._load_session_btn.tooltip = "Restore the selected session and reload all processed layers."

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        """Arranges all widgets in the container using native layout."""
        from qtpy.QtWidgets import QLabel, QSpacerItem, QSizePolicy

        self._form = Container(labels=True)
        self._form.extend([self._load_session_file])
        self._form.native.layout().setContentsMargins(0, 10, 0, 20)

        self._desc = QLabel(
            'Load a zFISHer session from a previous project. Session files are '
            'found in a project\'s output directory and are numbered variants of '
            'a "zfisher_session_x.json" file.'
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
        _layout.addWidget(self._form.native)
        _layout.addSpacerItem(spacer())
        _layout.addWidget(self._load_session_btn.native)
        _layout.addStretch(1)

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
            from .. import viewer as viewer_module
            viewer_module._suppress_custom_controls = True
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

                dialog.freeze_canvas()

                # STEP 1: Load Raw Data ONLY if no aligned data exists
                if r1_path and r2_path and not has_aligned_data:
                    def raw_progress(p, text):
                        dialog.update_progress(10 + int(p * 0.5), f"Raw: {text}")
                    load_raw_data_into_viewer(self._viewer, r1_path, r2_path, output_dir=output_dir, progress_callback=raw_progress)
                else:
                    dialog.update_progress(60, "Skipping raw reload, jumping to aligned data...")

                # STEP 2: Restore Processed Layers (The Aligned 70-slice files)
                if processed_files:
                    # Get scale and offset from the session if saved by headless run, otherwise fallback.
                    canvas_scale = session.get_data("canvas_scale")
                    canvas_offset_pixels = session.get_data("canvas_offset_pixels")
                    logger.debug("load_session: canvas_scale=%s, canvas_offset_pixels=%s", canvas_scale, canvas_offset_pixels)

                    if canvas_scale:
                        scale = tuple(canvas_scale)
                        logger.debug("load_session: using scale from session: %s", scale)
                    else:
                        # Fallback for older sessions: get scale from a raw layer if present.
                        logger.debug("load_session: 'canvas_scale' not found, falling back to default.")
                        raw_img_layer = next((l for l in self._viewer.layers if isinstance(l, napari.layers.Image)), None)
                        scale = raw_img_layer.scale if raw_img_layer else (1.0, 1.0, 1.0)
                        logger.debug("load_session: using fallback scale: %s", scale)

                    sanitized_scale = tuple(s if isinstance(s, (int, float)) and s > 0 and not np.isnan(s) else 1.0 for s in scale)

                    def processed_progress(p, text):
                        dialog.update_progress(60 + int(p * 0.35), f"Restoring: {text}")
                    
                    viewer_helpers.restore_processed_layers(
                        self._viewer,
                        processed_files,
                        sanitized_scale,
                        canvas_offset_pixels,
                        progress_callback=processed_progress
                    )
                
                dialog.update_progress(95, "Finalizing UI...")
                refresh_rules_display()

                # Lock puncta layers if session has proceeded past warping
                from .. import events as _events
                from ... import constants as _constants
                has_warped = any(
                    _constants.ALIGNED_PREFIX in name or _constants.WARPED_PREFIX in name
                    for name in processed_files.keys()
                )
                if has_warped:
                    for layer in self._viewer.layers:
                        if isinstance(layer, napari.layers.Points) and _constants.PUNCTA_SUFFIX in layer.name:
                            _events.lock_layer(layer)

                self._viewer.status = "Session Restored."

                if hasattr(self._viewer.window, 'custom_scale_bar'):
                    self._viewer.window.custom_scale_bar.show()
                
                dialog.update_progress(100, "Done.")
            finally:
                session.set_loading(False)
        # Reset suppression AFTER the dialog closes — the dialog's __exit__
        # calls processEvents() which would trigger custom control popups
        # if suppression were already cleared.
        viewer_module._suppress_custom_controls = False
        # Trigger custom layer controls for the currently selected layer
        try:
            self._viewer.layers.selection.events.changed(added=set(), removed=set())
        except Exception:
            pass