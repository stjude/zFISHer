import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt

from .. import style

# Import the child widget
from .colocalization_widget import colocalization_widget
from .capture_widget import capture_widget

class ExportVisualizationWidget(widgets.Container):
    """
    A widget to hold tools for exporting data and capturing views,
    with collapsible sections for Export and Capture.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        # Store reference to child widget
        self.export_widget = colocalization_widget
        self.capture_widget = capture_widget

        # Get the native QWidget's layout and remove any default margins/spacing.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Export Section ---
        # This QWidget will act as a container for a nested QToolBox.
        export_outer_container = QWidget()
        export_layout = QVBoxLayout(export_outer_container)
        export_layout.setContentsMargins(0, 0, 0, 0)
        export_layout.setSpacing(0)

        # Create the inner, nested QToolBox for the export steps.
        export_toolbox = QToolBox()
        export_toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # Add the colocalization widget as a collapsible item.
        export_toolbox.addItem(self.export_widget.native, "Colocalization & Export")
        export_layout.addWidget(export_toolbox)
        self.toolbox.addItem(export_outer_container, "Export")

        # --- Capture Section ---
        # This QWidget will act as a container for a nested QToolBox.
        capture_outer_container = QWidget()
        capture_layout = QVBoxLayout(capture_outer_container)
        capture_layout.setContentsMargins(0, 0, 0, 0)
        capture_layout.setSpacing(0)

        # Create the inner, nested QToolBox for the capture steps.
        capture_toolbox = QToolBox()
        capture_toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # Add the capture widget as a collapsible item.
        capture_toolbox.addItem(self.capture_widget.native, "Capture View")
        capture_layout.addWidget(capture_toolbox)
        self.toolbox.addItem(capture_outer_container, "Capture")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """A placeholder reset_choices method to satisfy the viewer loop."""
        if hasattr(self.export_widget, "reset_choices"):
            self.export_widget.reset_choices()
        if hasattr(self.capture_widget, "reset_choices"):
            self.capture_widget.reset_choices()