import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt

from .. import style

class ExportVisualizationWidget(widgets.Container):
    """
    A widget to hold tools for exporting data and capturing views,
    with collapsible sections for Export and Capture.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        # Get the native QWidget's layout and remove any default margins/spacing.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Export Section ---
        export_content = widgets.Container(labels=False)
        export_content.append(widgets.Label(value="Export controls will go here."))
        self.toolbox.addItem(export_content.native, "Export")

        # --- Capture Section ---
        capture_content = widgets.Container(labels=False)
        capture_content.append(widgets.Label(value="Capture controls will go here."))
        self.toolbox.addItem(capture_content.native, "Capture")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """A placeholder reset_choices method to satisfy the viewer loop."""
        pass