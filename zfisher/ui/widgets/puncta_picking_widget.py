import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt

from .. import style

class PunctaPickingWidget(widgets.Container):
    """
    A widget to hold tools for puncta detection and editing,
    with collapsible sections for Algorithmic and Manual workflows.
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

        # --- Algorithmic Section ---
        algorithmic_content = widgets.Container(labels=False)
        algorithmic_content.append(widgets.Label(value="Algorithmic puncta picking controls will go here."))
        self.toolbox.addItem(algorithmic_content.native, "Algorithmic")

        # --- Manual Section ---
        manual_content = widgets.Container(labels=False)
        manual_content.append(widgets.Label(value="Manual puncta picking controls will go here."))
        self.toolbox.addItem(manual_content.native, "Manual")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """A placeholder reset_choices method to satisfy the viewer loop."""
        pass