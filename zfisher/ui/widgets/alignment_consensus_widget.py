import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox
from qtpy.QtCore import Qt

from .. import style

class AlignmentConsensusWidget(widgets.Container):
    """
    A widget to hold tools for alignment and consensus mask generation,
    with collapsible sections for Automated and Manual workflows.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        # Get the native QWidget's layout and remove any default margins/spacing.
        # This ensures the nested QToolBox fills the entire area, matching the parent.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Automated Section ---
        automated_container = widgets.Container(labels=False)
        automated_label = widgets.Label(value="Automated processing controls will go here.")
        automated_label.native.setAlignment(Qt.AlignCenter)
        automated_container.append(automated_label)
        self.toolbox.addItem(automated_container.native, "Automated")

        # --- Manual Section ---
        manual_container = widgets.Container(labels=False)
        manual_label = widgets.Label(value="Manual processing controls will go here.")
        manual_label.native.setAlignment(Qt.AlignCenter)
        manual_container.append(manual_label)
        self.toolbox.addItem(manual_container.native, "Manual")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """A placeholder reset_choices method to satisfy the viewer loop."""
        pass