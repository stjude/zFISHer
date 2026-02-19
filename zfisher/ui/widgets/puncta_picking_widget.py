import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt

from .. import style

# Import the child widgets that this composite widget will manage
from .puncta_widget import puncta_widget
from .puncta_editor_widget import puncta_editor_widget

class PunctaPickingWidget(widgets.Container):
    """
    A widget to hold tools for puncta detection and editing,
    with collapsible sections for Algorithmic and Manual workflows.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        # Store references to child widgets
        self.algorithmic_widget = puncta_widget
        self.manual_widget = puncta_editor_widget

        # Get the native QWidget's layout and remove any default margins/spacing.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Algorithmic Section ---
        # This QWidget will act as a container for a nested QToolBox.
        algorithmic_outer_container = QWidget()
        algorithmic_layout = QVBoxLayout(algorithmic_outer_container)
        algorithmic_layout.setContentsMargins(0, 0, 0, 0)
        algorithmic_layout.setSpacing(0)

        # Create the inner, nested QToolBox for the algorithmic steps.
        algorithmic_toolbox = QToolBox()
        algorithmic_toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # Add the puncta detection widget as a collapsible item.
        algorithmic_toolbox.addItem(self.algorithmic_widget.native, "Puncta Detection")
        algorithmic_layout.addWidget(algorithmic_toolbox)
        self.toolbox.addItem(algorithmic_outer_container, "Algorithmic")

        # --- Manual Section ---
        # This QWidget will act as a container for a nested QToolBox.
        manual_outer_container = QWidget()
        manual_layout = QVBoxLayout(manual_outer_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(0)

        # Create the inner, nested QToolBox for the manual steps.
        manual_toolbox = QToolBox()
        manual_toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # Add the puncta editor widget as a collapsible item.
        manual_toolbox.addItem(self.manual_widget.native, "Puncta Editor")
        manual_layout.addWidget(manual_toolbox)
        self.toolbox.addItem(manual_outer_container, "Manual")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """A placeholder reset_choices method to satisfy the viewer loop."""
        if hasattr(self.algorithmic_widget, "reset_choices"):
            self.algorithmic_widget.reset_choices()
        if hasattr(self.manual_widget, "reset_choices"):
            self.manual_widget.reset_choices()