import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout
from qtpy.QtCore import Qt

# Import the child widgets that this composite widget will manage
from .automated_preprocessing_widget import automated_preprocessing_widget
from .registration_widget import registration_widget
from .canvas_widget import canvas_widget
from .nuclei_matching_widget import nuclei_matching_widget

from .. import style

class AlignmentConsensusWidget(widgets.Container):
    """
    A widget to hold tools for alignment and consensus mask generation,
    with collapsible sections for Automated and Manual workflows.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        # Store references to the child widgets. This allows the main viewer.py
        # to access them for event handling (e.g., auto-selecting layers).
        self.automated_widget = automated_preprocessing_widget
        self.registration_widget = registration_widget
        self.canvas_widget = canvas_widget
        self.nuclei_matching_widget = nuclei_matching_widget

        # Get the native QWidget's layout and remove any default margins/spacing.
        # This ensures the nested QToolBox fills the entire area, matching the parent.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Automated Section --- (Now contains the actual widget)
        self.toolbox.addItem(self.automated_widget.native, "Automated")

        # --- Manual Section ---
        # This QWidget will act as a container for a nested QToolBox.
        manual_outer_container = QWidget()
        manual_layout = QVBoxLayout(manual_outer_container)
        manual_layout.setContentsMargins(0, 0, 0, 0)
        manual_layout.setSpacing(0)

        # Create the inner, nested QToolBox for the manual steps.
        manual_toolbox = QToolBox()
        manual_toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # Add each manual step as a separate, collapsible item.
        manual_toolbox.addItem(self.registration_widget.native, "Registration")
        manual_toolbox.addItem(self.canvas_widget.native, "Global Canvas")
        manual_toolbox.addItem(self.nuclei_matching_widget.native, "Match Nuclei")

        manual_layout.addWidget(manual_toolbox)
        self.toolbox.addItem(manual_outer_container, "Manual")

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """Passes the `reset_choices` call to all child widgets."""
        for widget in [
            self.automated_widget, self.registration_widget,
            self.canvas_widget, self.nuclei_matching_widget
        ]:
            if hasattr(widget, "reset_choices"):
                widget.reset_choices()
