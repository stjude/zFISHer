import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout

from .new_session_widget import NewSessionWidget
from .load_session_widget import LoadSessionWidget
from .batch_process_widget import BatchProcessWidget
from .. import style

class StartSessionWidget(widgets.Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self.viewer = viewer

        # Store references to child widgets
        self.new_session_widget = NewSessionWidget(viewer)
        self.load_session_widget = LoadSessionWidget(viewer)
        self.batch_process_widget = BatchProcessWidget(viewer)

        # Get the native QWidget's layout and remove any default margins/spacing.
        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create a QToolBox to get the collapsible/accordion style
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        # --- Create containers for each section to hold the magicgui widgets ---
        
        # New Session
        self.toolbox.addItem(self.new_session_widget.native, "New Session")

        # Load Session
        self.toolbox.addItem(self.load_session_widget.native, "Load Previous Session")

        # Batch Process
        self.toolbox.addItem(self.batch_process_widget.native, "Batch Process")

        # Ensure QToolBox internal scroll areas allow content to resize with panel
        from qtpy.QtWidgets import QScrollArea
        for sa in self.toolbox.findChildren(QScrollArea):
            sa.setWidgetResizable(True)

        # Add the toolbox to the layout of this magicgui container's native widget
        layout.addWidget(self.toolbox)

    def reset_choices(self):
        """Passes the `reset_choices` call to all child widgets if they have it."""
        for widget in [self.new_session_widget, self.load_session_widget, self.batch_process_widget]:
            if hasattr(widget, "reset_choices"):
                widget.reset_choices()
