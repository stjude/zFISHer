import napari
from magicgui import widgets
from qtpy.QtWidgets import QToolBox, QWidget, QVBoxLayout

from .dapi_segmentation_widget import dapi_segmentation_widget
from .mask_editor_widget import mask_editor_widget

from .. import style


class NucleiSegmentationWidget(widgets.Container):
    """
    Step 2: Nuclei Segmentation.
    Contains DAPI mapping and mask editing tools so users can
    generate and refine per-round nuclear masks before puncta picking.
    """
    def __init__(self, viewer: "napari.Viewer"):
        super().__init__(labels=False)
        self.viewer = viewer

        self.dapi_widget = dapi_segmentation_widget
        self.mask_editor_widget = mask_editor_widget

        layout = self.native.layout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet(style.NESTED_TOOLBOX_STYLESHEET)

        self.toolbox.addItem(self.dapi_widget.native, "DAPI Mapping")
        self.toolbox.addItem(self.mask_editor_widget.native, "Mask Editor")

        layout.addWidget(self.toolbox)

    def reset_choices(self):
        for widget in [self.dapi_widget, self.mask_editor_widget]:
            if hasattr(widget, "reset_choices"):
                widget.reset_choices()
