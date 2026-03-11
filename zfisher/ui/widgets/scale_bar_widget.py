import math
from qtpy.QtWidgets import QWidget
from qtpy.QtGui import QPainter
from qtpy.QtCore import Qt, QPoint, QEvent
from .. import style


class DraggableScaleBar(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.dragging = False
        self.drag_start_position = QPoint()
        self.locked = False
        self.show_pixels = False
        self.pen_color = style.SCALE_BAR_PEN_COLOR
        self.font_color = style.SCALE_BAR_FONT_COLOR
        self.font = style.SCALE_BAR_FONT
        self.resize(200, 60)

        if self.parent():
            self.parent().installEventFilter(self)

        self.viewer.camera.events.zoom.connect(self.on_zoom)
        self.viewer.layers.events.inserted.connect(self.on_layer_change)
        self.viewer.layers.events.removed.connect(self.on_layer_change)

        self.pixel_size_um = 1.0
        self.bar_length_um = 10
        self.bar_length_px = 100
        self.text = ""
        self.recalculate()

    def eventFilter(self, watched, event):
        if watched == self.parent() and event.type() == QEvent.Resize:
            self.move_to_bottom_right()
        return super().eventFilter(watched, event)

    def move_to_bottom_right(self):
        self.adjustSize()
        parent = self.parent()
        if parent:
            p_w, p_h = parent.width(), parent.height()
            if p_w > 0 and p_h > 0:
                self.move(p_w - self.width() - 20, p_h - self.height() - 20)

    def get_pixel_size(self):
        return 1.0

    def recalculate(self):
        self.pixel_size_um = self.get_pixel_size()
        self.on_zoom()

    def on_layer_change(self, event=None):
        self.recalculate()

    def on_zoom(self, event=None):
        zoom = self.viewer.camera.zoom
        if zoom == 0: return
        target_px = 150

        active_layer = self.viewer.layers.selection.active
        if active_layer:
             pixel_size_x = active_layer.scale[-1]
        else:
             pixel_size_x = 1.0

        um_per_canvas_px = pixel_size_x / zoom if pixel_size_x > 0 else 1.0 / zoom

        target_um = target_px * um_per_canvas_px
        if target_um <= 0: return

        exponent = math.floor(math.log10(target_um))
        fraction = target_um / (10 ** exponent)

        if fraction < 1.5: nice_fraction = 1
        elif fraction < 3.5: nice_fraction = 2
        elif fraction < 7.5: nice_fraction = 5
        else: nice_fraction = 10

        self.bar_length_um = nice_fraction * (10 ** exponent)
        self.bar_length_px = self.bar_length_um / um_per_canvas_px
        self.text = f"{self.bar_length_um:.4g} um"
        if self.show_pixels:
            self.text += f" ({int(self.bar_length_px)} px)"
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(self.font_color)
        painter.setFont(self.font)
        rect = self.rect()
        text_rect = painter.boundingRect(rect, Qt.AlignHCenter | Qt.AlignTop, self.text)
        total_h = text_rect.height() + 5 + 6
        start_y = (rect.height() - total_h) / 2
        painter.drawText(rect.left(), int(start_y), rect.width(), text_rect.height(), Qt.AlignHCenter, self.text)
        bar_y = start_y + text_rect.height() + 5
        start_x = (rect.width() - self.bar_length_px) / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.pen_color)
        painter.drawRect(int(start_x), int(bar_y), int(self.bar_length_px), 6)

    def mousePressEvent(self, event):
        if self.locked: return
        if event.button() == Qt.RightButton:
            self.dragging = True
            self.drag_start_position = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            self.move(self.mapToParent(event.pos() - self.drag_start_position))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.dragging = False
