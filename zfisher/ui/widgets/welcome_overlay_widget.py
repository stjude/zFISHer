from qtpy.QtWidgets import QWidget, QLabel, QVBoxLayout
from qtpy.QtGui import QPainter, QPalette
from qtpy.QtCore import Qt, QTimer
from .. import style


class WelcomeOverlay(QWidget):
    """A solid black splash screen that manages its own visibility."""
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer

        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, style.WELCOME_WIDGET_BG_COLOR)
        self.setPalette(palette)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignCenter)

        label_html = (
            f"<h1 {style.WELCOME_WIDGET_STYLE['h1']}>zFISHer</h1>"
            f"<p {style.WELCOME_WIDGET_STYLE['p']}>Version 2.0</p>"
        )
        self.label = QLabel(label_html)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.label)

        self.viewer.layers.events.inserted.connect(self._check_visibility)
        self.viewer.layers.events.removed.connect(self._check_visibility)

        if self.parent():
            self.parent().installEventFilter(self)
            QTimer.singleShot(100, lambda: (self.resize_to_parent(), self._check_visibility()))

    def _check_visibility(self, event=None):
        if len(self.viewer.layers) > 0:
            self.hide()
            self.setEnabled(False)
        else:
            self.show()
            self.raise_()
            self.setEnabled(True)

    def paintEvent(self, event):
        if self.isVisible():
            painter = QPainter(self)
            painter.fillRect(self.rect(), style.WELCOME_WIDGET_BG_COLOR)

    def eventFilter(self, obj, event):
        if obj is self.parent() and event.type() == event.Resize:
            self.resize_to_parent()
        return super().eventFilter(obj, event)

    def resize_to_parent(self):
        if not self.parent(): return
        self.resize(self.parent().size())
        self.move(0, 0)
