import logging
import warnings

from qtpy.QtWidgets import (
    QProgressDialog, QMessageBox, QApplication,
    QDialog, QVBoxLayout, QLabel, QProgressBar
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QColor, QPainterPath

# Saved state for napari notification suppression
_saved_napari_handlers = []
_saved_showwarning = None


def _suppress_napari_notifications():
    """Aggressively suppress napari notification toasts.

    1. Disable the notification_manager (if the attribute exists).
    2. Remove any napari-related logging handlers from the root logger
       so that log messages don't trigger notification toasts.
    3. Replace warnings.showwarning with a no-op so that even if
       catch_warnings doesn't fully block napari's hook, nothing shows.
    """
    global _saved_napari_handlers, _saved_showwarning

    # 1. Disable notification manager
    try:
        from napari.utils.notifications import notification_manager
        notification_manager.records = []
        notification_manager.enabled = False
    except Exception:
        pass

    # 2. Temporarily remove napari notification handlers from root logger
    root = logging.getLogger()
    _saved_napari_handlers = []
    for h in root.handlers[:]:
        mod = getattr(type(h), '__module__', '') or ''
        cls = type(h).__name__
        if 'napari' in mod or 'Notification' in cls:
            root.removeHandler(h)
            _saved_napari_handlers.append(h)

    # 3. Replace showwarning with a no-op
    _saved_showwarning = warnings.showwarning
    warnings.showwarning = lambda *a, **kw: None


def _restore_napari_notifications():
    """Re-enable napari notifications and restore removed handlers."""
    global _saved_napari_handlers, _saved_showwarning

    try:
        from napari.utils.notifications import notification_manager
        notification_manager.enabled = True
    except Exception:
        pass

    # Restore napari logging handlers
    root = logging.getLogger()
    for h in _saved_napari_handlers:
        root.addHandler(h)
    _saved_napari_handlers = []

    # Restore showwarning
    if _saved_showwarning is not None:
        warnings.showwarning = _saved_showwarning
        _saved_showwarning = None


class ProgressDialog(QProgressDialog):
    """
    A custom, non-cancellable QProgressDialog that provides a simple interface
    for updating progress and status text.

    Suppresses Python warnings and napari notification toasts while active
    so that third-party library chatter doesn't spawn popup windows.

    Usage:
        with ProgressDialog(parent, title="Work") as dialog:
            dialog.update_progress(25, "Step 1...")
            ...
            dialog.update_progress(100, "Done.")
    """
    def __init__(self, parent=None, title="Processing...", text="Please wait..."):
        super().__init__(text, None, 0, 100, parent)
        self._canvas_frozen = False
        self._warnings_ctx = None
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        # Frameless window — no title bar, no close button, matches macOS look.
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowModal)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Style child widgets only; background is painted in paintEvent.
        self.setStyleSheet("""
            QLabel {
                color: white;
                background: transparent;
            }
            QProgressBar {
                border: 1px solid #7a6b8a;
                border-radius: 6px;
                background-color: #251f2e;
                text-align: center;
                color: white;
                min-height: 18px;
            }
            QProgressBar::chunk {
                background-color: #4aa87c;
                border-radius: 5px;
            }
        """)
        self.setContentsMargins(14, 14, 14, 14)
        self.setMinimumDuration(0)  # Show immediately
        self.setCancelButton(None)  # No cancel button
        self.setValue(0)
        self.show()
        QApplication.processEvents()

    def paintEvent(self, event):
        """Draw a rounded, bordered background behind the dialog contents."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(self.width()), float(self.height()), 12.0, 12.0)
        painter.fillPath(path, QColor("#1a1421"))
        painter.setPen(QColor("#7a6b8a"))
        painter.drawPath(path)
        painter.end()

    def update_progress(self, value: int, text: str = None):
        """Updates the progress bar and optionally the label text."""
        if text:
            self.setLabelText(text)
        self.setValue(value)
        if not self._canvas_frozen:
            QApplication.processEvents()  # Ensure UI updates are visible

    def freeze_canvas(self):
        """Suppress processEvents during layer mutations to prevent vispy
        GL draws on partially-constructed layer state.

        Call this before adding/removing layers. While frozen,
        update_progress() will still update the dialog text and value
        but will NOT call processEvents(), so vispy never gets a
        chance to draw with stale GL handles.  Automatically unfrozen
        when the dialog closes via __exit__.
        """
        self._canvas_frozen = True

    def unfreeze_canvas(self):
        """Re-enable processEvents and force a single clean redraw."""
        if self._canvas_frozen:
            self._canvas_frozen = False
            QApplication.processEvents()

    def __enter__(self):
        # Suppress Python warnings and napari notification toasts
        self._warnings_ctx = warnings.catch_warnings()
        self._warnings_ctx.__enter__()
        warnings.simplefilter("ignore")
        _suppress_napari_notifications()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unfreeze_canvas()
        self.close()
        # Restore warnings and notifications
        if self._warnings_ctx:
            self._warnings_ctx.__exit__(None, None, None)
            self._warnings_ctx = None
        _restore_napari_notifications()

class BatchProgressDialog(QDialog):
    """
    A non-cancellable dialog with two progress bars:
    one for overall batch progress and one for the current item's pipeline progress.
    """
    def __init__(self, parent=None, title="Batch Processing...", text="Please wait..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(450)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(self)

        self._batch_label = QLabel("Batch Progress")
        layout.addWidget(self._batch_label)
        self._batch_bar = QProgressBar()
        self._batch_bar.setRange(0, 100)
        self._batch_bar.setValue(0)
        layout.addWidget(self._batch_bar)

        self._item_label = QLabel(text)
        layout.addWidget(self._item_label)
        self._item_bar = QProgressBar()
        self._item_bar.setRange(0, 100)
        self._item_bar.setValue(0)
        layout.addWidget(self._item_bar)

        self.show()
        QApplication.processEvents()

    def update_batch_progress(self, value: int, text: str = None):
        """Updates the overall batch progress bar."""
        if text:
            self._batch_label.setText(text)
        self._batch_bar.setValue(value)
        QApplication.processEvents()

    def update_item_progress(self, value: int, text: str = None):
        """Updates the current item's pipeline progress bar."""
        if text:
            self._item_label.setText(text)
        self._item_bar.setValue(value)
        QApplication.processEvents()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def select_nuclear_channel(parent, channels):
    """
    Shows a dialog asking the user to pick the nuclear stain channel.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog.
    channels : list[str]
        Available channel names.

    Returns
    -------
    str or None
        The selected channel name, or None if the user cancels.
    """
    from qtpy.QtWidgets import QInputDialog
    choice, ok = QInputDialog.getItem(
        parent,
        "Select Nuclear Channel",
        "No known nuclear stain detected.\nPlease select the nuclear channel:",
        channels,
        0,
        False,
    )
    return choice if ok else None


def show_info_popup(parent, title, text):
    """Shows a simple informational message box (e.g., for success)."""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Information)
    msg.exec_()

def show_error_popup(parent, title, text):
    """Shows a simple error message box."""
    msg = QMessageBox(parent)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QMessageBox.Critical)
    msg.exec_()