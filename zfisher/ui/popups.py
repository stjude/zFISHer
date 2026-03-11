from qtpy.QtWidgets import (
    QProgressDialog, QMessageBox, QApplication,
    QDialog, QVBoxLayout, QLabel, QProgressBar
)
from qtpy.QtCore import Qt

class ProgressDialog(QProgressDialog):
    """
    A custom, non-cancellable QProgressDialog that provides a simple interface 
    for updating progress and status text.

    Usage:
        with ProgressDialog(parent, title="Work") as dialog:
            dialog.update_progress(25, "Step 1...")
            ...
            dialog.update_progress(100, "Done.")
    """
    def __init__(self, parent=None, title="Processing...", text="Please wait..."):
        super().__init__(text, None, 0, 100, parent)
        self._canvas_frozen = False
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)  # Show immediately
        self.setCancelButton(None)  # No cancel button
        self.setValue(0)
        self.show()
        QApplication.processEvents()

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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unfreeze_canvas()
        self.close()

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