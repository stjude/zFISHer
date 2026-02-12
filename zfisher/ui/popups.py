from qtpy.QtWidgets import QProgressDialog, QMessageBox, QApplication
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
        QApplication.processEvents()  # Ensure UI updates are visible
    
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