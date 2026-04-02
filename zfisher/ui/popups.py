import logging
import warnings

from qtpy.QtWidgets import (
    QApplication,
    QDialog, QVBoxLayout, QLabel, QProgressBar, QWidget
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QPainter, QColor, QPainterPath

# Saved state for napari notification suppression
_saved_napari_handlers = []
_saved_showwarning = None
_stray_popup_suppressing = False


def _install_stray_popup_filter():
    """Monkey-patch napari widgets that flash as top-level windows.

    During layer-control construction, napari creates QColorSwatchEdit and
    QtWrappedLabel without a parent, causing them to briefly appear as
    top-level windows.  We patch their setHidden/setVisible so that while
    ``_stray_popup_suppressing`` is True, they stay hidden.
    """
    # Patch QColorSwatchEdit — the main offender
    try:
        from napari._qt.widgets.qt_color_swatch import QColorSwatchEdit
        if not hasattr(QColorSwatchEdit, '_orig_setHidden'):
            QColorSwatchEdit._orig_setHidden = QColorSwatchEdit.setHidden
            def _patched_setHidden(self, hidden):
                if not hidden and _stray_popup_suppressing and self.isWindow():
                    return  # block setHidden(False) while suppressing
                QColorSwatchEdit._orig_setHidden(self, hidden)
            QColorSwatchEdit.setHidden = _patched_setHidden
    except ImportError:
        pass

    # Patch QtWrappedLabel
    try:
        from napari._qt.layer_controls.widgets.qt_widget_controls_base import QtWrappedLabel
        if not hasattr(QtWrappedLabel, '_orig_setHidden'):
            QtWrappedLabel._orig_setHidden = QtWrappedLabel.setHidden
            def _patched_setHidden_label(self, hidden):
                if not hidden and _stray_popup_suppressing and self.isWindow():
                    return
                QtWrappedLabel._orig_setHidden(self, hidden)
            QtWrappedLabel.setHidden = _patched_setHidden_label
    except ImportError:
        pass


def _suppress_napari_notifications():
    """Aggressively suppress napari notification toasts.

    1. Block the notification_ready signal so no toasts are dispatched.
    2. Remove any napari-related logging handlers from the root logger
       so that log messages don't trigger notification toasts.
    3. Replace warnings.showwarning with a no-op so that even if
       catch_warnings doesn't fully block napari's hook, nothing shows.
    4. Dismiss any already-visible notification toast widgets.
    """
    global _saved_napari_handlers, _saved_showwarning

    # 1. Block the notification signal (napari 0.6+ has no 'enabled' flag)
    try:
        from napari.utils.notifications import notification_manager
        notification_manager.records = []
        notification_manager.notification_ready.block()
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

    # 4. Close any existing napari notification toast widgets
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification
        for w in QApplication.topLevelWidgets():
            if isinstance(w, NapariQtNotification):
                w.close()
    except Exception:
        pass

    # 5. Install a global event filter to block stray napari widget popups
    global _stray_popup_suppressing
    _stray_popup_suppressing = True
    _install_stray_popup_filter()


def _restore_napari_notifications():
    """Re-enable napari notifications and restore removed handlers."""
    global _saved_napari_handlers, _saved_showwarning, _stray_popup_suppressing
    _stray_popup_suppressing = False

    # Dismiss any toast widgets that appeared despite suppression
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification
        for w in QApplication.topLevelWidgets():
            if isinstance(w, NapariQtNotification):
                w.close()
    except Exception:
        pass

    try:
        from napari.utils.notifications import notification_manager
        # Clear any records that accumulated while blocked so they
        # don't appear as toasts the moment we unblock.
        notification_manager.records = []
        notification_manager.notification_ready.unblock()
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


class _DimOverlay(QWidget):
    """Semi-transparent dark overlay covering the parent window during loading."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        if parent:
            self.setGeometry(parent.rect())
        self.show()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        painter.end()

    def resizeToParent(self):
        if self.parent():
            self.setGeometry(self.parent().rect())


class ProgressDialog(QDialog):
    """
    A custom, non-cancellable progress dialog that provides a simple interface
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
        super().__init__(parent)
        self._canvas_frozen = False
        self._warnings_ctx = None
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowModal)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMinimumWidth(320)

        _pad = 28
        layout = QVBoxLayout(self)
        layout.setContentsMargins(_pad, _pad, _pad, _pad)
        layout.setSpacing(10)

        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setStyleSheet("""
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
        layout.addWidget(self._bar)

        # Create dim overlay behind the dialog
        self._overlay = None
        par = self.parent()
        if par is not None:
            self._overlay = _DimOverlay(par)
            self._overlay.raise_()
        self.show()
        self.raise_()
        self._center_on_parent()
        QApplication.processEvents()

    def _center_on_parent(self):
        """Position the dialog at the center of its parent widget."""
        parent = self.parent()
        if parent is not None:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)

    def paintEvent(self, event):
        """Draw a rounded, bordered background behind the dialog contents."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(self.width()), float(self.height()), 20.0, 20.0)
        painter.fillPath(path, QColor("#1a1421"))
        painter.setPen(QColor("#7a6b8a"))
        painter.drawPath(path)
        painter.end()

    def setLabelText(self, text):
        """Compatibility shim for code that calls setLabelText."""
        self._label.setText(text)

    def setValue(self, value):
        """Compatibility shim for code that calls setValue."""
        self._bar.setValue(value)

    def update_progress(self, value: int, text: str = None):
        """Updates the progress bar and optionally the label text.

        Pass value=-1 to switch to indeterminate (bouncing) mode.
        Any value >= 0 switches back to determinate mode.
        """
        if text:
            self._label.setText(text)
        if value < 0:
            if self._bar.maximum() != 0:
                self._bar.setRange(0, 0)  # indeterminate
        else:
            if self._bar.maximum() == 0:
                self._bar.setRange(0, 100)  # back to determinate
            self._bar.setValue(value)
        if not self._canvas_frozen:
            QApplication.processEvents()

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
        # Remove dim overlay
        if self._overlay is not None:
            self._overlay.close()
            self._overlay.deleteLater()
            self._overlay = None
        # Flush any remaining Qt events while notifications are still suppressed
        # so that deferred napari toasts from layer mutations don't appear.
        QApplication.processEvents()
        # Now safe to restore warnings and notifications
        if self._warnings_ctx:
            self._warnings_ctx.__exit__(None, None, None)
            self._warnings_ctx = None
        _restore_napari_notifications()
        # Flush again and close any toasts that slipped through during restore
        QApplication.processEvents()
        try:
            from napari._qt.dialogs.qt_notification import NapariQtNotification
            for w in QApplication.topLevelWidgets():
                if isinstance(w, NapariQtNotification):
                    w.close()
        except Exception:
            pass

class BatchProgressDialog(QDialog):
    """
    A non-cancellable dialog with two progress bars:
    one for overall batch progress and one for the current item's pipeline progress.
    """
    _BAR_SS = """
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
    """

    def __init__(self, parent=None, title="Batch Processing...", text="Please wait..."):
        super().__init__(parent)
        self._warnings_ctx = None
        self._canvas_frozen = False
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumWidth(380)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint | Qt.WindowModal)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        _pad = 28
        layout = QVBoxLayout(self)
        layout.setContentsMargins(_pad, _pad, _pad, _pad)
        layout.setSpacing(8)

        self._batch_label = QLabel("Batch Progress")
        self._batch_label.setAlignment(Qt.AlignCenter)
        self._batch_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._batch_label)
        self._batch_bar = QProgressBar()
        self._batch_bar.setRange(0, 100)
        self._batch_bar.setValue(0)
        self._batch_bar.setTextVisible(True)
        self._batch_bar.setStyleSheet(self._BAR_SS)
        layout.addWidget(self._batch_bar)

        layout.addSpacing(6)

        self._item_label = QLabel(text)
        self._item_label.setAlignment(Qt.AlignCenter)
        self._item_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self._item_label)
        self._item_bar = QProgressBar()
        self._item_bar.setRange(0, 100)
        self._item_bar.setValue(0)
        self._item_bar.setTextVisible(True)
        self._item_bar.setStyleSheet(self._BAR_SS)
        layout.addWidget(self._item_bar)

        # Create dim overlay behind the dialog
        self._overlay = None
        par = self.parent()
        if par is not None:
            self._overlay = _DimOverlay(par)
            self._overlay.raise_()
        self.show()
        self.raise_()
        self._center_on_parent()
        QApplication.processEvents()

    def _center_on_parent(self):
        parent = self.parent()
        if parent is not None:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0.0, 0.0, float(self.width()), float(self.height()), 20.0, 20.0)
        painter.fillPath(path, QColor("#1a1421"))
        painter.setPen(QColor("#7a6b8a"))
        painter.drawPath(path)
        painter.end()

    def update_batch_progress(self, value: int, text: str = None):
        """Updates the overall batch progress bar."""
        if text:
            self._batch_label.setText(text)
        self._batch_bar.setValue(value)
        if not self._canvas_frozen:
            QApplication.processEvents()

    def update_item_progress(self, value: int, text: str = None):
        """Updates the current item's pipeline progress bar."""
        if text:
            self._item_label.setText(text)
        self._item_bar.setValue(value)
        if not self._canvas_frozen:
            QApplication.processEvents()

    def freeze_canvas(self):
        self._canvas_frozen = True

    def unfreeze_canvas(self):
        if self._canvas_frozen:
            self._canvas_frozen = False
            QApplication.processEvents()

    def __enter__(self):
        self._warnings_ctx = warnings.catch_warnings()
        self._warnings_ctx.__enter__()
        warnings.simplefilter("ignore")
        _suppress_napari_notifications()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unfreeze_canvas()
        self.close()
        if self._overlay is not None:
            self._overlay.close()
            self._overlay.deleteLater()
            self._overlay = None
        QApplication.processEvents()
        if self._warnings_ctx:
            self._warnings_ctx.__exit__(None, None, None)
            self._warnings_ctx = None
        _restore_napari_notifications()


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


class _ThemedPopup(QDialog):
    """Dark-themed popup dialog matching the ProgressDialog style."""

    def __init__(self, parent, title, text, icon_text="", buttons=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowModal)
        self.setStyleSheet("_ThemedPopup { background-color: #1a1421; }")
        self.setMinimumWidth(340)
        self.setMaximumWidth(480)
        self._result_value = None

        from qtpy.QtWidgets import QPushButton, QHBoxLayout

        _pad = 24
        layout = QVBoxLayout(self)
        layout.setContentsMargins(_pad, _pad, _pad, _pad)
        layout.setSpacing(12)

        # Title row with icon
        title_label = QLabel(f"{icon_text}  {title}" if icon_text else title)
        title_label.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Message body
        body_label = QLabel(text)
        body_label.setStyleSheet("color: #c0b8c8; font-size: 12px; background: transparent;")
        body_label.setWordWrap(True)
        layout.addWidget(body_label)

        layout.addSpacing(8)

        # Buttons
        if buttons is None:
            buttons = [("OK", True)]
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        for label, is_accept in buttons:
            btn = QPushButton(label)
            btn.setMinimumWidth(80)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #3a2f48;
                    color: white;
                    border: 1px solid #7a6b8a;
                    border-radius: 6px;
                    padding: 6px 16px;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #4a3f58;
                }
                QPushButton:pressed {
                    background-color: #2a1f38;
                }
            """)
            if is_accept:
                btn.clicked.connect(lambda _=False, v=label: self._on_click(v, True))
            else:
                btn.clicked.connect(lambda _=False, v=label: self._on_click(v, False))
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # Dim overlay
        self._overlay = None
        par = self.parent()
        if par is not None:
            self._overlay = _DimOverlay(par)
            self._overlay.raise_()
        self.show()
        self.raise_()
        self._center_on_parent()

    def _on_click(self, value, accept):
        self._result_value = value
        if accept:
            self.accept()
        else:
            self.reject()

    def _center_on_parent(self):
        parent = self.parent()
        if parent is not None:
            parent_rect = parent.geometry()
            x = parent_rect.x() + (parent_rect.width() - self.width()) // 2
            y = parent_rect.y() + (parent_rect.height() - self.height()) // 2
            self.move(x, y)

    def _cleanup_overlay(self):
        if self._overlay is not None:
            self._overlay.close()
            self._overlay.deleteLater()
            self._overlay = None

    def done(self, result):
        self._cleanup_overlay()
        super().done(result)

    def closeEvent(self, event):
        self._cleanup_overlay()
        super().closeEvent(event)


def show_info_popup(parent, title, text):
    """Shows a dark-themed informational popup."""
    dlg = _ThemedPopup(parent, title, text, icon_text="\u2713")
    dlg.exec_()

def show_error_popup(parent, title, text):
    """Shows a dark-themed error popup."""
    dlg = _ThemedPopup(parent, title, text, icon_text="\u2716")
    dlg.exec_()

def show_warning_popup(parent, title, text):
    """Shows a dark-themed warning popup."""
    dlg = _ThemedPopup(parent, title, text, icon_text="\u26A0")
    dlg.exec_()

def show_yes_no_popup(parent, title, text):
    """Shows a dark-themed Yes/No confirmation dialog. Returns True if Yes was clicked."""
    dlg = _ThemedPopup(parent, title, text, icon_text="\u26A0",
                        buttons=[("No", False), ("Yes", True)])
    result = dlg.exec_()
    return result == QDialog.Accepted