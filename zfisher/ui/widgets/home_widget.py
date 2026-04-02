import webbrowser
from pathlib import Path
from ...version import __version__

from magicgui.widgets import Container, PushButton, Label
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QPushButton as QtPushButton,
    QSpacerItem, QSizePolicy,
)
from qtpy.QtCore import Qt

from ...core import session
from .. import style


def _show_readme_dialog(viewer, readme_path):
    """Show the README.md rendered as HTML in a scrollable dialog."""
    try:
        import markdown
        html_body = markdown.markdown(
            readme_path.read_text(encoding="utf-8"),
            extensions=["tables", "fenced_code", "toc"],
        )
    except ImportError:
        raw = readme_path.read_text(encoding="utf-8")
        html_body = "<pre style='white-space: pre-wrap;'>" + raw + "</pre>"

    dialog = QDialog(viewer.window._qt_window)
    dialog.setWindowTitle("zFISHer — Help & Documentation")
    dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
    dialog.resize(750, 600)

    text_browser = QTextBrowser()
    text_browser.setOpenExternalLinks(True)
    text_browser.setStyleSheet(
        "QTextBrowser { background-color: #1e1e2e; color: #cdd6f4; "
        "font-size: 13px; padding: 12px; border: none; }"
        "a { color: #89b4fa; }"
    )
    text_browser.setHtml(
        f"<html><body style='font-family: sans-serif;'>{html_body}</body></html>"
    )

    close_btn = QtPushButton("Close")
    close_btn.clicked.connect(dialog.close)

    layout = QVBoxLayout(dialog)
    layout.addWidget(text_browser)
    layout.addWidget(close_btn)

    dialog.show()


class HomeWidget(Container):
    def __init__(self, viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        mint = style.COLORS['primary']
        workflow_html = f"""
        <h2 style='color: {mint}; margin-bottom: 2px;'>Workflow</h2>
        <table cellpadding='3' cellspacing='0' style='margin-left: 4px;'>
          <tr><td colspan='2'><b style='color: {mint};'>1. Session &amp; I/O</b></td></tr>
          <tr><td width='20'></td><td>Load .nd2 or .tif image stacks</td></tr>
          <tr><td colspan='2'><b style='color: {mint};'>2. Nuclei Segmentation</b></td></tr>
          <tr><td></td><td>Segment nuclei in each round separately</td></tr>
          <tr><td></td><td>Edit masks (merge, paint, erase)</td></tr>
          <tr><td colspan='2'><b style='color: {mint};'>3. Puncta Detection</b></td></tr>
          <tr><td></td><td>Detect puncta on fluorescent channels</td></tr>
          <tr><td></td><td>Manually add, remove, or edit puncta</td></tr>
          <tr><td colspan='2'><b style='color: {mint};'>4. Alignment &amp; Consensus</b></td></tr>
          <tr><td></td><td>Align rounds &#8594; Elastic warping to common space</td></tr>
          <tr><td></td><td>Transform puncta into aligned coordinates</td></tr>
          <tr><td></td><td>Match nuclei &#8594; Merged nuclei mask</td></tr>
          <tr><td></td><td>Remove puncta outside nuclei</td></tr>
          <tr><td colspan='2'><b style='color: {mint};'>5. Export &amp; Visualization</b></td></tr>
          <tr><td></td><td>Colocalization analysis &amp; generate reports</td></tr>
          <tr><td></td><td>Capture &amp; annotate images</td></tr>
        </table>
        """

        self._title = Label(value=f"<h1 {style.CREATE_WELCOME_WIDGET_STYLE['h1']}>Welcome to zFISHer</h1>")
        self._subtitle = Label(value=f"<em style='color: {mint};'>Multiplexed Sequential FISH Analysis in Cell Monolayer</em>")
        self._version = Label(value=f"<p>Version {__version__}</p>")
        self._workflow = Label(value=workflow_html)
        self._workflow.native.setWordWrap(True)

        self._help_btn = PushButton(text="Open README / Help", tooltip="Open the zFISHer documentation and user guide.")
        self._repo_btn = PushButton(text="Visit zFISHer Repository", tooltip="Open the zFISHer GitHub repository in your browser.")
        self._reset_btn = PushButton(text="Reset", tooltip="Clear all layers and reset the session to start fresh.")

        self._init_layout()
        self._connect_signals()

    def _init_layout(self):
        spacer = lambda: QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)

        _layout = self.native.layout()
        _layout.setSpacing(2)
        _layout.setContentsMargins(0, 20, 0, 0)
        _layout.addWidget(self._title.native)
        _layout.addWidget(self._subtitle.native)
        _layout.addWidget(self._version.native)
        _layout.addItem(spacer())
        _layout.addWidget(self._workflow.native)
        _layout.addItem(spacer())
        self._workflow.native.setStyleSheet("margin-top: 20px;")
        self._help_btn.native.setStyleSheet("margin-top: 20px;")
        self._reset_btn.native.setStyleSheet("margin-top: 40px;")
        _layout.addWidget(self._help_btn.native)
        _layout.addWidget(self._repo_btn.native)
        _layout.addWidget(self._reset_btn.native)
        _layout.addStretch(1)

    def _connect_signals(self):
        self._help_btn.changed.connect(self._on_help)
        self._repo_btn.changed.connect(self._on_repo)
        self._reset_btn.changed.connect(self._on_reset)

    def _on_help(self):
        readme_path = Path(__file__).parent.parent.parent.parent / "README.md"
        if readme_path.exists():
            _show_readme_dialog(self._viewer, readme_path)

    def _on_repo(self):
        webbrowser.open("https://github.com/stjude/zFISHer")

    def _on_reset(self):
        from ..popups import show_yes_no_popup
        if not show_yes_no_popup(
            self._viewer.window._qt_window,
            "Reset Session",
            "Are you sure you wish to reset the project?\n\nAll unsaved progress will be lost.",
        ):
            return
        import logging
        logging.getLogger(__name__).info("ACTION: Session reset (all layers cleared)")

        # Reset all widget module-level state before clearing layers
        try:
            from .mask_editor_widget import reset_mask_editor_state
            reset_mask_editor_state()
        except Exception:
            pass
        try:
            from .puncta_editor_widget import reset_puncta_editor_state
            reset_puncta_editor_state()
        except Exception:
            pass
        try:
            from .capture_widget import reset_capture_state
            reset_capture_state()
        except Exception:
            pass
        try:
            from .refilter_puncta_widget import reset_refilter_state
            reset_refilter_state()
        except Exception:
            pass
        try:
            from ..events import reset_events_state
            reset_events_state()
        except Exception:
            pass
        try:
            from .. import viewer as _viewer_mod
            _viewer_mod._suppress_custom_controls = False
        except Exception:
            pass

        # Set programmatic removal so layer-removed events don't delete
        # output files from disk (they belong to the saved session).
        from ..events import _set_programmatic_removal
        _set_programmatic_removal(True)
        try:
            self._viewer.layers.clear()
        finally:
            _set_programmatic_removal(False)
        session.clear_session()
        if hasattr(self._viewer.window, 'custom_scale_bar'):
            self._viewer.window.custom_scale_bar.hide()
