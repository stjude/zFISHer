import napari
import webbrowser
import math
import warnings
from pathlib import Path
from functools import partial

from qtpy.QtWidgets import QApplication, QToolBox, QToolButton, QPushButton, QWidget, QLabel, QVBoxLayout, QDockWidget
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtCore import Qt, QPoint, QTimer, QEvent
from magicgui import widgets
from ..core import session

# --- RESTORED IMPORTS ---
# Import all the individual widgets from their own scripts
from .widgets.start_session_widget import StartSessionWidget
from .widgets.nuclei_segmentation_widget import NucleiSegmentationWidget
from .widgets.alignment_consensus_widget import AlignmentConsensusWidget
from .widgets.dapi_segmentation_widget import dapi_segmentation_widget
from .widgets.registration_widget import registration_widget
from .widgets.canvas_widget import canvas_widget
from .widgets.nuclei_matching_widget import nuclei_matching_widget
from .widgets.puncta_picking_widget import PunctaPickingWidget
from .widgets.puncta_widget import puncta_widget
from .widgets.colocalization_widget import colocalization_widget
from .widgets.export_visualization_widget import ExportVisualizationWidget
from .widgets.mask_editor_widget import mask_editor_widget, delete_mask_under_mouse, erase_at_cursor
from .widgets.puncta_editor_widget import puncta_editor_widget, delete_point_under_mouse
from .widgets.capture_widget import capture_widget, capture_with_hotkey, region_capture_with_hotkey, ArrowOverlay

# Import the event handlers
from . import events, style

# --- Helper Classes ---

from qtpy.QtCore import Qt, QPoint, QTimer, QEvent, QEvent

# ... (keep existing imports) ...

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
        
        # New robust positioning logic
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
        # When the parent widget is resized, move this widget
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
        # This should ideally get the pixel size from the current layer
        # For now, it seems to be hardcoded in dependent calculations
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
        
        # This part has a bug if no layers are present. Let's get scale from layer if possible.
        active_layer = self.viewer.layers.selection.active
        if active_layer:
             pixel_size_x = active_layer.scale[-1]
        else:
             pixel_size_x = 1.0 # fallback

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

class WelcomeWidget(QWidget):
    """A solid black splash screen that manages its own visibility."""
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        
        # Opaque black background
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, style.WELCOME_WIDGET_BG_COLOR)
        self.setPalette(palette)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignCenter)
        
        # Icon
        icon_path = Path(__file__).parent.parent.parent / "icon.png"
        if icon_path.exists():
            icon_label = QLabel()
            pixmap = QPixmap(str(icon_path)).scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            icon_label.setPixmap(pixmap)
            icon_label.setAlignment(Qt.AlignCenter)
            self.layout().addWidget(icon_label)

        # Branding (Mint and White)
        label_html = (
            f"<h1 {style.WELCOME_WIDGET_STYLE['h1']}>zFISHer</h1>"
            f"<p {style.WELCOME_WIDGET_STYLE['p']}>Version 1.0</p>"
        )
        self.label = QLabel(label_html)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout().addWidget(self.label)

        # Connect internal visibility logic
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

# --- Creation Logic ---

def create_welcome_widget(viewer):
    container = widgets.Container(labels=False)

    mint = style.COLORS['primary']
    workflow_html = f"""
    <h2 style='color: {mint}; margin-bottom: 2px;'>Workflow</h2>
    <table cellpadding='3' cellspacing='0' style='margin-left: 4px;'>
      <tr><td colspan='2'><b style='color: {mint};'>1. Session &amp; I/O</b></td></tr>
      <tr><td width='20'></td><td>Load .nd2 or .tif image stacks</td></tr>
      <tr><td colspan='2'><b style='color: {mint};'>2. Nuclei Segmentation</b></td></tr>
      <tr><td></td><td>Segment nuclei channels &#8594; per-round masks</td></tr>
      <tr><td></td><td>Edit masks (merge, paint, erase)</td></tr>
      <tr><td colspan='2'><b style='color: {mint};'>3. Puncta Picking</b></td></tr>
      <tr><td></td><td>Detect puncta on raw channels</td></tr>
      <tr><td></td><td>Manually add, remove, or edit spots</td></tr>
      <tr><td colspan='2'><b style='color: {mint};'>4. Alignment &amp; Consensus</b></td></tr>
      <tr><td></td><td>Register rounds &#8594; Warp to common space</td></tr>
      <tr><td></td><td>Transform puncta into aligned coordinates</td></tr>
      <tr><td></td><td>Match nuclei &#8594; Consensus mask</td></tr>
      <tr><td></td><td>Remove extranuclear puncta</td></tr>
      <tr><td colspan='2'><b style='color: {mint};'>5. Export &amp; Visualization</b></td></tr>
      <tr><td></td><td>Colocalization analysis &amp; statistics</td></tr>
      <tr><td></td><td>Capture &amp; annotate images</td></tr>
    </table>
    """

    title_label = widgets.Label(value=f"<h1 {style.CREATE_WELCOME_WIDGET_STYLE['h1']}>Welcome to zFISHer</h1>")
    subtitle_label = widgets.Label(value=f"<em style='color: {mint};'>Multiplexed Sequential FISH Analysis in Cell Monolayer</em>")
    version_label = widgets.Label(value="<p>Version 1.0</p>")
    workflow_label = widgets.Label(value=workflow_html)
    workflow_label.native.setWordWrap(True)
    container.extend([title_label, subtitle_label, version_label, workflow_label])

    btn_row = widgets.Container(layout="horizontal", labels=False)
    help_btn = widgets.PushButton(text="Open README / Help")
    reset_btn = widgets.PushButton(text="Reset")
    btn_row.extend([help_btn, reset_btn])
    container.append(btn_row)
    
    @help_btn.changed.connect
    def open_help():
        readme_path = Path(__file__).parent.parent.parent / "README.md"
        if readme_path.exists():
            webbrowser.open(readme_path.as_uri())
            
    @reset_btn.changed.connect
    def reset_viewer():
        viewer.layers.clear()
        session.clear_session()
        if hasattr(viewer.window, 'custom_scale_bar'):
            viewer.window.custom_scale_bar.hide()
            
    return container

def _patch_vispy_arcball():
    """Fix vispy arcball bug: _arcball receives 3D coords but expects 2D."""
    try:
        import vispy.scene.cameras.arcball as _ab
        _orig = _ab._arcball
        def _safe_arcball(xy, wh):
            return _orig(xy[:2], wh)
        _ab._arcball = _safe_arcball
    except Exception:
        pass

def launch_zfisher():
    _patch_vispy_arcball()

    # FIX: Ensure QApplication is initialized properly
    app = QApplication.instance() or QApplication([])

    # --- 1. Amethyst & Mint Theme (RE-ENABLED) ---
    theme_name = style.register_napari_theme()

    # Build icon — generate .ico from .png for Windows taskbar compatibility
    icon_dir = Path(__file__).parent.parent.parent
    ico_path = icon_dir / "icon.ico"
    png_path = icon_dir / "icon.png"
    if not ico_path.exists() and png_path.exists():
        try:
            from PIL import Image
            img = Image.open(png_path).convert('RGBA')
            ico_sizes = [(s, s) for s in [16, 32, 48, 64, 128, 256]]
            img.save(ico_path, format='ICO', sizes=ico_sizes)
        except Exception:
            ico_path = png_path
    # Prefer the full-res PNG for Qt (handles high-DPI natively); .ico for Win32 API
    icon_file = ico_path if ico_path.exists() else png_path
    icon = QIcon()
    if png_path.exists():
        icon.addFile(str(png_path))
    elif icon_file.exists():
        icon.addFile(str(icon_file))
    app.setWindowIcon(icon)

    # --- 2. Create Viewer ---
    viewer = napari.Viewer(title="zFISHer - 3D Colocalization of Sequential Multiplexed FISH in Cell Monolayer", ndisplay=2)

    # Apply icon — deferred so it runs after napari finishes its own window setup
    def _apply_icon():
        if icon.isNull():
            return
        qt_win = viewer.window._qt_window
        qt_win.setWindowIcon(icon)
        # Force icon onto the Win32 window handle (bypasses Qt for taskbar/title bar)
        try:
            import ctypes
            ICON_SMALL, ICON_BIG, WM_SETICON = 0, 1, 0x80
            LR_LOADFROMFILE = 0x10
            # Query actual system icon sizes (DPI-aware) instead of hardcoding
            SM_CXICON, SM_CYICON = 11, 12      # big icon
            SM_CXSMICON, SM_CYSMICON = 49, 50  # small icon
            big_w = ctypes.windll.user32.GetSystemMetrics(SM_CXICON) or 48
            big_h = ctypes.windll.user32.GetSystemMetrics(SM_CYICON) or 48
            small_w = ctypes.windll.user32.GetSystemMetrics(SM_CXSMICON) or 16
            small_h = ctypes.windll.user32.GetSystemMetrics(SM_CYSMICON) or 16
            hwnd = int(qt_win.winId())
            ico_str = str(ico_path if ico_path.exists() else icon_file)
            hicon_big = ctypes.windll.user32.LoadImageW(
                None, ico_str, 1, big_w, big_h, LR_LOADFROMFILE
            )
            hicon_small = ctypes.windll.user32.LoadImageW(
                None, ico_str, 1, small_w, small_h, LR_LOADFROMFILE
            )
            if hicon_big:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
            if hicon_small:
                ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
        except Exception:
            pass

    from qtpy.QtCore import QTimer
    QTimer.singleShot(500, _apply_icon)
    
    # Permanently disable napari's native welcome screen
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = viewer.window.qt_viewer
    qt_viewer._show_welcome_screen = False
    if hasattr(qt_viewer, '_welcome_widget'):
        qt_viewer._welcome_widget.set_welcome_visible(False)
    
    try:
        # Apply the registered theme
        viewer.theme = theme_name
    except Exception as e:
        print(f"Could not apply theme: {e}")

    viewer.scale_bar.visible = False

    # Hide the menu bar and remove all Alt+key shortcuts so it can't be reopened
    menu_bar = viewer.window._qt_window.menuBar()
    menu_bar.setVisible(False)
    for action in menu_bar.actions():
        action.setShortcut("")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        viewer_canvas_native = viewer.window.qt_viewer.canvas.native

    # Setup Welcome Overlay
    welcome_widget = WelcomeWidget(viewer, parent=viewer_canvas_native)
    viewer.window.custom_welcome_widget = welcome_widget

    scale_bar_widget = DraggableScaleBar(viewer, parent=viewer_canvas_native)
    viewer.window.custom_scale_bar = scale_bar_widget

    arrow_overlay = ArrowOverlay(viewer, parent=viewer_canvas_native)
    viewer.window.arrow_overlay = arrow_overlay
    
    nuclei_segmentation_widget = NucleiSegmentationWidget(viewer)
    alignment_consensus_widget = AlignmentConsensusWidget(viewer)
    puncta_picking_widget = PunctaPickingWidget(viewer)
    export_visualization_widget = ExportVisualizationWidget(viewer)

    widget_map = {
        # Child widgets accessed via parent composite widgets
        "dapi_segmentation": nuclei_segmentation_widget.dapi_widget,
        "mask_editor": nuclei_segmentation_widget.mask_editor_widget,
        "registration": alignment_consensus_widget.registration_widget,
        "canvas": alignment_consensus_widget.canvas_widget,
        "nuclei_matching": alignment_consensus_widget.nuclei_matching_widget,
        "automated_preprocessing": alignment_consensus_widget.automated_widget,
        # Other top-level widgets
        "puncta_detection": puncta_picking_widget.algorithmic_widget,
        "puncta_editor": puncta_picking_widget.manual_widget,
        "colocalization": export_visualization_widget.export_widget,
        "capture": export_visualization_widget.capture_widget,
        "start_session": StartSessionWidget(viewer),
        "nuclei_segmentation": nuclei_segmentation_widget,
        "alignment_consensus": alignment_consensus_widget,
        "puncta_picking": puncta_picking_widget,
        "export_visualization": export_visualization_widget,
    }

    widgets_to_add = [
        (create_welcome_widget(viewer), "zFISHer Home"),
        (StartSessionWidget(viewer), "1. Session && I/O"),
        (nuclei_segmentation_widget, "2. Nuclei Segmentation"),
        (puncta_picking_widget, "3. Puncta Picking"),
        (alignment_consensus_widget, "4. Alignment && Consensus"),
        (export_visualization_widget, "5. Export && Visualization")
    ]

    # --- 3. Sidebar Toolbox Styling ---
    toolbox = QToolBox()
    toolbox.setMinimumWidth(350)
    toolbox.setStyleSheet(style.TOOLBOX_STYLESHEET)
    
    for widget, name in widgets_to_add:
        if hasattr(widget, "reset_choices"):
            widget.reset_choices()
        toolbox.addItem(widget.native, name)

    dock_widget = viewer.window.add_dock_widget(toolbox, area="right", name="zFISHer Workflow")

    def _hide_title_bar_buttons(dock):
        """Hide float/hide/close buttons in a dock's napari custom title bar."""
        title_bar = dock.titleBarWidget()
        if title_bar is not None:
            for btn in title_bar.findChildren(QPushButton):
                obj_name = btn.objectName()
                if obj_name in ('QTitleBarHideButton', 'QTitleBarFloatButton', 'QTitleBarCloseButton'):
                    btn.hide()

    def _patch_dock_visibility(dock):
        """Monkey-patch _on_visibility_changed so title bar buttons stay hidden
        even after napari recreates the title bar."""
        if getattr(dock, '_lock_patched', False):
            return
        orig = getattr(dock, '_on_visibility_changed', None)
        if orig is not None:
            def _patched(visible, _orig=orig, _dock=dock):
                _orig(visible)
                _hide_title_bar_buttons(_dock)
            dock._on_visibility_changed = _patched
            # Reconnect the signal to the patched version
            try:
                dock.visibilityChanged.disconnect(orig)
            except (TypeError, RuntimeError):
                pass
            dock.visibilityChanged.connect(_patched)
        dock._lock_patched = True

    def lock_ui():
        # Lock every dock widget: disable float/close but keep titles visible
        qt_window = viewer.window._qt_window
        for child in qt_window.findChildren(QDockWidget):
            child.setFeatures(QDockWidget.NoDockWidgetFeatures)
            _hide_title_bar_buttons(child)
            _patch_dock_visibility(child)
            # In QtViewerButtons: keep only home and 2D/3D, centered with spacing
            # In QtLayerButtons: keep only trash can
            for w in child.findChildren(QWidget):
                class_name = w.__class__.__name__
                if class_name == 'QtViewerButtons':
                    layout = w.layout()
                    if layout:
                        home_btn = nd_btn = None
                        for btn in w.findChildren(QPushButton):
                            mode = btn.property('mode')
                            if mode == 'home':
                                home_btn = btn
                            elif mode == 'ndisplay_button':
                                nd_btn = btn
                            else:
                                btn.hide()
                        # Clear layout and rebuild centered with spacing
                        if home_btn and nd_btn:
                            while layout.count():
                                layout.takeAt(0)
                            layout.addStretch(1)
                            layout.addWidget(home_btn)
                            layout.addSpacing(12)
                            layout.addWidget(nd_btn)
                            layout.addStretch(1)
                elif class_name == 'QtLayerButtons':
                    delete_btn = None
                    for btn in w.findChildren(QPushButton):
                        tooltip = (btn.toolTip() or '').lower()
                        if 'delete' in tooltip:
                            delete_btn = btn
                        elif btn.objectName() != 'toggleVisibilityBtn':
                            btn.hide()
                    # Add a show/hide all layers toggle button (once)
                    if delete_btn and not getattr(w, '_has_toggle', False):
                        from qtpy.QtSvg import QSvgRenderer
                        from qtpy.QtGui import QImage, QPixmap, QPainter
                        from qtpy.QtCore import QSize, Qt as _Qt

                        def _tinted_icon_from_svg(svg_path, color='white', size=28):
                            renderer = QSvgRenderer(svg_path)
                            img = QImage(QSize(size, size), QImage.Format_ARGB32)
                            img.fill(_Qt.transparent)
                            p = QPainter(img)
                            renderer.render(p)
                            p.setCompositionMode(QPainter.CompositionMode_SourceIn)
                            p.fillRect(img.rect(), QColor(color))
                            p.end()
                            return QIcon(QPixmap.fromImage(img))

                        icons_dir = Path(napari.__file__).parent / 'resources' / 'icons'
                        icon_on = _tinted_icon_from_svg(str(icons_dir / 'visibility.svg'))
                        icon_off = _tinted_icon_from_svg(str(icons_dir / 'visibility_off.svg'))

                        toggle_btn = QPushButton(w)
                        toggle_btn.setObjectName('toggleVisibilityBtn')
                        toggle_btn.setToolTip('Show/hide all layers')
                        toggle_btn.setFixedSize(28, 28)
                        toggle_btn.setIconSize(QSize(20, 20))
                        toggle_btn.setIcon(icon_on)
                        toggle_btn.setStyleSheet(
                            "QPushButton { border: none; }"
                            "QPushButton:hover { background: rgba(255,255,255,30); border-radius: 4px; }"
                        )
                        toggle_btn._all_visible = True
                        def _toggle_all_layers(checked=False, btn=toggle_btn, _on=icon_on, _off=icon_off):
                            btn._all_visible = not btn._all_visible
                            btn.setIcon(_on if btn._all_visible else _off)
                            for layer in viewer.layers:
                                layer.visible = btn._all_visible
                        toggle_btn.clicked.connect(_toggle_all_layers)
                        layout = w.layout()
                        if layout:
                            idx = layout.indexOf(delete_btn)
                            layout.insertWidget(idx, toggle_btn)
                        w._has_toggle = True

    # Run immediately so the user never sees the controls flash
    lock_ui()
    # Run again after Qt finishes layout in case napari adds docks lazily
    QTimer.singleShot(0, lock_ui)

    # Event Binding
    events.install_layer_lock(viewer)
    viewer.layers.events.inserted.connect(partial(events.on_layer_inserted, widgets=widget_map))
    viewer.layers.events.removed.connect(partial(events.on_layer_removed, widgets=widget_map))

    viewer.bind_key('Shift-P', capture_with_hotkey, overwrite=True)
    viewer.bind_key('Control-A', region_capture_with_hotkey, overwrite=True)
    viewer.bind_key('x', delete_point_under_mouse, overwrite=True)
    viewer.bind_key('c', delete_mask_under_mouse, overwrite=True)
    viewer.bind_key('Shift-E', erase_at_cursor, overwrite=True)

    napari.run()