import logging
import napari
from pathlib import Path
from magicgui.widgets import Container

from ...core import io, session
from .. import viewer_helpers, popups
from ..style import COLORS

logger = logging.getLogger(__name__)


def make_header_divider():
    """Create a full-width divider widget for use after widget title/description."""
    from qtpy.QtWidgets import QFrame
    wrapper = Container(labels=False)
    wrapper.native.setFixedHeight(10)
    wrapper.native.setContentsMargins(0, 0, 0, 0)
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(
        f"background-color: {COLORS['separator_color']}; border: none; margin: 4px 0px;"
    )
    wrapper.native.layout().addWidget(line)
    return wrapper


# --- Canonical section-styling helpers (shared by all sidebar widgets) ---------
# These replace per-widget copies of the same four helpers. The separator color
# is sourced from the theme (COLORS['separator_color']) so a theme change applies
# everywhere instead of being silently missed by hardcoded copies.

def make_divider():
    """Full-width 2px horizontal divider (raw QFrame) for inside a widget form."""
    from qtpy.QtWidgets import QFrame
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
    return line


def make_section_header(text):
    """Bold accent-colored section header label."""
    from qtpy.QtWidgets import QLabel
    label = QLabel(f"<b style='color: {COLORS['separator_color']};'>{text}</b>")
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("margin: 0px 2px; padding: 0px;")
    return label


def make_section_desc(text):
    """Wrapped descriptive text label shown under a section header."""
    from qtpy.QtWidgets import QLabel
    desc = QLabel(text)
    desc.setWordWrap(True)
    desc.setStyleSheet("color: white; margin: 2px 2px 10px 2px;")
    return desc


def make_spacer(height=20):
    """Fixed-height invisible spacer widget."""
    from qtpy.QtWidgets import QWidget
    s = QWidget()
    s.setFixedHeight(height)
    return s


def load_raw_data_into_viewer(viewer, round1_path, round2_path, output_dir=None, progress_callback=None):
    """
    Orchestrates loading raw image data (ND2/TIFF), converting it, and adding it to the viewer.
    """
    # Setup input storage directory
    input_storage_dir = None
    if output_dir:
        input_storage_dir = Path(output_dir) / "input"
        input_storage_dir.mkdir(parents=True, exist_ok=True)

    all_paths = [(Path(round1_path), "R1"), (Path(round2_path), "R2")]
    steps_per_file = 3 # Load, Convert, Add to viewer
    total_steps = len(all_paths) * steps_per_file

    for i, (path, prefix) in enumerate(all_paths):
        base_progress = i * steps_per_file

        if not path.exists():
            logger.error("File not found: %s", path)
            if progress_callback:
                progress_callback(int(((base_progress + steps_per_file) / total_steps) * 100), f"Not found: {path.name}")
            continue

        if progress_callback:
            progress_callback(int(((base_progress + 0) / total_steps) * 100), f"Loading {prefix}: {path.name}...")

        # 1. Core I/O Logic
        image_session = io.load_image_session(path)

        # 2. Conversion/Processing Logic
        if path.suffix.lower() == '.nd2' and input_storage_dir:
            def conversion_progress(msg):
                if progress_callback:
                    progress_callback(int(((base_progress + 1) / total_steps) * 100), msg)
            io.convert_nd2_to_ome(image_session, input_storage_dir, prefix, conversion_progress)

        # 3. UI Logic
        if progress_callback:
            progress_callback(int(((base_progress + 2) / total_steps) * 100), f"Adding {prefix} layers to viewer...")
        viewer_helpers.add_image_session_to_viewer(viewer, image_session, prefix)

    # Resolve and store the nuclear channel name (first round's channels used)
    if not session.get_data("nuclear_channel"):
        # Collect channels from whichever round loaded successfully
        all_channels = []
        for path in [Path(round1_path), Path(round2_path)]:
            try:
                s = io.load_image_session(path)
                all_channels = s.channels
                break
            except Exception:
                continue

        if all_channels:
            nuc = io.find_nuclear_channel(all_channels)
            if nuc is None:
                # Auto-detection failed — ask the user
                nuc = popups.select_nuclear_channel(
                    viewer.window._qt_window, all_channels
                )
            if nuc:
                logger.info("Nuclear channel selected: %s", nuc)
                session.update_data("nuclear_channel", nuc)

    # Force the Z-slider to appear
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.reset_view()
