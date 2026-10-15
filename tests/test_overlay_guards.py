"""The two overlays drawn over the canvas survive values they cannot draw.

``ArrowOverlay`` (capture arrows) and ``DraggableScaleBar`` paint with a
QPainter and convert coordinates with ``int()``. A non-finite value (an arrow
endpoint that projects to nan, a camera zoom of nan or inf) raised inside
``paintEvent``; PySide swallows that, the painter stays active, and the next
repaint can end the process with an access violation, the mechanism found for
the field's 3D-rotation crash (gap analysis 5.9). Now such values are skipped,
and the painter is always ended.

Headless (offscreen Qt); the projection is given to the overlay directly.

Run:  python -m pytest tests/test_overlay_guards.py -q
"""
import math
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def trouble(monkeypatch):
    """Exceptions that reach sys.excepthook (PySide's route for errors inside
    paintEvent) and Qt's painter warnings, while the test runs."""
    from qtpy.QtCore import qInstallMessageHandler
    caught = []
    monkeypatch.setattr(sys, "excepthook", lambda t, v, tb: caught.append(f"{t.__name__}: {v}"))
    previous = qInstallMessageHandler(lambda mode, context, message: caught.append(message)
                                      if "Painter" in message or "painter" in message else None)
    yield caught
    qInstallMessageHandler(previous)


def _model():
    from napari.components import ViewerModel
    return ViewerModel()


# --- ArrowOverlay -----------------------------------------------------------------

def _overlay(qapp, project):
    from zfisher.ui.widgets.capture_widget import ArrowOverlay
    viewer = _model()
    viewer.dims.ndisplay = 3                         # no slice filter: every arrow is drawn
    overlay = ArrowOverlay(viewer)
    overlay.resize(200, 200)
    overlay._build_projection = lambda: project
    overlay._arrow_endpoints = [(np.array([1.0, 1.0, 1.0]), np.array([2.0, 5.0, 5.0])),
                                (np.array([1.0, 2.0, 2.0]), np.array([np.nan, 3.0, 3.0]))]
    return overlay


def _project(point):
    z, y, x = np.asarray(point, dtype=float)
    return x * 10 + z, y * 10                         # nan in, nan out


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, 1e300])
def test_an_arrow_that_projects_outside_the_pixel_range_is_skipped(qapp, trouble, bad):
    overlay = _overlay(qapp, lambda p: (bad, bad) if np.isnan(np.asarray(p, float)).any() else _project(p))
    from qtpy.QtGui import QPaintEvent
    overlay.paintEvent(QPaintEvent(overlay.rect()))   # outside a paint cycle errors propagate
    overlay.grab()                                    # a real repaint
    overlay.grab()                                    # and the next one: the painter was ended
    assert not [m for m in trouble if "Error" in m]


def test_an_error_while_painting_still_ends_the_painter(qapp, trouble):
    def broken(point):
        raise RuntimeError("projection failed")
    overlay = _overlay(qapp, broken)
    overlay.grab()
    overlay.grab()
    assert not [m for m in trouble if "still active" in m or "one painter" in m]


# --- DraggableScaleBar ---------------------------------------------------------------

def _bar(qapp):
    from zfisher.ui.viewer import DraggableScaleBar
    bar = DraggableScaleBar(_model())
    bar.viewer = types.SimpleNamespace(camera=types.SimpleNamespace(zoom=1.0))
    return bar


@pytest.mark.parametrize("zoom", [math.nan, math.inf, -math.inf, 0.0, -1.0])
def test_the_scale_bar_ignores_a_zoom_it_cannot_measure(qapp, trouble, zoom):
    bar = _bar(qapp)
    bar.viewer.camera.zoom = 2.0
    bar.on_zoom()
    text, length = bar.text, bar.bar_length_px
    bar.viewer.camera.zoom = zoom
    bar.on_zoom()                                     # math.log10 of nan or inf raised here
    bar.grab()
    assert bar.text == text and bar.bar_length_px == length
    assert not trouble


@pytest.mark.parametrize("zoom", [1e-12, 1e12])
def test_the_scale_bar_draws_at_extreme_zooms(qapp, trouble, zoom):
    bar = _bar(qapp)
    bar.viewer.camera.zoom = zoom
    bar.on_zoom()
    bar.grab()
    assert math.isfinite(bar.bar_length_px) and not trouble


def test_a_nonfinite_bar_length_is_not_drawn(qapp, trouble):
    bar = _bar(qapp)
    bar.bar_length_px = math.nan                     # however it got there
    bar.grab()
    bar.grab()
    assert not trouble
