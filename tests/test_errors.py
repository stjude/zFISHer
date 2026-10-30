"""Typed errors and how the GUI reports them (restructuring plan 5a-5b).

Errors zFISHer raises on purpose derive from ``core.errors.ZFisherError`` and
keep the built-in type code already caught. ``error_handler`` shows them by
their message, logs a traceback only for unexpected errors, returns
``FAILED`` and, with its test flag set, raises the error again.

Run:  python -m pytest tests/test_errors.py -q
"""
import logging
import os
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from zfisher.core import errors, io, puncta, segmentation, session


def test_the_typed_errors_keep_their_built_in_types():
    assert issubclass(io.RawImageNotFound, errors.InputError) and issubclass(io.RawImageNotFound, FileNotFoundError)
    assert issubclass(segmentation.NoNucleiFound, errors.DetectionError)
    assert issubclass(segmentation.NoNucleiFound, ValueError)
    assert issubclass(session.SessionSaveError, errors.SessionError) and issubclass(session.SessionSaveError, OSError)
    assert issubclass(puncta.PunctaTableError, errors.SchemaError) and issubclass(puncta.PunctaTableError, ValueError)
    for cls in (errors.SessionError, errors.RegistrationError, errors.DetectionError, errors.StaleError):
        assert issubclass(cls, RuntimeError) and issubclass(cls, errors.ZFisherError)


@pytest.fixture
def handler(monkeypatch, caplog):
    pytest.importorskip("napari")
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from zfisher.ui import decorators, popups
    shown = []
    viewer = types.SimpleNamespace(status="", window=types.SimpleNamespace(_qt_window=None))
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: shown.append((title, text)))
    monkeypatch.setattr(decorators, "RAISE_ERRORS", False)
    log = logging.getLogger("zfisher")
    log.propagate = True
    yield types.SimpleNamespace(decorators=decorators, shown=shown, viewer=viewer, caplog=caplog)
    log.propagate = False


def test_a_detection_error_raised_in_core_reaches_the_user(handler):
    @handler.decorators.error_handler("Segmentation Failed")
    def segment():
        raise segmentation.NoNucleiFound("R1 has no nuclei: its nuclear channel looks empty.")

    with handler.caplog.at_level(logging.ERROR):
        assert segment() is handler.decorators.FAILED
    assert handler.shown == [("Segmentation Failed", "R1 has no nuclei: its nuclear channel looks empty.")]
    assert handler.viewer.status == "Error: R1 has no nuclei: its nuclear channel looks empty."
    (record,) = [r for r in handler.caplog.records if r.levelno == logging.ERROR]
    assert record.exc_info is None                                  # an expected error: no traceback


def test_an_unexpected_error_is_logged_with_its_traceback(handler):
    @handler.decorators.error_handler("Export Failed")
    def export():
        return {}["missing"]

    with handler.caplog.at_level(logging.ERROR):
        assert export() is handler.decorators.FAILED
    assert any(r.exc_info for r in handler.caplog.records if r.levelno == logging.ERROR)


def test_the_test_flag_raises_after_reporting(handler, monkeypatch):
    monkeypatch.setattr(handler.decorators, "RAISE_ERRORS", True)

    @handler.decorators.error_handler("Registration Failed")
    def register():
        raise errors.RegistrationError("too few nuclei pairs")

    with pytest.raises(errors.RegistrationError):
        register()
    assert handler.shown == [("Registration Failed", "too few nuclei pairs")]


def test_a_successful_call_returns_its_value(handler):
    @handler.decorators.error_handler("Fine")
    def fine():
        return 3

    assert fine() == 3 and handler.shown == []
