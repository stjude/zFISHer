"""Starting a batch from the GUI writes the open session's pending mask edits
to that session first.

Mask edits are saved a moment after the last stroke. The batch pipeline makes
each dataset's session the active one, so an edit still waiting when a batch
started was saved into the first batch dataset's folder when its timer fired,
or never. Batch Process now flushes the pending edits before it starts, as
Load Session and New Session do.

Headless: a napari ``ViewerModel`` behind the attributes the widget uses, the
real mask editor and session modules, and a stand-in for the pipeline that
swaps the session the way the real one does.

Run:  python -m pytest tests/test_batch_flushes_mask_edits.py -q
"""
import contextlib
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("qtpy")

from zfisher import constants
from zfisher.core import session

MASK = "R1 - DAPI_masks"


class _Dialog(contextlib.nullcontext):
    """Stands in for the batch progress dialog, which needs a window."""

    def __init__(self, *args, **kwargs):
        super().__init__(self)

    def update_batch_progress(self, *args):
        pass

    def update_item_progress(self, *args):
        pass


@pytest.fixture
def gui(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    import napari
    from napari.components import ViewerModel
    from zfisher.ui import popups, viewer_helpers
    from zfisher.ui.widgets import batch_process_widget as bpw
    from zfisher.ui.widgets import mask_editor_widget as me

    model = ViewerModel()

    class _Viewer:
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

        def __setattr__(self, name, value):
            setattr(model, name, value)

    viewer = _Viewer()
    monkeypatch.setattr(napari, "current_viewer", lambda: viewer)
    monkeypatch.setattr(viewer_helpers, "resync_puncta_nucleus_ids",
                        lambda *a, **k: {"updated_layers": 0, "removed_total": 0, "changed_total": 0})

    open_dir, batch_dir = tmp_path / "open", tmp_path / "batch"
    session.clear_session()
    session.update_data("output_dir", str(open_dir))
    layer = model.add_labels(np.zeros((2, 16, 16), np.int32), name=MASK)
    me._save_timer.stop()
    me._flush_timer.stop()
    me._save_pending_layer = None
    me._dirty_masks.clear()

    excel = tmp_path / "batch.xlsx"
    excel.write_bytes(b"stood in for")
    config = {"datasets": [{"name": "ds1", "output_dir": None, "r1": str(tmp_path / "r1.nd2"),
                            "r2": str(tmp_path / "r2.nd2"), "seg_method": "Classical", "merge_splits": False,
                            "r1_nuclear_channel": "DAPI", "r2_nuclear_channel": "DAPI"}],
              "puncta_rules": [], "coloc_rules": [], "_warnings": []}
    monkeypatch.setattr(bpw, "parse_batch_config", lambda path: (dict(config), None))
    monkeypatch.setattr(bpw, "validate_channels", lambda cfg: [])
    monkeypatch.setattr(bpw, "resolve_puncta_for_dataset", lambda name, rules: {})
    monkeypatch.setattr(bpw, "resolve_coloc_for_dataset", lambda name, rules: ([], []))
    monkeypatch.setattr(popups, "show_yes_no_popup", lambda parent, title, text: True)
    monkeypatch.setattr(popups, "BatchProgressDialog", _Dialog)
    infos, errors = [], []
    monkeypatch.setattr(popups, "show_info_popup", lambda parent, title, text: infos.append(text))
    monkeypatch.setattr(popups, "show_error_popup", lambda parent, title, text: errors.append((title, text)))

    mask_file = open_dir / constants.SEGMENTATION_DIR / f"{MASK}.ome.tif"
    seen = {}

    def pipeline(**kwargs):
        """What the real pipeline does to the session: a new one per dataset."""
        seen["mask_saved_before"] = mask_file.exists()
        session.clear_session()
        session.update_data("output_dir", str(kwargs["output_dir"]))

    monkeypatch.setattr(bpw.pipeline, "run_full_zfisher_pipeline", pipeline)

    widget = bpw.BatchProcessWidget(viewer)
    widget._batch_file.value = excel
    widget._batch_output_dir.value = batch_dir
    yield types.SimpleNamespace(widget=widget, layer=layer, me=me, mask_file=mask_file, batch_dir=batch_dir,
                                seen=seen, infos=infos, errors=errors)
    me._save_timer.stop()
    me._flush_timer.stop()
    me._save_pending_layer = None
    me._dirty_masks.clear()
    session.clear_session()


def _let_the_timers_fire(ms=1200):
    from qtpy.QtTest import QTest
    QTest.qWait(ms)


def test_pending_mask_edits_reach_the_open_session_before_the_batch_starts(gui):
    from zfisher.core import io
    gui.layer.data[0, 5, 5] = 3
    gui.me._mark_mask_dirty(gui.layer)          # a brush stroke, saved a moment later
    gui.me._schedule_save(gui.layer)            # an editor action, saved after half a second

    gui.widget._on_batch_run()
    _let_the_timers_fire()

    assert gui.errors == [] and gui.infos and "ds1: Success" in gui.infos[0]
    assert gui.seen["mask_saved_before"], "the mask must be on disk before the pipeline replaces the session"
    assert io.read_label_tif(gui.mask_file)[0, 5, 5] == 3
    assert not (gui.batch_dir / "ds1" / constants.SEGMENTATION_DIR).exists()   # nothing strayed into the batch's folder
    assert gui.me._save_pending_layer is None and not gui.me._dirty_masks


def test_a_batch_with_nothing_pending_writes_no_mask(gui):
    gui.widget._on_batch_run()
    _let_the_timers_fire(300)

    assert gui.errors == [] and not gui.mask_file.exists()
    assert not (gui.batch_dir / "ds1" / constants.SEGMENTATION_DIR).exists()
