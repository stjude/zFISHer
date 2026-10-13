"""Loading raw images reads each round from disk once.

``load_raw_data_into_viewer`` used to load both rounds into the viewer and then
load them again only to read their channel names for the nuclear channel: a
second full read of a multi-GB round on every New Session and every Load Session
showing raw images. It now takes the names from the first round it loaded.

Run (in an env with napari):  python -m pytest tests/test_raw_load_once.py -q
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="module")
def _qapp():
    pytest.importorskip("napari")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def env(_qapp, tmp_path, monkeypatch):
    """A headless viewer, an in-memory session, and a count of round loads."""
    from napari.components import ViewerModel
    from zfisher.core import io, session
    from zfisher.ui.widgets import _shared
    data = {"output_dir": str(tmp_path)}
    monkeypatch.setattr(session, "get_data", lambda key=None, default=None: data.get(key, default) if key else data)
    monkeypatch.setattr(session, "update_data", lambda key, value: data.__setitem__(key, value))
    loads = []
    real = io.load_image_session
    monkeypatch.setattr(io, "load_image_session", lambda path: (loads.append(path.name), real(path))[1])
    model = ViewerModel()

    class _Viewer:                       # the model, plus the window the nuclear-channel prompt uses
        window = types.SimpleNamespace(_qt_window=None)

        def __getattr__(self, name):
            return getattr(model, name)

    def write_round(name, channels):
        path = tmp_path / name
        tifffile.imwrite(path, np.zeros((4, len(channels), 8, 8), np.uint16), photometric="minisblack",
                         metadata={"axes": "ZCYX", "Channel": {"Name": channels}})
        return path

    return types.SimpleNamespace(viewer=_Viewer(), data=data, loads=loads, write=write_round,
                                 load=_shared.load_raw_data_into_viewer, tmp=tmp_path)


def test_each_round_is_read_once(env):
    r1 = env.write("R1.ome.tif", ["DAPI", "FITC"])
    r2 = env.write("R2.ome.tif", ["DAPI", "Cy5"])
    env.load(env.viewer, r1, r2, output_dir=env.tmp)
    assert sorted(env.loads) == ["R1.ome.tif", "R2.ome.tif"]
    assert env.data["nuclear_channel"] == "DAPI"
    assert "R1 - FITC" in env.viewer.layers and "R2 - Cy5" in env.viewer.layers


def test_a_missing_first_round_takes_the_names_from_the_second(env):
    r2 = env.write("R2.ome.tif", ["Hoechst", "FITC"])
    env.load(env.viewer, env.tmp / "gone.tif", r2, output_dir=env.tmp)
    assert env.loads == ["R2.ome.tif"]
    assert env.data["nuclear_channel"] == "Hoechst"


def test_the_user_is_still_asked_when_no_name_looks_nuclear(env, monkeypatch):
    from zfisher.ui import popups
    asked = []
    monkeypatch.setattr(popups, "select_nuclear_channel",
                        lambda parent, channels: (asked.append(list(channels)), "Ch2")[1])
    r1 = env.write("R1.ome.tif", ["Ch1", "Ch2"])
    r2 = env.write("R2.ome.tif", ["Ch1", "Ch2"])
    env.load(env.viewer, r1, r2, output_dir=env.tmp)
    assert asked == [["Ch1", "Ch2"]] and env.data["nuclear_channel"] == "Ch2"
    assert len(env.loads) == 2


def test_a_known_nuclear_channel_is_kept(env):
    env.data["nuclear_channel"] = "FITC"
    r1 = env.write("R1.ome.tif", ["DAPI", "FITC"])
    r2 = env.write("R2.ome.tif", ["DAPI", "FITC"])
    env.load(env.viewer, r1, r2, output_dir=env.tmp)
    assert env.data["nuclear_channel"] == "FITC" and len(env.loads) == 2
