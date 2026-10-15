"""Rotating the 3D view of a session with puncta does not crash.

The regression test for the field's 3D-rotation crash (scope plan M1). It runs
``tools/check_3d_rotation.py`` in a child process: zFISHer's real window (drawn
with OpenGL, never put on screen) on a synthetic batch session, with puncta
and their ID labels, nucleus-ID labels, the consensus mask, a volume image and
capture arrows, one with a corrupt endpoint; the camera sweeps three
perspectives and nine sets of angles, two extreme zooms and point-count
changes in 3D, drawing every frame. A native crash, the field symptom, ends
the child early and fails the test, which then shows the crash log.

With the arrow overlay as it was before its guards, this crashes
(segmentation fault) as soon as the 3D view is drawn.

What it cannot reproduce: the lab machines' graphics drivers and GPU memory,
real mouse dragging, and full-size data. It needs a display: it is skipped on
machines without one (GitHub's Linux runners).

Run:  python -m pytest tests/test_3d_rotation.py -q        (about 40 s)
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

no_display = sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


@pytest.mark.skipif(no_display, reason="needs a display for OpenGL drawing")
def test_rotating_the_3d_view_draws_every_frame_without_crashing(tmp_path):
    pytest.importorskip("napari")
    env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}   # the real platform, for GL
    result = subprocess.run([sys.executable, str(REPO / "tools" / "check_3d_rotation.py"), str(tmp_path)],
                            cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    log = (tmp_path / "screens" / "driver.log")
    text = log.read_text() if log.exists() else result.stdout
    crash = "".join(p.read_text() for p in (tmp_path / "batch" / "logs").glob("crash_*.log")) \
        if (tmp_path / "batch" / "logs").exists() else ""
    assert result.returncode == 0 and "ROTATION OK" in text, (
        f"exit code {result.returncode}\n--- driver log ---\n{text[-3000:]}\n--- crash log ---\n{crash[:3000]}")
