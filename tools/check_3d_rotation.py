"""Rotate the 3D view of a session with puncta, in zFISHer's real window.

The regression check for the field's 3D-rotation crash (scope plan M1). It runs
batch on small synthetic rounds in a scratch folder, opens the session with
Load Session, shows the warped image as a volume with its puncta (with ID
labels), the consensus mask and its nucleus-ID labels, adds capture arrows
(one with a corrupt, nan endpoint), and then, in 3D, sweeps the camera through
three perspectives and nine sets of angles, two extreme zooms, and point-count
changes, drawing a real frame (OpenGL) at every step.

It passes when every frame is drawn, nothing raises, nothing napari catches
reaches the session log, and the process gets to the end: a native crash,
the field symptom, ends it early instead. Last line of the log on success:
``ROTATION OK frames=<n>``.

    python tools/check_3d_rotation.py <scratch folder>
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from app_driver import App                    # noqa: E402

PERSPECTIVES = (0, 45, 90)
ANGLES = [(0, 0, 90), (0, 30, 60), (45, 45, 45), (89.9999, 0, 0), (0, 89.9999, 0),
          (90, 90, 90), (180, 0, 0), (270, 180, 90), (-30, 120, -60)]
# Errors seen in the session log that are not about drawing; see the report.
KNOWN = ("Slot 'StallMonitor::_tick()' not found",)

out = Path(sys.argv[1])
app = App(out / "screens")
try:
    import numpy as np
    from synthetic_rounds import run_batch, write_rounds
    from zfisher.core import session
    from zfisher.ui import viewer_helpers

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_rounds(data)
    run_batch(data, out / "batch")
    session.clear_session()
    logging.setLogRecordFactory(logging.LogRecord)   # see tests/synthetic_rounds.run_batch

    viewer = app.start()
    app.load_session(next((out / "batch").rglob("zfisher_session_1.json")))
    raised = []
    sys.excepthook = lambda t, v, tb: raised.append(f"{t.__name__}: {v}")

    shown = ("Warped R2 - FITC", "Warped R2 - FITC_puncta", "Consensus_Nuclei_masks", "Consensus_Nuclei_masks_IDs")
    for layer in viewer.layers:
        layer.visible = layer.name in shown
    points = viewer.layers["Warped R2 - FITC_puncta"]
    viewer.layers.selection.active = points

    overlay = viewer.window.arrow_overlay
    overlay._arrow_endpoints = [(np.array([10.0, 30.0, 30.0]), np.array([16.0, 60.0, 80.0])),
                                (np.array([12.0, 100.0, 40.0]), np.array([20.0, 140.0, 140.0])),
                                (np.array([12.0, 90.0, 90.0]), np.array([np.nan, 120.0, 60.0]))]   # corrupt
    viewer.dims.ndisplay = 3
    app.wait(0.5)

    frames = 0

    def frame(label):
        global frames
        app.wait(0.05)
        image = viewer.screenshot(canvas_only=True)
        overlay.grab()
        frames += 1
        if not np.isfinite(image).all() or image[..., :3].max() == 0:
            app.log(f"FRAME {label}: nothing drawn")
            raise AssertionError(f"frame {label} drew nothing")

    for perspective in PERSPECTIVES:
        viewer.camera.perspective = perspective
        for angles in ANGLES:
            viewer.camera.angles = angles
            frame(f"perspective={perspective} angles={angles}")
    app.shot("rotated_3d")

    zoom = viewer.camera.zoom
    for factor in (1e-4, 1e4):
        viewer.camera.zoom = zoom * factor
        frame(f"zoom x{factor:g}")
    viewer.camera.zoom = zoom

    # Point counts changing while the 3D view is drawn, the GL-buffer case.
    full = (np.array(points.data), points.features.copy())
    viewer_helpers.subset_points_layer(points, np.arange(len(full[0])) % 3 == 0)
    frame("puncta shrunk")
    viewer_helpers.set_points_data(points, full[0][:3], full[1].iloc[:3].reset_index(drop=True))
    frame("three puncta")
    viewer_helpers.set_points_data(points, full[0], full[1])
    frame("puncta restored")
    viewer.dims.ndisplay = 2
    frame("back to 2D")

    log = sorted((out / "batch" / "logs").glob("zfisher_*.log"))[-1].read_text()
    caught = [line for line in log.splitlines()
              if ("caught by napari" in line or "could not be painted" in line)
              and not any(known in line for known in KNOWN)]
    if raised or caught:
        app.log(f"ROTATION FAILED: raised={raised} caught={caught[:5]}")
        app.finish(1)
    app.log(f"ROTATION OK frames={frames}")
    app.finish(0)
except Exception:
    import traceback
    app.log("ROTATION FAILED:\n" + traceback.format_exc())
    app.finish(1)
