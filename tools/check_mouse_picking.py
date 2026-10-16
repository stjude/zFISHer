"""Pick puncta with simulated mouse clicks on the original image, in zFISHer's real window.

Runs batch on small synthetic rounds in a scratch folder, opens the session,
opens R2 - FITC on its original image (Pick on Original Image), and then, with
Qt mouse and key events sent to the canvas (QTest), the way a person's input
arrives:

1. presses "2" (napari's add-points key) and clicks on an empty spot of the
   image: a punctum must appear under the click, marked manual, with a new id;
2. ticks the Fishing Hook, views the plane of a FITC spot, holds F and clicks
   on the spot one voxel off its centre: the punctum must land on the spot,
   in its plane (the hook used to skip the plane under the cursor);

then warps the picks into aligned space and checks both arrived. Screenshots
of each step go to the output folder. Real hands, a real trackpad and macOS
input settings are not part of it.

    python tools/check_mouse_picking.py <scratch folder>
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from app_driver import App                    # noqa: E402

out = Path(sys.argv[1])
app = App(out / "screens")
try:
    import gc
    import numpy as np
    import pandas as pd
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest
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
    from zfisher.ui.widgets.raw_picking_widget import RawPickingWidget
    from zfisher.ui.widgets import puncta_editor_widget as pe
    app.open_page("3. Puncta Picking", "Manual", "Pick on Original Image")
    picking = next(w for w in gc.get_objects() if isinstance(w, RawPickingWidget))
    picking.reset_choices()
    picking.channel.value = "R2 - FITC"
    app.click(picking.open_btn, wait=1.5)
    points = viewer.layers["R2 - FITC_puncta"]
    image = viewer.layers["R2 - FITC"]
    app.log(f"opened: {len(points.data)} raw puncta; editor on {pe._puncta_editor_widget.points_layer.value.name}")

    canvas = viewer.window._qt_viewer.canvas
    native = canvas.native
    viewer.dims.ndisplay = 2
    viewer.dims.set_point(0, 16 * image.scale[0] + image.translate[0])      # the nuclei's middle plane
    viewer.camera.zoom = 40.0                        # 4 screen pixels per 0.1 um voxel, zoomed in to pick
    app.wait(0.5)

    def world_of(data_zyx):
        return np.asarray(data_zyx, float) * np.asarray(image.scale) + np.asarray(image.translate)

    def pixel_of(data_zyx):
        """The canvas pixel showing raw voxel ``data_zyx`` in the 2D view."""
        world = world_of(data_zyx)
        transform = canvas.view.transform * canvas.view.scene.transform
        x, y = transform.map([world[2], world[1]])[:2]
        return QPoint(int(round(x)), int(round(y)))

    def centre_on(data_zyx):
        viewer.camera.center = tuple(world_of(data_zyx))
        app.wait(0.3)

    def click(data_zyx, key=None):
        pos = pixel_of(data_zyx)
        if key is not None:
            QTest.keyPress(native, key)
            app.wait(0.1)
        QTest.mousePress(native, Qt.LeftButton, Qt.NoModifier, pos)
        app.wait(0.05)
        QTest.mouseRelease(native, Qt.LeftButton, Qt.NoModifier, pos)
        app.wait(0.3)
        if key is not None:
            QTest.keyRelease(native, key)
            app.wait(0.1)

    def new_rows(before_ids):
        feats = points.features
        rows = feats[~feats["puncta_id"].isin(before_ids)]
        return [(int(r.puncta_id), np.round(points.data[i], 2).tolist(), r.Source) for i, r in rows.iterrows()]

    # 1. Plain add: "2", then a click on an empty spot between nuclei.
    viewer.layers.selection.active = points
    native.setFocus()
    empty = np.array([16.0, 90.0, 140.0])
    centre_on(empty)
    before = set(points.features["puncta_id"])
    QTest.keyClick(native, Qt.Key_2)
    app.wait(0.2)
    app.log(f"after pressing 2: layer mode = {points.mode}")
    click(empty)
    added = new_rows(before)
    app.log(f"plain add: {added}")
    app.shot("01_plain_add")
    points.mode = "pan_zoom"

    # 2. Fishing Hook: on the spot's own plane, hold F and click on the spot,
    #    one voxel off its centre (the detected punctum there is removed first,
    #    or the hook reports "Point already exists here").
    raw = pd.read_csv(out / "batch" / "reports" / "R2_-_FITC_puncta.csv")
    best = raw.sort_values("Intensity", ascending=False).iloc[0]
    spot = best[["Z", "Y", "X"]].to_numpy(float)
    keep = (points.features["puncta_id"] != best["puncta_id"]).to_numpy()
    viewer_helpers.subset_points_layer(points, keep)
    beside = spot + np.array([0.0, 1.0, 1.0])
    viewer.dims.set_point(0, spot[0] * image.scale[0] + image.translate[0])
    centre_on(beside)
    pe._puncta_editor_widget.fishing_hook.value = True
    app.wait(0.2)
    click(beside)                                    # the first press binds the F key
    before = set(points.features["puncta_id"])
    click(beside, key=Qt.Key_F)
    hooked = new_rows(before)
    app.log(f"fishing hook: clicked on the spot at {spot.tolist()} (+0,+1,+1) -> {hooked}; status: {viewer.status}")
    app.shot("02_fishing_hook")
    pe._puncta_editor_widget.fishing_hook.value = False

    # 3. Warp both picks into aligned space.
    picked = [pid for pid, _, _ in added + hooked]
    app.open_page("3. Puncta Picking", "Manual", "Pick on Original Image")
    app.click(picking.warp_btn, wait=1.5)
    aligned = viewer.layers["Warped R2 - FITC_puncta"]
    arrived = aligned.features[aligned.features["puncta_id"].isin(picked)]
    app.log(f"after warp: {picking._status.text()} picks in aligned space: "
            f"{[(int(r.puncta_id), r.Source) for _, r in arrived.iterrows()]}")
    app.shot("03_warped")
    on_spot = len(hooked) == 1 and round(hooked[0][1][0]) == spot[0] and \
        max(abs(hooked[0][1][1] - spot[1]), abs(hooked[0][1][2] - spot[2])) <= 1
    ok = (len(added) == 1 and added[0][2] == "manual" and on_spot and hooked[0][2] == "manual"
          and len(arrived) == 2)
    app.log("MOUSE PICKING " + ("OK" if ok else "FAILED"))
    app.finish(0 if ok else 1)
except Exception:
    import traceback
    app.log("MOUSE PICKING FAILED:\n" + traceback.format_exc())
    app.finish(1)
