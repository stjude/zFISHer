"""Check picking on the original image, in zFISHer's real window.

Runs batch on small synthetic rounds in a scratch folder, opens the session
with Load Session, opens R2 - FITC on its original image (with the round's
nuclei mask) from Puncta Picking > Manual > Pick on Original Image, adds one
punctum by hand at a nucleus centre, warps the picks into aligned space, and
saves screenshots of each step.

    python tools/check_raw_picking.py <scratch folder>
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
    import numpy as np
    import pandas as pd
    from synthetic_rounds import run_batch, write_rounds
    from zfisher.core import puncta, session

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_rounds(data)
    run_batch(data, out / "batch")
    session.clear_session()
    logging.setLogRecordFactory(logging.LogRecord)   # see tests/synthetic_rounds.run_batch

    viewer = app.start()
    app.load_session(next((out / "batch").rglob("zfisher_session_1.json")))
    app.open_page("3. Puncta Picking", "Manual", "Pick on Original Image")

    from zfisher.ui.widgets.raw_picking_widget import RawPickingWidget
    from zfisher.ui.widgets import puncta_editor_widget as pe
    import gc
    widget = next(w for w in gc.get_objects() if isinstance(w, RawPickingWidget))
    widget.reset_choices()
    app.log(f"channels offered: {list(widget.channel.choices)}")
    widget.channel.value = "R2 - FITC"
    widget.show_mask.value = True
    app.shot("01_page")
    app.click(widget.open_btn, wait=1.5)
    app.log(f"after open: {widget._status.text()}")
    app.log("visible layers: " + ", ".join(l.name for l in viewer.layers if l.visible))
    editor_layer = getattr(pe._puncta_editor_widget.points_layer.value, "name", None)
    app.log(f"Puncta Editor works on: {editor_layer}")
    viewer.dims.set_point(0, 16 * viewer.layers["R2 - FITC"].scale[0])
    app.shot("02_original_open")

    # One pick at a nucleus centre of the R2 round (raw voxels), as the editor adds it.
    from zfisher.ui import viewer_helpers
    points = viewer.layers["R2 - FITC_puncta"]
    feats = points.features.copy()
    row = {c: 0 for c in feats.columns}
    pid = int(puncta.allocate_puncta_ids(points.name, 1, feats["puncta_id"])[0])
    row.update(puncta_id=pid, Source="manual")
    viewer_helpers.set_points_data(points, np.vstack([points.data, [16.0, 43.0, 38.0]]),
                                   pd.concat([feats, pd.DataFrame([row])], ignore_index=True))
    app.wait(0.5)
    app.shot("03_picked")

    app.click(widget.warp_btn, wait=1.5)
    app.log(f"after warp: {widget._status.text()}")
    aligned = viewer.layers["Warped R2 - FITC_puncta"]
    hit = aligned.features.index[aligned.features["puncta_id"] == pid]
    app.log(f"pick {pid} in Warped R2 - FITC_puncta at {np.round(aligned.data[hit[0]], 2).tolist() if len(hit) else None}; "
            f"layers left: {[l.name for l in viewer.layers if l.name in ('R2 - FITC', 'R2 - FITC_puncta', 'R2 nuclei (original image)')]}")
    app.log("visible layers: " + ", ".join(l.name for l in viewer.layers if l.visible))
    viewer.dims.set_point(0, float(aligned.data[hit[0]][0]) * aligned.scale[0] if len(hit) else 0)
    app.shot("04_warped")
    app.finish(0)
except Exception:
    import traceback
    app.log("FAILED:\n" + traceback.format_exc())
    app.finish(1)
