"""Check re-detecting a channel from the GUI, in zFISHer's real window.

Runs batch on small synthetic rounds in a scratch folder, opens the session
with Load Session, exports a report, then re-detects Warped R2 - FITC from
Puncta Detection with "Detect on the original image, then warp" ticked, and
saves screenshots of each step to the output folder.

    python tools/check_redetect.py <scratch folder>
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
    import pandas as pd
    from synthetic_rounds import run_batch, write_rounds
    from zfisher.core import session

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_rounds(data)
    run_batch(data, out / "batch")
    session.clear_session()
    app.log("batch finished on synthetic rounds")
    logging.setLogRecordFactory(logging.LogRecord)   # see tests/synthetic_rounds.run_batch

    viewer = app.start()
    app.shot("01_started")
    app.load_session(next((out / "batch").rglob("zfisher_session_1.json")))
    app.shot("02_session_loaded")

    # A report to go out of date.
    from zfisher.ui.widgets import colocalization_widget as coloc
    app.open_page("5. Export && Visualization", "Export", "Colocalization & Export")
    coloc._filename.value = "check_report.xlsx"
    app.click(coloc._export_btn, wait=2.0)
    app.log(f"report note after export: {coloc._report_note.text()!r} shown={coloc._report_note.isVisible()}")

    # Re-detect Warped R2 - FITC on the original image.
    from zfisher.ui.widgets import puncta_widget as pw
    app.open_page("3. Puncta Picking", "Algorithmic", "Puncta Detection")
    image = app.layer("Warped R2 - FITC")
    pw._puncta_widget.image_layer.value = image
    app.wait(0.5)
    box = pw._detect_on_original
    app.log(f"tick box: shown={box.isVisible()} enabled={box.isEnabled()} checked={box.isChecked()} "
            f"nuclei mask chosen={getattr(pw._puncta_widget.nuclei_layer.value, 'name', None)}")
    app.shot("03_detection_page")

    points = app.layer("Warped R2 - FITC_puncta")
    before = len(points.data)
    pw._puncta_widget.threshold.value = 0.6
    app.answer("replace_or_merge", "replace")
    app.click(pw._puncta_widget.call_button, wait=1.0)
    app.wait_until(lambda: "Re-detected" in str(viewer.status) or "Error" in str(viewer.status),
                   timeout=120, what="the re-detection")
    after = len(points.data)
    app.log(f"status: {viewer.status}")
    app.log(f"Warped R2 - FITC_puncta: {before} -> {after} points; same layer object: "
            f"{app.layer('Warped R2 - FITC_puncta') is points}; locked: {getattr(points, '_locked', False)}")

    for layer in viewer.layers:
        layer.visible = layer.name in ("Warped R2 - FITC", "Warped R2 - FITC_puncta", "Consensus_Nuclei_masks")
    viewer.layers.selection.active = points
    viewer.dims.set_point(0, 16 * viewer.layers["Warped R2 - FITC"].scale[0])   # the nuclei's middle plane
    app.shot("04_redetected_2d")
    viewer.dims.ndisplay = 3
    viewer.camera.angles = (0, 35, 60)
    app.shot("05_redetected_3d")
    viewer.dims.ndisplay = 2

    app.open_page("5. Export && Visualization", "Export", "Colocalization & Export")
    app.log(f"report note after re-detect: {coloc._report_note.text()!r} shown={coloc._report_note.isVisible()}")
    app.scroll_to(coloc._export_btn)
    app.shot("06_report_note")

    csv = pd.read_csv(out / "batch" / "reports" / "Warped_R2_-_FITC_puncta.csv")
    app.log(f"aligned CSV: {len(csv)} rows; matches the layer: {len(csv) == after}")
    app.finish(0)
except Exception:
    import traceback
    app.log("FAILED:\n" + traceback.format_exc())
    app.finish(1)
