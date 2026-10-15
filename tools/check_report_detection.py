"""Check that the report says how each channel was detected, in zFISHer's real window.

Runs batch on small synthetic rounds in a scratch folder, opens the session
with Load Session, detects Warped R2 - FITC on the aligned image (the box
"Detect on original image, then warp" unticked, the R2 question answered yes),
exports a report, and logs its Parameters sheet.

    python tools/check_report_detection.py <scratch folder>
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
    from zfisher import constants
    from zfisher.core import session

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_rounds(data)
    run_batch(data, out / "batch")
    session.clear_session()
    logging.setLogRecordFactory(logging.LogRecord)   # see tests/synthetic_rounds.run_batch

    viewer = app.start()
    app.load_session(next((out / "batch").rglob("zfisher_session_1.json")))

    from zfisher.ui.widgets import colocalization_widget as coloc
    from zfisher.ui.widgets import puncta_widget as pw
    app.open_page("3. Puncta Picking", "Algorithmic", "Puncta Detection")
    pw._puncta_widget.image_layer.value = app.layer("Warped R2 - FITC")
    app.wait(0.5)
    pw._detect_on_original.setChecked(False)
    app.answer("yes_no", True)                       # "Detect on the aligned R2 image?"
    app.answer("replace_or_merge", "replace")
    app.click(pw._puncta_widget.call_button, wait=1.0)
    app.wait_until(lambda: "Found" in str(viewer.status), timeout=120, what="the detection")
    app.log(f"status: {viewer.status}")
    app.shot("01_detected_on_aligned")

    app.open_page("5. Export && Visualization", "Export", "Colocalization & Export")
    coloc._filename.value = "check_report.xlsx"
    app.click(coloc._export_btn, wait=2.0)
    report = sorted((out / "batch" / "reports").glob("check_report*.xlsx"))[-1]
    sheet = pd.read_excel(report, sheet_name=constants.PARAMETERS_SHEET)
    app.log(f"Parameters sheet of {report.name}, columns {list(sheet.columns)}")
    for _, row in sheet.iterrows():
        app.log(f"  {row['Channel']}: {row['Detected on']}")
    app.finish(0)
except Exception:
    import traceback
    app.log("FAILED:\n" + traceback.format_exc())
    app.finish(1)
