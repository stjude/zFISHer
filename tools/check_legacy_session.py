"""Check what a session without a saved warp tells the user, in zFISHer's real window.

Runs batch on small synthetic rounds in a scratch folder, removes the saved
registration from the session file, as in a session made before zFISHer 2.1.0,
opens it with Load Session, and shows Puncta Detection on Warped R2 - FITC and
the Pick on Original Image page, with screenshots of both.

    python tools/check_legacy_session.py <scratch folder>
"""
import json
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
    from synthetic_rounds import run_batch, write_rounds
    from zfisher.core import session

    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    write_rounds(data)
    run_batch(data, out / "batch")
    session.clear_session()
    logging.setLogRecordFactory(logging.LogRecord)   # see tests/synthetic_rounds.run_batch

    session_file = next((out / "batch").rglob("zfisher_session_1.json"))
    stored = json.loads(session_file.read_text())
    stored.pop("registration", None)                  # as before 2.1.0
    session_file.write_text(json.dumps(stored, indent=4))

    viewer = app.start()
    app.load_session(session_file)
    app.log(f"status after load: {viewer.status}")

    from zfisher.ui.widgets import puncta_widget as pw
    app.open_page("3. Puncta Picking", "Algorithmic", "Puncta Detection")
    pw._puncta_widget.image_layer.value = app.layer("Warped R2 - FITC")
    app.wait(0.5)
    box, note = pw._detect_on_original, pw._original_note
    app.log(f"tick box: shown={box.isVisible()} enabled={box.isEnabled()} checked={box.isChecked()}; "
            f"note shown={note.isVisible()}: {note.text()}")
    app.shot("01_detection_page")

    from zfisher.ui.widgets.raw_picking_widget import RawPickingWidget
    app.open_page("3. Puncta Picking", "Manual", "Pick on Original Image")
    widget = next(w for w in gc.get_objects() if isinstance(w, RawPickingWidget))
    widget.reset_choices()
    app.log(f"picking page: note shown={widget._unavailable.isVisible()}, open enabled={widget.open_btn.enabled}")
    app.shot("02_picking_page")
    app.finish(0)
except Exception:
    import traceback
    app.log("FAILED:\n" + traceback.format_exc())
    app.finish(1)
