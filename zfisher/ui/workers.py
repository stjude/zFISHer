"""Running slow work off the main thread while the GUI keeps drawing."""
import threading

from qtpy.QtWidgets import QApplication


def run_on_worker(fn):
    """Run ``fn`` on a worker thread while this thread processes GUI events;
    returns what ``fn`` returns and raises what it raises."""
    out, error = [None], [None]

    def run():
        try:
            out[0] = fn()
        except BaseException as exc:
            error[0] = exc
    thread = threading.Thread(target=run, daemon=True, name="zfisher-worker")
    thread.start()
    while thread.is_alive():
        thread.join(timeout=0.05)
        QApplication.processEvents()
    if error[0] is not None:
        raise error[0]
    return out[0]
