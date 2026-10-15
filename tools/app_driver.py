"""Drive zFISHer's real window from a script, to check GUI changes without
someone at the screen.

``App`` starts the full zFISHer viewer (theme, workflow panel, widgets) in a
window that is laid out and drawn but never put on screen (Qt's
``WA_DontShowOnScreen``), opens panel pages by title, presses buttons, answers pop-ups from
a script, and saves screenshots of the whole window for a person, or Claude, to
look at. Everything it does is written to ``driver.log`` next to the
screenshots.

What it changes, in this process only: the window and zFISHer's progress
dialog are never shown on screen; pop-ups that wait for a click (information,
warning, yes/no, replace-or-merge, file choosers) take their answer from
``App.answer`` or a default, and their text is logged; any other dialog that
would wait is refused and logged, so a script cannot hang on it.

Limits: no real mouse or keyboard, no macOS file dialogs, and drawing happens
on this machine's graphics. Closing a never-shown napari window crashes on
macOS (napari 0.6.6), so ``finish`` ends the process without closing it.

Use only test data in a scratch folder, never the lab's data.

    from app_driver import App
    app = App(out_dir)
    app.start()
    app.load_session(session_file)
    app.open_page("3. Puncta Picking", "Algorithmic")
    app.shot("puncta_page")
    app.finish()
"""
import faulthandler
import gc
import os
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class App:
    def __init__(self, out_dir, timeout_s=600, size=(1500, 950)):
        self.size = size
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self._log = open(self.out / "driver.log", "w", encoding="utf-8")
        self._answers = {}            # pop-up kind -> deque of answers
        self.popups_seen = []         # (kind, title, text)
        self.viewer = None
        self._t0 = time.perf_counter()
        # A script that hangs prints where it is stuck and exits.
        faulthandler.dump_traceback_later(timeout_s, exit=True)

    # --- log --------------------------------------------------------------------

    def log(self, text):
        line = f"[{time.perf_counter() - self._t0:7.1f} s] {text}"
        print(line, flush=True)
        self._log.write(line + "\n")
        self._log.flush()

    # --- start and stop -------------------------------------------------------------

    def start(self):
        """Start zFISHer's viewer with its window hidden and return it."""
        os.environ.setdefault("NAPARI_ASYNC", "0")
        import napari
        from qtpy.QtWidgets import QApplication, QDialog
        self.app = QApplication.instance() or QApplication([])

        from qtpy.QtCore import Qt
        real_viewer = napari.Viewer

        def hidden_viewer(*args, **kwargs):
            # Laid out and drawn as if shown, at a usable size, but not put on screen.
            kwargs["show"] = False
            viewer = real_viewer(*args, **kwargs)
            win = viewer.window._qt_window
            win.setAttribute(Qt.WA_DontShowOnScreen, True)
            win.resize(*self.size)
            win.show()
            return viewer
        napari.Viewer = hidden_viewer
        napari.run = lambda *a, **k: None
        self._script_popups()

        def refuse(dialog, *a, **k):
            self.log(f"UNEXPECTED DIALOG refused: {dialog.windowTitle()!r}")
            self.popups_seen.append(("unexpected", dialog.windowTitle(), ""))
            return QDialog.Rejected
        QDialog.exec_ = refuse
        QDialog.exec = refuse

        from zfisher.ui.viewer import launch_zfisher
        launch_zfisher()
        self.viewer = napari.current_viewer()
        win = self.viewer.window._qt_window
        self.wait(0.5)
        self.log(f"zFISHer started, window {win.width()}x{win.height()}")
        return self.viewer

    def finish(self, code=0):
        """Write the log and end the process (see the module docstring)."""
        self.log(f"finished, {len(self.popups_seen)} pop-up(s) answered")
        self._log.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    # --- pop-ups ------------------------------------------------------------------

    def answer(self, kind, *values):
        """Queue answers for the next pop-ups of ``kind``: ``'yes_no'`` (True or
        False), ``'replace_or_merge'`` ('replace', 'merge', 'replace_all',
        'cancel'), ``'file'`` (a path, or None to cancel)."""
        self._answers.setdefault(kind, deque()).extend(values)

    def _next(self, kind, default):
        queue = self._answers.get(kind)
        return queue.popleft() if queue else default

    def _script_popups(self):
        from zfisher.ui import popups, viewer_helpers

        def message(kind):
            def show(parent, title, text, *a, **k):
                self.popups_seen.append((kind, title, text))
                self.log(f"POP-UP {kind}: {title} | {' '.join(str(text).split())}")
            return show
        popups.show_info_popup = message("info")
        popups.show_warning_popup = message("warning")
        popups.show_error_popup = message("error")

        def yes_no(parent, title, text):
            answer = self._next("yes_no", False)
            self.popups_seen.append(("yes_no", title, text))
            self.log(f"POP-UP yes/no -> {answer}: {title} | {' '.join(str(text).split())}")
            return answer
        popups.show_yes_no_popup = yes_no

        def replace_or_merge(parent, layer_name, n_existing, n_manual=0):
            answer = self._next("replace_or_merge", "cancel")
            self.popups_seen.append(("replace_or_merge", layer_name, f"{n_existing} points, {n_manual} manual"))
            self.log(f"POP-UP replace-or-merge -> {answer}: {layer_name} ({n_existing} points, {n_manual} manual)")
            return answer
        viewer_helpers.ask_replace_or_merge = replace_or_merge

        def choose_file(parent, round_id, missing):
            answer = self._next("file", None)
            self.log(f"POP-UP locate {round_id} image -> {answer}")
            return answer
        popups.choose_raw_image = choose_file

        # zFISHer's progress dialog shows itself; keep it off the screen.
        popups.ProgressDialog.show = lambda dialog: None
        popups.ProgressDialog.raise_ = lambda dialog: None

    # --- driving -------------------------------------------------------------------

    def wait(self, seconds=0.2):
        """Let Qt process events for ``seconds``."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.app.processEvents()
            time.sleep(0.01)

    def open_page(self, *titles):
        """Open a page of the workflow panel, then pages inside it, by title
        (``open_page("3. Puncta Picking", "Algorithmic", "Puncta Detection")``)."""
        from qtpy.QtWidgets import QToolBox
        scope = self.viewer.window._qt_window
        for title in titles:
            for box in scope.findChildren(QToolBox):
                names = [box.itemText(i).replace("&&", "&") for i in range(box.count())]
                wanted = title.replace("&&", "&")
                if wanted in names:
                    index = names.index(wanted)
                    box.setCurrentIndex(index)
                    scope = box.widget(index)
                    break
            else:
                raise LookupError(f"No panel page {title!r}")
        self.wait(0.3)
        self.log(f"opened {' > '.join(titles)}")

    def click(self, button, wait=0.5):
        """Press a magicgui or Qt button, and let the app react."""
        native = getattr(button, "native", button)
        native.click()
        self.wait(wait)

    def wait_until(self, condition, timeout=120, what="condition"):
        end = time.perf_counter() + timeout
        while time.perf_counter() < end:
            self.app.processEvents()
            if condition():
                return True
            time.sleep(0.05)
        self.log(f"TIMEOUT waiting for {what}")
        return False

    def scroll_to(self, widget):
        """Scroll the panel so ``widget`` (Qt or magicgui) is in view."""
        from qtpy.QtWidgets import QScrollArea
        native = getattr(widget, "native", widget)
        parent = native.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                parent.ensureWidgetVisible(native, 50, 50)
            parent = parent.parentWidget()
        self.wait(0.2)

    def shot(self, name):
        """Save a screenshot of the whole window as ``<name>.png``."""
        self.wait(0.3)
        path = self.out / f"{name}.png"
        self.viewer.screenshot(str(path), canvas_only=False)
        self.log(f"screenshot {path.name}")
        return path

    def load_session(self, session_file):
        """Load a session through the Load Session widget, as a person would."""
        from zfisher.ui.widgets.load_session_widget import LoadSessionWidget
        win = self.viewer.window._qt_window
        widgets = [w for w in gc.get_objects() if isinstance(w, LoadSessionWidget)]
        shown = [w for w in widgets if _is_inside(w.native, win)]
        widget = (shown or widgets)[0]
        self.open_page("1. Session && I/O", "Load Previous Session")
        widget._load_session_file.value = str(session_file)
        self.click(widget._load_session_btn, wait=1.0)
        self.log(f"loaded {Path(session_file).name}: {len(self.viewer.layers)} layers")

    def layer(self, name):
        return self.viewer.layers[name]


def _is_inside(widget, ancestor):
    parent = widget
    while parent is not None:
        if parent is ancestor:
            return True
        parent = parent.parentWidget()
    return False
