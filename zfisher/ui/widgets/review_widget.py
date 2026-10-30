"""The Review Queue page: what an assistant proposed, and which datasets of a
batch need a look.

**Proposals** lists the changes proposed for the open session (through the
MCP server's ``propose``) that nobody has answered yet, from its decisions
log, with who proposed each, through which client, and why. Accept or
Dismiss logs a person's answer (``api.review_proposal``); accepting records
agreement, and the change itself is made as any other.

**Batch Check** runs the batch quality check (``api.flag_batch``) on a batch
folder and lists the datasets it flags, with the reason for each.
"""
from magicgui.widgets import Container, Label, LineEdit, PushButton
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QListWidget, QListWidgetItem, QSizePolicy

from ...core import api, session
from ._shared import make_divider, make_section_desc, make_section_header


def _describe(item):
    scope, parts = dict(item.scope), []
    if "round" in scope and "channel" in scope:          # named as the session names channels
        parts.append(f"{scope.pop('round')} - {scope.pop('channel')}")
    where = ", ".join(parts + [str(v) for v in scope.values()]) or "session"
    old = f" (now {item.old})" if item.old is not None else ""
    via = f" via {item.via}" if item.via else ""
    text = f"{where}: {item.name} → {item.value}{old}"
    if item.justification:
        text += f"\n    {item.justification}"
    return text + f"\n    {item.actor}{via}, {item.time.replace('T', ' ')}"


class ReviewQueueWidget(Container):
    def __init__(self, viewer):
        super().__init__(labels=False)
        self._viewer = viewer
        header = Label(value="Review Queue")
        header.native.setObjectName("widgetHeader")
        info = Label(value="<i>Changes an assistant proposed for the open session, and the datasets of a "
                           "batch that need a look.</i>")
        info.native.setObjectName("widgetInfo")
        info.native.setWordWrap(True)
        info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._proposals = QListWidget()
        self._proposals.setWordWrap(True)
        self._proposals.setMinimumHeight(140)
        self._note = LineEdit(value="", tooltip="Why you accept or dismiss it; logged with your answer.")
        self._note.native.setPlaceholderText("Note (optional)")
        self._accept_btn = PushButton(text="Accept", tooltip="Log that you agree with the selected proposal.")
        self._dismiss_btn = PushButton(text="Dismiss", tooltip="Log that you turn the selected proposal down.")
        self._refresh_btn = PushButton(text="Refresh", tooltip="Read the open session's decisions log again.")
        self._proposals_note = Label(value="")

        self._batch_btn = PushButton(text="Check a Batch Folder...", tooltip=(
            "Compare each dataset of a batch with the rest and list those that need a look."))
        self._flags = QListWidget()
        self._flags.setWordWrap(True)
        self._flags.setMinimumHeight(120)
        self._batch_note = Label(value="")
        for label in (self._proposals_note, self._batch_note):
            label.native.setWordWrap(True)
            label.native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        layout = self.native.layout()
        layout.setSpacing(2)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(header.native)
        layout.addWidget(info.native)
        layout.addWidget(make_divider())
        layout.addWidget(make_section_header("Proposals"))
        layout.addWidget(make_section_desc("Proposed for the open session and not yet answered."))
        layout.addWidget(self._proposals)
        layout.addWidget(self._proposals_note.native)
        layout.addWidget(self._note.native)
        for button in (self._accept_btn, self._dismiss_btn, self._refresh_btn):
            layout.addWidget(button.native)
        layout.addWidget(make_divider())
        layout.addWidget(make_section_header("Batch Check"))
        layout.addWidget(make_section_desc("The datasets of a batch whose quality-control numbers stand out."))
        layout.addWidget(self._batch_btn.native)
        layout.addWidget(self._batch_note.native)
        layout.addWidget(self._flags)
        layout.addStretch(1)

        self._accept_btn.changed.connect(lambda: self._answer(True))
        self._dismiss_btn.changed.connect(lambda: self._answer(False))
        self._refresh_btn.changed.connect(self.refresh)
        self._batch_btn.changed.connect(self._on_batch)
        self.refresh()

    def reset_choices(self):
        self.refresh()

    # --- proposals -------------------------------------------------------------------

    def refresh(self):
        self._proposals.clear()
        current = session.default()
        if not current.get("output_dir"):
            self._proposals_note.value = "No session is open."
            return
        try:
            items = api.review_queue(current)
        except Exception as exc:
            self._proposals_note.value = f"Could not read the decisions log: {exc}"
            return
        for item in items:
            row = QListWidgetItem(_describe(item))
            row.setData(Qt.UserRole, item.id)
            self._proposals.addItem(row)
        self._proposals_note.value = (f"{len(items)} proposal(s) waiting." if items
                                      else "No proposals are waiting.")

    def _answer(self, accept):
        row = self._proposals.currentItem()
        if row is None:
            self._proposals_note.value = "Select a proposal first."
            return
        try:
            api.review_proposal(session.default(), row.data(Qt.UserRole), accept, note=self._note.value or None)
        except LookupError as exc:
            self._proposals_note.value = str(exc)
        else:
            self._viewer.status = f"{'Accepted' if accept else 'Dismissed'} the proposal; logged in decisions.jsonl."
            self._note.value = ""
        self.refresh()

    # --- batch check -----------------------------------------------------------------

    def _on_batch(self):
        folder = QFileDialog.getExistingDirectory(self._viewer.window._qt_window, "Batch output folder")
        if folder:
            self.check_batch(folder)

    def check_batch(self, folder):
        """List the datasets under ``folder`` that the batch quality check flags."""
        from .. import popups
        from ..workers import run_on_worker
        self._flags.clear()
        with popups.ProgressDialog(self._viewer.window._qt_window, "Checking the batch...") as dialog:
            dialog.update_progress(10, "Reading each dataset's numbers...")
            try:
                result = run_on_worker(lambda: api.flag_batch(folder, write_report=False))
            except Exception as exc:
                self._batch_note.value = f"Could not check it: {exc}"
                return
        for flag in result.flags:
            text = f"{flag['dataset']}: {flag['action']}. {flag['reason']}"
            self._flags.addItem(QListWidgetItem(text))
        flagged = len({f["dataset"] for f in result.flags})
        self._batch_note.value = (f"{flagged} of {result.n_datasets} dataset(s) need a look." if result.flags
                                  else f"None of the {result.n_datasets} dataset(s) stands out.")
