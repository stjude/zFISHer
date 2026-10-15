"""Button, checkbox and panel-page labels show every "&".

Qt reads a single "&" in such a label as a keyboard-shortcut marker and drops
it, so "Run Analysis & Export Master Report" showed as "Run Analysis  Export
Master Report". A literal "&" is written "&&". This scans the UI source for
label strings with a lone "&".

Run:  python -m pytest tests/test_ui_labels.py -q
"""
import re
from pathlib import Path

UI = Path(__file__).resolve().parents[1] / "zfisher" / "ui"

# A label argument: addItem(widget, "…"), PushButton(text="…"), call_button="…",
# QPushButton("…"), QCheckBox("…"), CheckBox(text="…").
LABEL = re.compile(r'(?:addItem\([^,()]+,\s*|PushButton\(text=|call_button=|QPushButton\(|QCheckBox\(|'
                   r'CheckBox\(text=)"([^"]*)"')


def test_no_label_has_a_lone_ampersand():
    found = []
    for path in sorted(UI.rglob("*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for label in LABEL.findall(line):
                if re.search(r"(?<!&)&(?!&)", label):
                    found.append(f"{path.relative_to(UI.parent.parent)}:{n}: {label!r}")
    assert not found, "Write a literal & as && in:\n" + "\n".join(found)


def test_the_scan_finds_labels():
    """The pattern must keep matching the UI's labels, or the test above passes vacuously."""
    labels = [label for path in UI.rglob("*.py") for label in LABEL.findall(path.read_text(encoding="utf-8"))]
    assert "Run Analysis && Export Master Report" in labels and len(labels) > 20
