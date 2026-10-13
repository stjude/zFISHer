"""The release workflow's check of a tag against the repository.

``.github/scripts/release_notes.py`` refuses a tag that does not match
``zfisher/version.py`` or ``CITATION.cff``, or whose CHANGELOG section is
missing, empty or still marked unreleased, and otherwise prints that section
as the release notes.

Run:  python -m pytest tests/test_release_notes.py -q
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
release_notes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes)


def _repo(tmp_path, version="2.1.0", cff="2.1.0", heading="## 2.1.0 — 2026-10-20"):
    (tmp_path / "zfisher").mkdir()
    (tmp_path / "zfisher" / "version.py").write_text(f'__version__ = "{version}"\n')
    if cff:
        (tmp_path / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: {cff}\n")
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n{heading}\n\nFaster saves.\n\n### Fixes\n\n- One.\n\n## 2.0.1\n\nOld.\n")
    return tmp_path


def test_notes_are_the_version_section(tmp_path):
    notes = release_notes.release_notes("v2.1.0", _repo(tmp_path))
    assert notes.startswith("Faster saves.") and "- One." in notes and "Old." not in notes


@pytest.mark.parametrize("kwargs, tag, message", [
    ({}, "v2.1.1", "does not match"),
    ({"cff": "2.0.1"}, "v2.1.0", "CITATION.cff says version 2.0.1"),
    ({"heading": "## 2.1.0 — unreleased"}, "v2.1.0", "still marks 2.1.0 as unreleased"),
    ({"heading": "## 2.2.0 — 2026-10-20", "version": "2.1.0"}, "v2.1.0", "no '## 2.1.0' section"),
])
def test_mismatches_are_refused_with_the_reason(tmp_path, kwargs, tag, message):
    with pytest.raises(SystemExit, match=message):
        release_notes.release_notes(tag, _repo(tmp_path, **kwargs))


def test_citation_without_a_version_is_accepted(tmp_path):
    assert release_notes.release_notes("v2.1.0", _repo(tmp_path, cff=None))


def test_runs_as_a_script_on_this_repository():
    """The workflow runs it on the real files: the current version's tag either
    gives notes or fails with a reason, never a traceback. Which one depends on
    the repository's state (no changelog yet, a section still unreleased, or a
    dated one), so any clear refusal passes."""
    version = release_notes.package_version()
    result = subprocess.run([sys.executable, str(SCRIPT), f"v{version}"], capture_output=True, text=True)
    assert "Traceback" not in result.stderr
    if result.returncode == 0:
        assert result.stdout.strip()
    else:
        assert result.stderr.strip()
