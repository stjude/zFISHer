"""Preparing a release from the changelog's Unreleased section.

``.github/scripts/prepare_release.py`` sets the new version in
``zfisher/version.py`` and ``CITATION.cff``, dates the Unreleased heading, and
leaves files the release workflow's check accepts; it changes nothing when it
refuses.

Run:  python -m pytest tests/test_prepare_release.py -q
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
spec = importlib.util.spec_from_file_location("prepare_release", SCRIPTS / "prepare_release.py")
prepare_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_release)
release_notes = prepare_release.release_notes

CHANGELOG = ("# Changelog\n\n## Unreleased\n\n### Fixes\n\n- No more crash.\n\n"
             "## 2.1.0 — 2026-10-13\n\nOld.\n")


def _repo(tmp_path, changelog=CHANGELOG):
    (tmp_path / "zfisher").mkdir()
    (tmp_path / "zfisher" / "version.py").write_text('"""Version."""\n\n__version__ = "2.1.0"\n\n\ndef f():\n    pass\n')
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\nversion: 2.1.0\ndate-released: 2026-10-13\nlicense: MIT\n")
    (tmp_path / "CHANGELOG.md").write_text(changelog)
    return tmp_path


def _files(root):
    return {p.name: p.read_text() for p in [root / "zfisher" / "version.py", root / "CITATION.cff",
                                             root / "CHANGELOG.md"]}


@pytest.mark.parametrize("bump, version", [("patch", "2.1.1"), ("minor", "2.2.0"),
                                            ("major", "3.0.0"), ("2.3.0", "2.3.0")])
def test_next_version(bump, version):
    assert prepare_release.next_version("2.1.0", bump) == version


def test_prepares_files_the_release_check_accepts(tmp_path):
    root = _repo(tmp_path)
    assert prepare_release.prepare("patch", "2026-10-20", root) == "2.1.1"
    files = _files(root)
    assert '__version__ = "2.1.1"' in files["version.py"] and "def f():" in files["version.py"]
    assert "version: 2.1.1\n" in files["CITATION.cff"] and "date-released: 2026-10-20\n" in files["CITATION.cff"]
    assert "cff-version: 1.2.0\n" in files["CITATION.cff"]
    assert files["CHANGELOG.md"] == CHANGELOG.replace("## Unreleased", "## 2.1.1 — 2026-10-20")
    assert release_notes.release_notes("v2.1.1", root).strip() == "### Fixes\n\n- No more crash."


@pytest.mark.parametrize("changelog, bump, message", [
    (CHANGELOG.replace("## Unreleased\n\n### Fixes\n\n- No more crash.\n\n", ""), "patch", "no '## Unreleased'"),
    (CHANGELOG.replace("### Fixes\n\n- No more crash.\n", ""), "patch", "Unreleased section is empty"),
    (CHANGELOG, "2.1.0", "not higher"),
    (CHANGELOG, "2.0.9", "not higher"),
    (CHANGELOG, "2.1", "not a version"),
    (CHANGELOG.replace("## 2.1.0 — 2026-10-13", "## 2.1.1 — 2026-10-13"), "patch", "already has a section"),
])
def test_refuses_and_changes_nothing(tmp_path, changelog, bump, message):
    root = _repo(tmp_path, changelog)
    before = _files(root)
    with pytest.raises(SystemExit, match=message):
        prepare_release.prepare(bump, "2026-10-20", root)
    assert _files(root) == before
