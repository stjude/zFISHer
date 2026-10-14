"""Turn the changelog's Unreleased section into a release, ready to commit.

Usage:  python .github/scripts/prepare_release.py patch|minor|major|<X.Y.Z> [--date YYYY-MM-DD]

``patch`` makes 2.1.0 into 2.1.1, ``minor`` into 2.2.0, ``major`` into 3.0.0.
It sets the version in ``zfisher/version.py`` and ``CITATION.cff`` (and the
citation's ``date-released``), renames ``## Unreleased`` in CHANGELOG.md to
``## <version> — <date>``, checks the result with ``release_notes.py`` (the
check the release workflow runs on the tag), and prints the git commands that
are left to run. It runs no git command itself. Once the release is merged into
main, the release workflow tags it and creates the draft release.

Refuses, changing nothing, if the Unreleased section is missing or empty, the
new version is not higher than the current one, or the changelog already has
a section for it.
"""
import argparse
import datetime
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("release_notes", Path(__file__).with_name("release_notes.py"))
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

UNRELEASED = re.compile(r"^##[ \t]+Unreleased[ \t]*$", re.IGNORECASE | re.MULTILINE)


def parse_version(text):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if not match:
        raise SystemExit(f"{text!r} is not a version like 2.1.1")
    return tuple(int(part) for part in match.groups())


def next_version(current, bump):
    major, minor, patch = parse_version(current)
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "major":
        return f"{major + 1}.0.0"
    if parse_version(bump) <= (major, minor, patch):
        raise SystemExit(f"{bump} is not higher than the current version {current}")
    return bump


def prepare(bump, date, root=ROOT):
    """Write the release into the files under ``root``; returns the new version."""
    current = release_notes.package_version(root)
    version = next_version(current, bump)

    changelog_path = root / "CHANGELOG.md"
    changelog = changelog_path.read_text(encoding="utf-8")
    heading = UNRELEASED.search(changelog)
    if heading is None:
        raise SystemExit("CHANGELOG.md has no '## Unreleased' section to release")
    following = changelog.find("\n## ", heading.end())
    body = changelog[heading.end():following if following != -1 else len(changelog)]
    if not body.strip():
        raise SystemExit("CHANGELOG.md's Unreleased section is empty")
    if re.search(rf"^##\s+v?{re.escape(version)}(\s|$)", changelog, re.MULTILINE):
        raise SystemExit(f"CHANGELOG.md already has a section for {version}")

    version_path = root / "zfisher" / "version.py"
    version_py = re.sub(r'^(__version__\s*=\s*)["\'][^"\']+["\']', rf'\g<1>"{version}"',
                        version_path.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE)
    citation_path = root / "CITATION.cff"
    citation = None
    if citation_path.exists():
        citation = citation_path.read_text(encoding="utf-8")
        citation = re.sub(r"^version:.*$", f"version: {version}", citation, count=1, flags=re.MULTILINE)
        citation = re.sub(r"^date-released:.*$", f"date-released: {date}", citation, count=1, flags=re.MULTILINE)

    # Every check passed: write all three files.
    changelog_path.write_text(changelog[:heading.start()] + f"## {version} — {date}" + changelog[heading.end():],
                              encoding="utf-8")
    version_path.write_text(version_py, encoding="utf-8")
    if citation is not None:
        citation_path.write_text(citation, encoding="utf-8")
    release_notes.release_notes(f"v{version}", root)       # what the release workflow will check
    return version


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bump", help="patch, minor, major, or a version such as 2.1.1")
    parser.add_argument("--date", default=datetime.date.today().isoformat(),
                        help="release date, YYYY-MM-DD (default: today)")
    args = parser.parse_args(argv)
    datetime.date.fromisoformat(args.date)
    old = release_notes.package_version(ROOT)
    version = prepare(args.bump, args.date, ROOT)
    print(f"Prepared zFISHer {version} (was {old}), dated {args.date}.\n")
    print("Review the changes:\n  git diff\n")
    print("Then commit on v2fixes, push, and merge v2fixes into main with a pull request:")
    print(f'  git commit -am "Release {version}"')
    print("  git push origin v2fixes\n")
    print(f"After the merge, GitHub tags main v{version}, runs the tests and creates a draft")
    print("release (Actions > Release). Review the draft under Releases and click Publish.")


if __name__ == "__main__":
    sys.exit(main())
