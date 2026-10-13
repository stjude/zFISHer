"""Check a release tag against the repository and print its release notes.

Usage:  python .github/scripts/release_notes.py v2.1.0 > notes.md

Fails, naming the problem, unless:
- the tag is ``v`` followed by ``__version__`` in ``zfisher/version.py``;
- ``CITATION.cff`` has the same ``version``, if it has one;
- ``CHANGELOG.md`` has a ``## <version>`` section whose heading is dated, not
  still marked "unreleased".

Prints that section's body, the release notes. Run by the release workflow
before it creates the draft release; runs the same way on any machine.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def package_version(root=ROOT):
    text = (root / "zfisher" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise SystemExit("zfisher/version.py has no __version__")
    return match.group(1)


def citation_version(root=ROOT):
    path = root / "CITATION.cff"
    if not path.exists():
        return None
    match = re.search(r"^version:\s*['\"]?([^'\"\s]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def changelog_section(version, root=ROOT):
    """The heading line and body of ``## <version>`` in CHANGELOG.md."""
    path = root / "CHANGELOG.md"
    if not path.exists():
        raise SystemExit("CHANGELOG.md is missing")
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = re.compile(rf"^##\s+v?{re.escape(version)}(\s|$)")
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        raise SystemExit(f"CHANGELOG.md has no '## {version}' section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return lines[start], "\n".join(lines[start + 1:end]).strip() + "\n"


def release_notes(tag, root=ROOT):
    version = package_version(root)
    if tag != f"v{version}":
        raise SystemExit(f"Tag {tag} does not match zfisher/version.py, which says {version}")
    cff = citation_version(root)
    if cff is not None and cff != version:
        raise SystemExit(f"CITATION.cff says version {cff}, not {version}")
    heading, body = changelog_section(version, root)
    if "unreleased" in heading.lower():
        raise SystemExit(f"CHANGELOG.md still marks {version} as unreleased: {heading!r}. "
                         "Put the release date in its heading.")
    if not body.strip():
        raise SystemExit(f"CHANGELOG.md's {version} section is empty")
    return body


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sys.stdout.write(release_notes(sys.argv[1]))
