"""Decide whether a push makes a release, for the release workflow.

Usage:  python .github/scripts/release_plan.py <ref type> <ref name>

with ``$GITHUB_REF_TYPE`` and ``$GITHUB_REF_NAME``. Prints ``tag=v<version>``
and ``release=true`` or ``release=false``, the lines the workflow appends to
``$GITHUB_OUTPUT``, and the reason on stderr.

- A pushed version tag is released, as it always was.
- A push to main is released when ``zfisher/version.py`` names a version that
  has no tag yet: the merge brought in a release made with
  ``prepare_release.py``, and the workflow tags that commit itself. A merge
  that leaves the version alone already has its tag and releases nothing.

Whether the version, citation and changelog agree is checked afterwards by
``release_notes.py``, as for a tag pushed by hand.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("release_notes", Path(__file__).with_name("release_notes.py"))
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

RELEASE_BRANCH = "main"


def plan(ref_type, ref_name, tags, root=ROOT):
    """``(tag, release, reason)`` for a push of ``ref_name`` (a ``ref_type``
    of ``'tag'`` or ``'branch'``) to a repository whose tags are ``tags``."""
    if ref_type == "tag":
        return ref_name, True, f"tag {ref_name} was pushed"
    tag = f"v{release_notes.package_version(root)}"
    if ref_type != "branch" or ref_name != RELEASE_BRANCH:
        return tag, False, f"{ref_type} {ref_name} is not {RELEASE_BRANCH}; releases are made from {RELEASE_BRANCH}"
    if tag in tags:
        return tag, False, f"{tag} is already tagged: this merge does not change the version"
    return tag, True, f"zfisher/version.py names {tag[1:]}, which has no tag yet"


def repository_tags(root=ROOT):
    out = subprocess.run(["git", "tag", "--list"], cwd=root, capture_output=True, text=True, check=True).stdout
    return set(out.split())


def main(argv, root=ROOT, tags=None):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    tag, release, reason = plan(argv[0], argv[1], repository_tags(root) if tags is None else tags, root)
    print(reason, file=sys.stderr)
    print(f"tag={tag}")
    print(f"release={'true' if release else 'false'}")


if __name__ == "__main__":
    main(sys.argv[1:])
