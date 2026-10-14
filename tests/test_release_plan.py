"""The release workflow's decision whether a push makes a release.

``.github/scripts/release_plan.py`` releases a pushed tag, and a push to main
whose ``zfisher/version.py`` names a version with no tag yet (a merged release
from ``prepare_release.py``, which the workflow then tags). Other pushes to
main, and pushes to other branches, release nothing.

Run:  python -m pytest tests/test_release_plan.py -q
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "release_plan.py"
spec = importlib.util.spec_from_file_location("release_plan", SCRIPT)
release_plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_plan)


def _repo(tmp_path, version="2.1.3"):
    (tmp_path / "zfisher").mkdir()
    (tmp_path / "zfisher" / "version.py").write_text(f'__version__ = "{version}"\n')
    return tmp_path


@pytest.mark.parametrize("ref_type, ref_name, tags, expected", [
    ("branch", "main", {"v2.1.1", "v2.1.2"}, ("v2.1.3", True)),          # a merged release
    ("branch", "main", {"v2.1.2", "v2.1.3"}, ("v2.1.3", False)),         # an ordinary merge
    ("branch", "v2fixes", set(), ("v2.1.3", False)),                     # not the release branch
    ("tag", "v2.1.3", {"v2.1.3"}, ("v2.1.3", True)),                     # tagged by hand, as before
])
def test_plan(tmp_path, ref_type, ref_name, tags, expected):
    tag, release, reason = release_plan.plan(ref_type, ref_name, tags, _repo(tmp_path))
    assert (tag, release) == expected and reason


def test_prints_the_workflow_outputs(tmp_path, capsys):
    release_plan.main(["branch", "main"], root=_repo(tmp_path), tags={"v2.1.2"})
    out = capsys.readouterr()
    assert out.out == "tag=v2.1.3\nrelease=true\n" and "no tag yet" in out.err
    with pytest.raises(SystemExit):
        release_plan.main(["main"])


def test_workflow_tags_only_after_the_check_and_only_for_main():
    """The order the comments in release.yml promise: check, then tag, then the draft."""
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((SCRIPT.parents[1] / "workflows" / "release.yml").read_text())
    assert workflow[True]["push"] == {"branches": ["main"], "tags": ["v*"]}   # YAML reads "on" as True
    steps = [s.get("name", s.get("uses")) for s in workflow["jobs"]["draft-release"]["steps"]]
    assert steps.index("Check the tag and take the release notes from CHANGELOG.md") \
        < steps.index("Tag the merge commit") < steps.index("Create the draft release")
    tag_step = next(s for s in workflow["jobs"]["draft-release"]["steps"] if s.get("name") == "Tag the merge commit")
    assert tag_step["if"] == "github.ref_type == 'branch'"
    for job in ("tests", "draft-release"):
        assert workflow["jobs"][job]["if"] == "needs.plan.outputs.release == 'true'"
