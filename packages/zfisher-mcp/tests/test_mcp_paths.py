"""The server's path boundary (``zfisher_mcp.paths``): sessions are named
relative to a root, nothing outside a root can be opened, and no result names
a folder above a root.

Run:  python -m pytest packages/zfisher-mcp/tests -q
"""
import pytest

pytest.importorskip("mcp")
pytest.importorskip("zfisher_mcp")

from zfisher_mcp.paths import OutsideRoots, Roots  # noqa: E402


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "Smith lab" / "batch"
    (root / "dataset_a").mkdir(parents=True)
    (tmp_path / "Smith lab" / "batch2").mkdir()
    (tmp_path / "elsewhere").mkdir()
    return root


def test_sessions_are_named_relative_to_the_root(lab):
    roots = Roots.parse([str(lab)])
    assert roots.single and roots.resolve("dataset_a") == lab.resolve() / "dataset_a"
    assert roots.resolve("") == lab.resolve() and roots.resolve("./dataset_a/") == lab.resolve() / "dataset_a"
    assert roots.relative(lab / "dataset_a" / "reports" / "x.csv") == "dataset_a/reports/x.csv"
    assert roots.relative(lab) == "." and roots.relative(lab.parent / "batch2") is None


@pytest.mark.parametrize("bad", ["../batch2", "dataset_a/../../batch2", "/etc", "~/x", "C:\\data", "C:x",
                                 "\\\\server\\share"])
def test_nothing_outside_the_root_can_be_named(lab, bad):
    with pytest.raises(OutsideRoots):
        Roots.parse([str(lab)]).resolve(bad)


def test_a_link_out_of_the_root_is_refused(lab):
    (lab / "sneaky").symlink_to(lab.parent.parent / "elsewhere")
    with pytest.raises(OutsideRoots, match="outside the root"):
        Roots.parse([str(lab)]).resolve("sneaky")


def test_a_root_reached_through_a_link_is_recognised_by_either_name(lab, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(lab)
    roots = Roots.parse([str(link)])
    assert roots.relative(link / "dataset_a") == "dataset_a" == roots.relative(lab / "dataset_a")


def test_paths_in_results_lose_everything_above_the_root(lab):
    roots = Roots.parse([str(lab)])
    assert roots.scrub_text(str(lab / "dataset_a" / "aligned" / "R1.tif")) == "dataset_a/aligned/R1.tif"
    assert roots.scrub_text(str(lab.parent / "batch2" / "raw" / "R1 copy.nd2")) == "R1 copy.nd2"
    assert roots.scrub_text("Z:\\Smith lab\\raw\\R1.nd2") == "R1.nd2"
    message = f"No session file in {lab}/dataset_a/sessions or {lab.parent}/batch2/x.json."
    assert roots.scrub_text(message) == "No session file in dataset_a/sessions or x.json."
    assert roots.scrub_text(f"Root is {lab}.") == "Root is .."
    assert roots.scrub_text("file Z:\\lab\\raw\\R1.nd2 is missing") == "file R1.nd2 is missing"
    assert roots.scrub_text("copy /My Data/a/x.csv to /Other Data/b/y.csv") == "copy x.csv to y.csv"
    assert roots.scrub_text("0.2/0.05 µm/px, R1 - FITC") == "0.2/0.05 µm/px, R1 - FITC"
    nested = roots.scrub({"output_dir": str(lab / "dataset_a"), "files": [str(lab / "dataset_a" / "x.csv")],
                          str(lab / "k"): 3, "n": 1})
    assert nested == {"output_dir": "dataset_a", "files": ["dataset_a/x.csv"], "k": 3, "n": 1}


def test_a_root_does_not_match_a_sibling_that_starts_with_its_name(lab):
    roots = Roots.parse([str(lab)])
    sibling = lab.parent / "batch2" / "x.csv"
    assert roots.relative(sibling) is None
    assert roots.scrub_text(f"see {sibling}") == "see x.csv"


def test_several_roots_are_named(lab, tmp_path):
    other = tmp_path / "elsewhere"
    roots = Roots.parse([f"screen={lab}", f"pilot={other}"])
    assert roots.names == ["pilot", "screen"] and not roots.single
    assert roots.resolve("screen/dataset_a") == lab.resolve() / "dataset_a"
    assert roots.relative(lab / "dataset_a") == "screen/dataset_a"
    assert roots.scrub_text(str(other / "x.csv")) == "pilot/x.csv"
    with pytest.raises(OutsideRoots, match="pilot, screen"):
        roots.resolve("dataset_a")
    by_folder = Roots.parse([str(lab), str(other)])
    assert by_folder.names == ["batch", "elsewhere"]
    with pytest.raises(ValueError, match="name of its own"):
        Roots.parse([str(lab), str(tmp_path / "Smith lab" / "batch2" / ".." / "batch")])


def test_a_root_must_be_a_folder(tmp_path):
    with pytest.raises(ValueError, match="not a folder"):
        Roots.parse([str(tmp_path / "missing")])
