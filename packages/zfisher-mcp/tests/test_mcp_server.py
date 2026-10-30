"""The MCP server, called through an MCP client.

A batch root holds two copies of a small synthetic batch session
(``tests/test_stages.py``), one with an exported report. Each test connects a
client to the server in process (the SDK's in-memory transport) and calls its
tools as any MCP client would; one test runs the ``zfisher-mcp`` command over
stdio. The server must answer from the files, name every file relative to the
root, refuse anything outside it, and write nothing.

Run:  python -m pytest packages/zfisher-mcp/tests -q
"""
import asyncio
import json
import logging
import os
import shutil
import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("zfisher_mcp")
pytest.importorskip("SimpleITK")

from mcp import Client, StdioServerParameters  # noqa: E402

from zfisher import constants  # noqa: E402
from zfisher.core import api, session  # noqa: E402
from zfisher_mcp.__main__ import main  # noqa: E402
from zfisher_mcp.paths import Roots  # noqa: E402
from zfisher_mcp.server import SHEETS, ServerConfig, build_server  # noqa: E402

from test_stages import CHANNEL, finished  # noqa: F401,E402  (fixture)

R1, R2 = f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"
READ_TOOLS = {"list_sessions", "get_session_info", "get_status", "get_parameters", "get_metrics",
              "query_distances", "query_report", "get_provenance", "get_decisions", "read_log"}


@pytest.fixture(scope="module")
def root(finished, tmp_path_factory):
    """``batch/dataset_a`` (with a report) and ``batch/group/dataset_b`` (without)."""
    root = tmp_path_factory.mktemp("Smith lab") / "batch"
    shutil.copytree(finished / "out", root / "dataset_a")
    shutil.copytree(finished / "out", root / "group" / "dataset_b")
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    try:
        api.export_report(api.open_session(root / "dataset_a", read_only=False),
                          rules=[{"source": R1, "target": R2, "threshold": 1.0}])
    finally:
        logging.setLogRecordFactory(factory)
        session.clear_session()
    return root


@pytest.fixture(scope="module")
def server(root):
    return build_server(ServerConfig(roots=Roots.parse([str(root)]), page_size=20))


def run(server, script):
    """``script(client)`` run against ``server`` through an in-process client."""
    async def go():
        async with Client(server) as client:
            return await script(client)
    return asyncio.run(go())


def call(server, tool, **arguments):
    """The structured result of one tool call; fails the test on a tool error."""
    result = run(server, lambda c: c.call_tool(tool, arguments))
    assert not result.is_error, result.content[0].text
    return result.structured_content


def error(server, tool, **arguments):
    """The message of a call that must fail."""
    result = run(server, lambda c: c.call_tool(tool, arguments))
    assert result.is_error, result.structured_content
    return result.content[0].text


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from strings(k)
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def every_answer(server):
    """One call of every tool, as a client would make them."""
    return {
        "list_sessions": call(server, "list_sessions"),
        "get_session_info": call(server, "get_session_info", session="dataset_a"),
        "get_status": call(server, "get_status", session="dataset_a"),
        "get_parameters": call(server, "get_parameters", session="dataset_a"),
        "get_metrics": call(server, "get_metrics", session="dataset_a", include_saturation=True),
        "query_distances": call(server, "query_distances", session="dataset_a", source=R1, target=R2,
                                max_um=0.8, bin_um=0.2),
        "query_report": call(server, "query_report", session="dataset_a", sheet="Distances", limit=5),
        "get_provenance": call(server, "get_provenance", session="dataset_a"),
        "get_decisions": call(server, "get_decisions", session="dataset_a"),
        "read_log": call(server, "read_log", session="dataset_a", kind="STAGE"),
    }


def test_query_mode_offers_the_read_only_tools_only(server):
    tools = run(server, lambda c: c.list_tools()).tools
    assert {t.name for t in tools} == READ_TOOLS
    assert all(t.annotations.read_only_hint and not t.annotations.destructive_hint for t in tools)
    assert all(t.output_schema and t.description for t in tools)
    with pytest.raises(ValueError, match="No mode 'admin'"):
        build_server(ServerConfig(roots=Roots.parse([str(os.getcwd())]), mode="admin"))


def test_sessions_are_listed_by_their_path_below_the_root(server):
    listed = call(server, "list_sessions")
    assert listed["total"] == 2
    assert [e["session"] for e in listed["sessions"]] == ["dataset_a", "group/dataset_b"]
    assert listed["sessions"][0]["session_file"] == "zfisher_session_2.json"      # the export wrote session 2
    second = call(server, "list_sessions", limit=1, offset=1)
    assert second["total"] == 2 and [e["session"] for e in second["sessions"]] == ["group/dataset_b"]
    assert [e["session"] for e in call(server, "list_sessions", under="group")["sessions"]] == ["group/dataset_b"]


def test_what_a_session_is_and_what_is_out_of_date(server):
    info = call(server, "get_session_info", session="dataset_a")
    assert info["output_dir"] == "dataset_a" and info["read_only"]
    assert info["session_file"] == "zfisher_session_2.json" and info["channels"] == [R1, R2]
    assert info["r1_path"] == "R1.ome.tif"                                   # raw image outside the root: name only
    status = call(server, "get_status", session="group/dataset_b")
    assert {a["kind"] for a in status["artifacts"]} == {"aligned_puncta", "report"}
    assert all(a["path"] is None or a["path"].startswith("group/dataset_b/") for a in status["artifacts"])
    assert status["stages"] and all(st["state"] == "fresh" for st in status["stages"])
    params = call(server, "get_parameters", session="dataset_a", channel=R2)
    assert set(params["detection"]) == {R2} and params["registration"]["rigid"]["method"]


def test_distances_and_report_rows_match_the_core_api(server, root):
    s = api.open_session(root / "dataset_a")
    direct = api.query_distances(s, R1, R2, 0.8, 0.2).model_dump()
    assert call(server, "query_distances", session="dataset_a", source=R1, target=R2, max_um=0.8,
                bin_um=0.2) == direct
    one_bin = call(server, "query_distances", session="dataset_a", source=R1, target=R2, max_um=0.8, bin_um=0)
    assert len(one_bin["bins"]) == 1 and one_bin["n_within"] == direct["n_within"]
    rows = api.query_report(s, constants.DISTANCES_SHEET)
    page = call(server, "query_report", session="dataset_a", sheet="Distances", limit=5, offset=1)
    assert page["total_rows"] == len(rows) and page["rows"] == rows[1:6]
    nucleus = rows[0]["Nucleus_ID"]
    some = call(server, "query_report", session="dataset_a", sheet="Distances", where={"Nucleus_ID": nucleus},
                limit=1000)                                                  # capped at the page size
    assert some["total_rows"] == sum(r["Nucleus_ID"] == nucleus for r in rows) and len(some["rows"]) <= 20


def test_metrics_provenance_decisions_and_logs(server):
    metrics = call(server, "get_metrics", session="dataset_a")
    assert set(metrics["channels"]) == {R1, R2} and metrics["nuclei"]["n_nuclei"] > 0
    csv = f"{constants.REPORTS_DIR}/Aligned_R1_-_{CHANNEL}_puncta.csv"
    records = call(server, "get_provenance", session="dataset_a", file=csv)
    assert records["total"] >= 1 and all(r["path"] == csv for r in records["records"])
    everything = call(server, "get_provenance", session="dataset_a", limit=3)
    assert everything["total"] > records["total"] and len(everything["records"]) == 3
    decisions = call(server, "get_decisions", session="dataset_a")
    assert decisions["total"] == len(decisions["records"])
    stages = call(server, "read_log", session="dataset_a", kind="STAGE", limit=200)
    assert stages["total"] >= 1 and len(stages["records"]) <= 20
    assert all(r["kind"] == "STAGE" and "duration_s" in r for r in stages["records"])


def test_no_answer_names_a_folder_above_the_root(server, root):
    answers = every_answer(server)
    text = json.dumps(answers)
    assert str(root) not in text and "Smith lab" not in text and str(root.resolve()) not in text
    absolute = [s for s in strings(answers) if s.startswith(("/", "\\")) or s[1:3] in (":\\", ":/")]
    assert absolute == []


def test_tools_refuse_paths_outside_the_root(server, root, tmp_path):
    outside = tmp_path / "outside"
    shutil.copytree(root / "group" / "dataset_b", outside)
    link = root / "group" / "link_out"
    link.symlink_to(outside)
    try:
        assert "go up a folder" in error(server, "get_session_info", session="../dataset_a")
        assert "absolute path" in error(server, "get_session_info", session=str(root / "dataset_a"))
        assert "absolute path" in error(server, "get_metrics", session="/etc")
        assert "outside the root" in error(server, "get_status", session="group/link_out")
        assert "No session at 'nope'" in error(server, "get_session_info", session="nope")
        assert "relative to the session folder" in error(server, "get_provenance", session="dataset_a",
                                                         file="../../x.csv")
        assert [e["session"] for e in call(server, "list_sessions")["sessions"]] == ["dataset_a", "group/dataset_b"]
    finally:
        link.unlink()


def test_bad_arguments_are_explained(server, root):
    assert "could mean" in error(server, "get_parameters", session="dataset_a", channel=CHANNEL)
    message = error(server, "query_report", session="group/dataset_b", sheet="Stats")
    assert "no exported report" in message and str(root) not in message
    assert "no column 'Nope'" in error(server, "query_report", session="dataset_a", sheet="Stats",
                                       where={"Nope": 1})
    assert "sheet" in error(server, "query_report", session="dataset_a", sheet="Nope")       # schema
    assert "limit" in error(server, "get_decisions", session="dataset_a", limit=0)
    assert "max_um" in error(server, "query_distances", session="dataset_a", source=R1, target=R2, max_um=-1)
    assert "kind" in error(server, "read_log", session="dataset_a", kind="NOPE")


def test_answering_writes_nothing(server, root):
    def snapshot():
        return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in root.rglob("*") if p.is_file()}
    before = snapshot()
    every_answer(server)
    run(server, lambda c: c.read_resource("zfisher://session/dataset_a"))
    assert snapshot() == before


def test_resources_and_the_status_prompt(server, root):
    contents = run(server, lambda c: c.read_resource("zfisher://session/group/dataset_b")).contents
    data = json.loads(contents[0].text)
    assert data["output_dir"] == "group/dataset_b" and str(root) not in contents[0].text
    provenance = json.loads(run(server, lambda c: c.read_resource("zfisher://provenance/dataset_a")).contents[0].text)
    assert provenance and all(not r["path"].startswith("/") for r in provenance)
    templates = {t.uri_template for t in run(server, lambda c: c.list_resource_templates()).resource_templates}
    assert templates == {"zfisher://session/{+session}", "zfisher://provenance/{+session}",
                         "zfisher://decisions/{+session}"}
    with pytest.raises(Exception):
        run(server, lambda c: c.read_resource("zfisher://session/../outside"))
    prompt = run(server, lambda c: c.get_prompt("explain_status", {"session": "dataset_a"}))
    text = prompt.messages[0].content.text
    assert '"dataset_a"' in text and "get_status" in text


def test_the_report_sheets_are_zfishers():
    sheets = {getattr(constants, n) for n in dir(constants) if n.endswith("_SHEET")}
    assert set(SHEETS.__args__) == sheets


def test_the_command_serves_over_stdio(root, tmp_path):
    params = StdioServerParameters(command=sys.executable, args=["-m", "zfisher_mcp", "--root", str(root)],
                                   cwd=str(tmp_path))

    async def go():
        async with Client(params) as client:
            listed = await client.call_tool("list_sessions", {})
            info = await client.call_tool("get_session_info", {"session": "group/dataset_b"})
            return listed.structured_content, info.structured_content
    listed, info = asyncio.run(go())
    assert [e["session"] for e in listed["sessions"]] == ["dataset_a", "group/dataset_b"]
    assert info["output_dir"] == "group/dataset_b"


def test_the_command_says_what_is_wrong_with_its_options(tmp_path, capsys):
    for argv, expected in ((["--root", str(tmp_path / "missing")], "not a folder"),
                           (["--root", str(tmp_path), "--mode", "admin"], "no mode 'admin'"),
                           (["--root", str(tmp_path), "--actor", "human"], "--actor names the agent"),
                           (["--root", str(tmp_path), "--page-size", "0"], "--page-size")):
        with pytest.raises(SystemExit) as exit_:
            main(argv)
        assert exit_.value.code == 2 and expected in capsys.readouterr().err
