"""The server's triage and operate modes, called through an MCP client.

Each test works on its own copy of a small synthetic batch session
(``tests/test_stages.py``). Each mode offers only its tools; every tool that
writes is a dry run unless told otherwise, and a dry run changes nothing; a
setting outside its range is refused; ``update`` runs what ``set_param``'s dry
run said it would; every change is logged with the server's actor and the
client's name; nothing outside the root can be read or written.

Run:  python -m pytest packages/zfisher-mcp/tests -q
"""
import asyncio
import json
import logging
import shutil
import sys

import pytest

pytest.importorskip("mcp")
pytest.importorskip("zfisher_mcp")
pytest.importorskip("SimpleITK")

from mcp import Client, StdioServerParameters  # noqa: E402
from mcp.types import Implementation  # noqa: E402

from zfisher import constants  # noqa: E402
from zfisher.core import api, session  # noqa: E402
from zfisher_mcp.paths import Roots  # noqa: E402
from zfisher_mcp.server import ServerConfig, build_server  # noqa: E402

from test_batch_template import _workbook  # noqa: E402
from test_stages import CHANNEL, finished  # noqa: F401,E402  (fixture)

R1, R2 = f"R1 - {CHANNEL}", f"R2 - {CHANNEL}"
CLIENT = Implementation(name="pytest-client", version="0")
READ = {"list_sessions", "get_session_info", "get_status", "get_parameters", "get_metrics", "query_distances",
        "query_report", "get_provenance", "get_decisions", "read_log"}
TRIAGE = {"flag_batch", "propose"}
OPERATE = {"set_param", "update", "detect_channel", "transform_channel", "export_report", "export_cells",
           "threshold_sweep", "run_batch"}
RULE = [{"source": R1, "target": R2, "threshold": 1.0}]


@pytest.fixture
def root(finished, tmp_path):
    """``lab/a`` with a report, ``lab/b`` without, and the raw rounds in ``lab/raw``."""
    root = tmp_path / "lab"
    shutil.copytree(finished / "out", root / "a")
    shutil.copytree(finished / "out", root / "b")
    (root / "raw").mkdir()
    for name in ("R1.ome.tif", "R2.ome.tif"):
        shutil.copy(finished / name, root / "raw" / name)
    factory = logging.getLogRecordFactory()
    levels = {name: logging.getLogger(name).level for name in ("zfisher", "stdout", "stderr")}
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    a = api.open_session(root / "a", read_only=False)
    a.set("colocalization_rules", RULE)                   # kept with the session, as the GUI keeps them
    api.export_report(a)
    yield root
    logging.setLogRecordFactory(factory)
    for name, level in levels.items():
        logging.getLogger(name).setLevel(level)
    session.clear_session()


def server_for(root, mode="operate", **config):
    return build_server(ServerConfig(roots=Roots.parse([str(root)]), mode=mode, actor="agent:test", **config))


def run(server, script):
    async def go():
        async with Client(server, client_info=CLIENT) as client:
            return await script(client)
    return asyncio.run(go())


def call(server, tool, **arguments):
    result = run(server, lambda c: c.call_tool(tool, arguments))
    assert not result.is_error, result.content[0].text
    return result.structured_content


def error(server, tool, **arguments):
    result = run(server, lambda c: c.call_tool(tool, arguments))
    assert result.is_error, result.structured_content
    return result.content[0].text


def snapshot(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


def no_absolute_paths(value, root):
    text = json.dumps(value)
    return str(root) not in text and str(root.resolve()) not in text


def test_each_mode_offers_its_own_tools(root):
    names = {mode: {t.name for t in run(server_for(root, mode), lambda c: c.list_tools()).tools}
             for mode in ("query", "triage", "operate")}
    assert names == {"query": READ, "triage": READ | TRIAGE, "operate": READ | TRIAGE | OPERATE}
    assert "set_param" in error(server_for(root, "triage"), "set_param", session="a", scope="registration",
                                name="apply_warp", value=False, justification="x")
    operate = {t.name: t for t in run(server_for(root), lambda c: c.list_tools()).tools}
    assert not operate["update"].annotations.read_only_hint and operate["update"].annotations.destructive_hint
    assert operate["get_status"].annotations.read_only_hint


def test_dry_runs_say_what_would_happen_and_change_nothing(root):
    server = server_for(root)
    before = snapshot(root)
    detect = call(server, "detect_channel", session="a", round="R1", channel=CHANNEL)
    assert detect["dry_run"] and detect["writes"] == [f"a/reports/R1_-_{CHANNEL}_puncta.csv"]
    assert detect["makes_stale"] == [f"a/reports/Aligned_R1_-_{CHANNEL}_puncta.csv",
                                     "a/reports/Master_Analysis_Report.xlsx"]
    assert call(server, "transform_channel", session="a", round="R2", channel=CHANNEL)["makes_stale"] == [
        "a/reports/Master_Analysis_Report.xlsx"]
    assert call(server, "export_report", session="b")["writes"][0] == "b/reports/Master_Analysis_Report.xlsx"
    assert call(server, "export_report", session="a")["writes"][0] == "a/reports/Master_Analysis_Report_2.xlsx"
    assert call(server, "export_cells", session="a")["writes"] == ["a/cells"]
    change = call(server, "set_param", session="a", scope="detection", name="sensitivity", value=0.2,
                  channel=CHANNEL, justification="testing")
    assert change["dry_run"] and change["result"]["applied"] is False and change["result"]["new"] == 0.2
    assert change["would_run"] == [f"detect {R1}", f"detect {R2}", f"transform {R1}", f"transform {R2}"]
    assert change["writes"] == ["a/sessions", "a/decisions.jsonl"] and change["result"]["rounds"] == ["R1", "R2"]
    assert call(server, "update", session="a") == {"dry_run": True, "writes": [], "makes_stale": [],
                                                   "would_run": [], "result": None}
    assert snapshot(root) == before


def test_a_setting_change_then_update_runs_what_was_said(root):
    server = server_for(root)
    change = call(server, "set_param", session="b", scope="consensus", name="overlap_method", value="Union",
                  justification="nuclei are under-segmented", dry_run=False)
    assert change["result"]["applied"] and change["result"]["old"] == "Intersection"
    assert change["would_run"] == ["consensus", f"transform {R1}", f"transform {R2}"]
    assert change["writes"][1] == "b/decisions.jsonl" and (root / change["writes"][0]).is_file()
    assert change["writes"][0] == f"b/sessions/{api.session_files(root / 'b')[-1].name}"
    assert call(server, "get_status", session="b")["update_would_run"] == change["would_run"]
    files = api.session_files(root / "b")
    again = call(server, "set_param", session="b", scope="consensus", name="overlap_method", value="Union",
                 justification="already so", dry_run=False)
    assert again["result"]["unchanged"] and again["writes"] == [] and api.session_files(root / "b") == files
    entry = api.get_decisions(api.open_session(root / "b"), "overlap_method")[-1]
    assert (entry["actor"], entry["via"], entry["justification"]) == (
        "agent:test", "mcp:pytest-client", "nuclei are under-segmented")
    assert call(server, "update", session="b")["would_run"] == change["would_run"]
    ran = call(server, "update", session="b", dry_run=False)
    assert ran["result"]["stages_run"] == change["would_run"] and not ran["dry_run"]
    assert call(server, "update", session="b")["would_run"] == []
    assert call(server, "get_status", session="b")["update_would_run"] == []


def test_settings_outside_their_range_are_refused(root):
    server = server_for(root)
    before = snapshot(root)
    assert "less than or equal" in error(server, "set_param", session="a", scope="registration",
                                         name="max_ransac_distance", value=500, justification="x", dry_run=False)
    assert "Intersection" in error(server, "set_param", session="a", scope="consensus", name="overlap_method",
                                   value="Sideways", justification="x")
    assert "justification" in error(server, "set_param", session="a", scope="consensus",
                                    name="overlap_method", value="Union", justification="")
    assert "less than or equal" in error(server, "detect_channel", session="a", round="R1", channel=CHANNEL,
                                         params={"threshold_rel": 5}, dry_run=False)
    assert "round" in error(server, "detect_channel", session="a", round="R3", channel=CHANNEL)
    assert "its channels are DAPI, FITC" in error(server, "detect_channel", session="a", round="R1", channel="Cy5")
    assert "channel of R2, not of R1" in error(server, "threshold_sweep", session="a", round="R1", channel=R2)
    assert call(server, "detect_channel", session="a", round="R1", channel=R1)["writes"] == [
        f"a/reports/R1_-_{CHANNEL}_puncta.csv"]
    assert "go up a folder" in error(server, "update", session="../a", dry_run=False)
    assert snapshot(root) == before


def test_a_channel_detected_again_warped_and_reported(root):
    server = server_for(root)
    detect = call(server, "detect_channel", session="a", round="R1", channel=CHANNEL, dry_run=False,
                  params={"method": "Local Maxima", "threshold_rel": 0.05, "min_distance": 2})
    assert detect["result"]["n_puncta"] > 0 and detect["result"]["csv"] == detect["writes"][0]
    states = {a["name"]: a["state"] for a in call(server, "get_status", session="a")["artifacts"]}
    assert states[f"Aligned R1 - {CHANNEL}_puncta"] == "stale"
    warped = call(server, "transform_channel", session="a", round="R1", channel=CHANNEL, dry_run=False)
    assert warped["result"]["n_puncta"] > 0
    report = call(server, "export_report", session="a", dry_run=False)
    assert report["result"]["path"] == report["writes"][0] == "a/reports/Master_Analysis_Report_2.xlsx"
    assert report["result"]["sheets"]["Colocalization"] > 0
    assert all(a["state"] == "fresh" for a in call(server, "get_status", session="a")["artifacts"])
    sweep = call(server, "threshold_sweep", session="a", round="R2", channel=CHANNEL)
    assert sweep["channel"] == R2 and len(sweep["counts"]) == len(sweep["thresholds"]) > 3
    cells = call(server, "export_cells", session="a", ids=[1], dry_run=False)
    assert cells["result"]["n_cells"] == 1 and cells["result"]["paths"][0].startswith("a/cells/")
    for answer in (detect, warped, report, sweep, cells):
        assert no_absolute_paths(answer, root)


def test_triage_flags_and_proposes_without_changing_the_sessions(root):
    server = server_for(root, "triage")
    sessions_before = {p: snapshot(root / p) for p in ("a/sessions", "b/sessions")}
    flags = call(server, "flag_batch")
    assert flags["n_datasets"] == 2 and flags["report"] is None and not (root / "qc_report.md").exists()
    assert flags["compared_with_batch"] is False and "at least 4" in flags["note"]
    written = call(server, "flag_batch", write_report=True)
    assert written["report"] == "qc_report.md" and (root / "qc_report.md").exists()
    proposal = call(server, "propose", session="a", name="threshold_rel", value=0.08, channel=CHANNEL,
                    round="R1", justification="SNR is low")["record"]
    assert proposal["status"] == "proposed" and proposal["via"] == "mcp:pytest-client"
    assert proposal["actor"] == "agent:test" and proposal["scope"] == {"round": "R1", "channel": CHANNEL}
    assert api.get_decisions(api.open_session(root / "a"))[-1]["status"] == "proposed"
    assert {p: snapshot(root / p) for p in ("a/sessions", "b/sessions")} == sessions_before


def test_a_batch_runs_from_a_template_inside_the_root(root, tmp_path):
    raw = root / "raw"
    _workbook(root / "batch.xlsx", [{"Name": "c", "R1": str(raw / "R1.ome.tif"), "R2": str(raw / "R2.ome.tif")}])
    server = server_for(root, max_workers=1)
    plan = call(server, "run_batch", template="batch.xlsx", output="runs", workers=4)
    assert plan["dry_run"] and plan["writes"] == ["runs"] and not (root / "runs").exists()
    (dataset,) = plan["result"]["datasets"]
    assert dataset["new"] and dataset["output_dir"] == "runs/c" and dataset["r1"] == "raw/R1.ome.tif"
    done = call(server, "run_batch", template="batch.xlsx", output="runs", workers=4, dry_run=False)
    (result,) = done["result"]["datasets"]
    assert result["ok"] and result["stages_run"][0] == "segment" and done["result"]["qc_report"] == "runs/qc_report.md"
    again = call(server, "run_batch", template="batch.xlsx", output="runs")
    assert again["result"]["datasets"][0]["would_run"] == []
    shutil.copy(raw / "R1.ome.tif", tmp_path / "R1.ome.tif")
    _workbook(root / "outside.xlsx", [{"Name": "d", "R1": str(tmp_path / "R1.ome.tif"),
                                       "R2": str(raw / "R2.ome.tif")}])
    message = error(server, "run_batch", template="outside.xlsx", output="runs")
    assert "its R1 image is outside" in message and str(tmp_path) not in message
    assert "go up a folder" in error(server, "run_batch", template="batch.xlsx", output="../elsewhere")


def test_operate_mode_over_stdio_runs_the_pipeline(root, tmp_path):
    params = StdioServerParameters(command=sys.executable, cwd=str(tmp_path), args=[
        "-m", "zfisher_mcp", "--root", str(root), "--mode", "operate", "--actor", "agent:stdio-test"])

    async def go():
        async with Client(params, client_info=CLIENT) as client:
            change = await client.call_tool("set_param", {
                "session": "b", "scope": "registration", "name": "max_ransac_distance", "value": 7,
                "justification": "over stdio", "dry_run": False})
            ran = await client.call_tool("update", {"session": "b", "dry_run": False})
            status = await client.call_tool("get_status", {"session": "b"})
            return change, ran, status
    change, ran, status = asyncio.run(go())
    assert not change.is_error and not ran.is_error, (change.content, ran.content)
    assert ran.structured_content["result"]["stages_run"] == change.structured_content["would_run"]
    assert all(st["state"] == "fresh" for st in status.structured_content["stages"])
    entry = api.get_decisions(api.open_session(root / "b"), "max_ransac_distance")[-1]
    assert (entry["actor"], entry["via"]) == ("agent:stdio-test", "mcp:pytest-client")
    assert (root / "b" / constants.SESSIONS_DIR).exists()
