"""Serving over HTTP behind a bearer token, and the host a program running
the server gives it (``zfisher_mcp.httpserver``, ``zfisher_mcp.host``).

Run:  python -m pytest packages/zfisher-mcp/tests -q
"""
import asyncio
import logging
import re
import shutil
import subprocess
import sys
import threading

import pytest

pytest.importorskip("mcp")
pytest.importorskip("zfisher_mcp")
pytest.importorskip("SimpleITK")

import httpx2  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402
from mcp.shared._httpx_utils import create_mcp_http_client  # noqa: E402
from mcp.types import Implementation  # noqa: E402

from zfisher.core import api, session  # noqa: E402
from zfisher_mcp.__main__ import main  # noqa: E402
from zfisher_mcp.host import DiskHost  # noqa: E402
from zfisher_mcp.httpserver import HttpServer  # noqa: E402
from zfisher_mcp.paths import Roots  # noqa: E402
from zfisher_mcp.server import ServerConfig, build_server  # noqa: E402

from test_stages import CHANNEL, finished  # noqa: F401,E402  (fixture)

CLIENT = Implementation(name="http-client", version="0")


@pytest.fixture
def root(finished, tmp_path):
    root = tmp_path / "lab"
    shutil.copytree(finished / "out", root / "a")
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    yield root
    logging.setLogRecordFactory(factory)
    session.clear_session()


def over_http(url, token, script):
    async def go():
        transport = streamable_http_client(url, http_client=create_mcp_http_client(
            headers={"Authorization": f"Bearer {token}"}))
        async with Client(transport, client_info=CLIENT) as client:
            return await script(client)
    return asyncio.run(go())


def test_http_answers_only_with_the_token(root):
    http = HttpServer(build_server(ServerConfig(roots=Roots.parse([str(root)]))), token="s3cret").start()
    try:
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/mcp", http.url) and http.running
        listed = over_http(http.url, "s3cret", lambda c: c.call_tool("list_sessions", {}))
        assert [e["session"] for e in listed.structured_content["sessions"]] == ["a"]
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "s3cret"}):
            response = httpx2.post(http.url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                   headers={"Accept": "application/json, text/event-stream", **headers})
            assert response.status_code == 401
    finally:
        http.stop()
    assert not http.running
    with pytest.raises(Exception):
        httpx2.post(http.url, json={}, headers={"Authorization": "Bearer s3cret"}, timeout=2)


def test_each_start_makes_a_new_token():
    roots = Roots.parse([str(__import__("tempfile").mkdtemp())])
    first = HttpServer(build_server(ServerConfig(roots=roots)))
    second = HttpServer(build_server(ServerConfig(roots=roots)))
    assert first.token != second.token and len(first.token) >= 24 and first.port != second.port
    for http in (first, second):
        http.stop()


class RecordingHost(DiskHost):
    """Runs every change on a thread of its own, as a GUI does, and notes it."""

    def __init__(self, refuse=None):
        self.calls, self.refuse = [], refuse

    def run_write(self, what, folder, fn):
        if self.refuse and what.startswith(self.refuse):
            raise ValueError(f"zFISHer is busy; {what} must wait.")
        self.calls.append((what, folder.name))
        out = {}
        thread = threading.Thread(target=lambda: out.setdefault("result", fn()))
        thread.start()
        thread.join()
        return out["result"]


def test_changes_go_through_the_host_with_the_actor(root):
    host = RecordingHost(refuse="export_report")
    server = build_server(ServerConfig(roots=Roots.parse([str(root)]), mode="operate", actor="agent:gui-test",
                                       host=host))

    async def go():
        async with Client(server, client_info=CLIENT) as client:
            detect = await client.call_tool("detect_channel", {"session": "a", "round": "R1", "channel": CHANNEL,
                                                               "dry_run": False})
            refused = await client.call_tool("export_report", {"session": "a", "dry_run": False})
            return detect, refused
    detect, refused = asyncio.run(go())
    assert not detect.is_error and detect.structured_content["result"]["n_puncta"] > 0
    assert host.calls == [(f"detect R1 - {CHANNEL}", "a")]
    assert refused.is_error and "zFISHer is busy" in refused.content[0].text
    entry = api.get_decisions(api.open_session(root / "a"))[-1]
    assert (entry["actor"], entry["via"]) == ("agent:gui-test", "mcp:http-client")    # across the host's thread


def test_the_command_serves_http_and_prints_its_token(root, tmp_path):
    process = subprocess.Popen([sys.executable, "-m", "zfisher_mcp", "--root", str(root), "--http", "127.0.0.1:0"],
                               stderr=subprocess.PIPE, text=True, cwd=str(tmp_path))
    try:
        url = token = None
        while url is None or token is None:
            line = process.stderr.readline()
            assert line, "the server ended before printing its address"
            url = url or (re.search(r"at (http://\S+)", line) or [None, None])[1]
            token = token or (re.search(r"bearer token (\S+)", line) or [None, None])[1]
        info = over_http(url, token, lambda c: c.call_tool("get_session_info", {"session": "a"}))
        assert info.structured_content["output_dir"] == "a"
    finally:
        process.terminate()
        process.wait(timeout=20)


def test_the_command_will_not_listen_beyond_this_machine_unless_told(tmp_path, capsys):
    for argv, expected in ((["--http", "0.0.0.0:8765"], "--allow-remote"), (["--http", "nonsense"], "[HOST:]PORT")):
        with pytest.raises(SystemExit):
            main(["--root", str(tmp_path), *argv])
        assert expected in capsys.readouterr().err
