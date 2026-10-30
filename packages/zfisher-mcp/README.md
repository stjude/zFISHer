# zfisher-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that answers questions about
finished zFISHer analyses. Any MCP client, a chat assistant or a script, can ask it what a
session holds, what is out of date, how a file was made, a session's quality-control numbers,
rows of its report, and how many puncta have a neighbour within a distance.

It reads session folders and writes nothing. Every tool calls `zfisher.core.api` and returns
that function's typed result; the server holds no analysis logic and imports nothing of
zFISHer's GUI.

## Install

zFISHer is not on PyPI, so install it from this repository first, then the server, into the
same environment:

```bash
pip install -e .
pip install -e packages/zfisher-mcp
```

This adds the `zfisher-mcp` command.

## Run

An MCP client starts the server itself, over stdio. Point it at one or more folders of
sessions (a batch's output folder, say):

```bash
zfisher-mcp --root /path/to/batch_output
zfisher-mcp --root screen=/data/screen --root pilot=/data/pilot
```

Most clients take a JSON entry that names the command and its arguments, along these lines;
use the full path of `zfisher-mcp` in your environment:

```json
{
  "mcpServers": {
    "zfisher": {
      "command": "/path/to/env/bin/zfisher-mcp",
      "args": ["--root", "/path/to/batch_output"]
    }
  }
}
```

| Option | Meaning |
|---|---|
| `--root [NAME=]PATH` | A folder the server may read. Repeat for several; each then needs a name, and paths in results start with it. |
| `--mode query` | Which tools to offer. `query` (the default) has the read-only tools below. |
| `--page-size N` | The most rows or records one call returns (default 200). |
| `--max-depth N` | How many folders below a root to look for sessions (default 3). |
| `-v` | Log at DEBUG to stderr. |

## Tools (`query` mode)

Every tool except `list_sessions` takes `session`, the session's path below the root as
`list_sessions` gives it.

| Tool | Answers |
|---|---|
| `list_sessions(under, limit, offset)` | The sessions under the root, with the newest session file of each |
| `get_session_info(session)` | Inputs, zFISHer version, nuclear channel, channels with puncta, what the session has made |
| `get_status(session)` | Which outputs are fresh, stale, edited or missing, and why; the state of each pipeline stage |
| `get_parameters(session, channel)` | Detection settings per channel, colocalization rules, the saved registration |
| `get_metrics(session, include_saturation)` | Quality-control numbers: registration fit, puncta counts and SNR, extranuclear and hand-placed fractions, nucleus volumes |
| `query_distances(session, source, target, max_um, bin_um)` | How many source puncta have their nearest target punctum within `max_um` µm, binned |
| `query_report(session, sheet, where, limit, offset)` | A page of rows of one report sheet, filtered by column values |
| `get_provenance(session, file, limit, offset)` | How each file was made: inputs, settings, software version |
| `get_decisions(session, name, limit, offset)` | Who set which parameter, and when |
| `read_log(session, kind, limit, offset)` | Timing lines of the session logs: `STAGE`, `ACTION`, `STALL` or `ENV` |

`query_distances` is nearest-neighbour-within-d: each source punctum counts once, by its
nearest target punctum. Distances are in micrometres, from the images' voxel sizes.

Resources: `zfisher://session/{session}` (the session file), `zfisher://provenance/{session}`
and `zfisher://decisions/{session}`. Prompt: `explain_status(session)`.

## What the server will not do

- Open anything outside its roots. A session path that is absolute, goes up a folder (`..`),
  or leads out through a link is refused.
- Name a folder above a root. Paths in results are relative to the root (provenance records
  name files relative to their session folder), and a path outside every root, such as a raw
  image on another drive, is cut to its file name. Lab folder names carry sample and
  investigator names, and results may be passed to a model or across the network.
- Change a file. Sessions are opened read-only; the tests check that answering every tool
  leaves the folder byte-for-byte as it was.

## Tests

From the repository root:

```bash
python -m pytest packages/zfisher-mcp/tests -q
```

The tests build a small synthetic batch session, call every tool through an MCP client, and
run the `zfisher-mcp` command over stdio once.
