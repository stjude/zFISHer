# zfisher-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that answers questions about
finished zFISHer analyses. Any MCP client, a chat assistant or a script, can ask it what a
session holds, what is out of date, how a file was made, a session's quality-control numbers,
rows of its report, and how many puncta have a neighbour within a distance.

In its default mode it reads session folders and writes nothing. Two further modes let a
client triage a batch and, on folders set aside for it, change settings and rerun stages.
Every tool calls `zfisher.core.api` and returns that function's typed result; the server holds
no analysis logic and imports nothing of zFISHer's GUI.

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
| `--mode MODE` | Which tools to offer: `query` (the default), `triage` or `operate`; see below. |
| `--actor NAME` | Who the decisions log names for changes made through the server (default `agent`; e.g. `agent:<model>`). |
| `--max-workers N` | The most datasets `run_batch` runs at once (default 1). |
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

## Modes

Only the tools of the mode the server was started in exist; a client cannot call the others.

| Mode | Tools | For |
|---|---|---|
| `query` | the read-only tools above | Questions about results; any lab member |
| `triage` | those, plus `flag_batch` and `propose` | Finding the datasets of a batch that need a look |
| `operate` | all of them, plus the tools below | Tuning settings on copies of datasets made for it |

**Triage tools.** `flag_batch(under, write_report)` compares each dataset of a batch with the
rest and with fixed floors and says which need a look and why (`write_report` also writes
`qc_report.md`). `propose(session, name, value, justification, channel, round)` logs a
suggested change in the session's decisions log, for a person to accept or dismiss; nothing else
changes.

**Operate tools.** Each takes `dry_run`, true unless the client sets it false. A dry run says
what the call would write, what it would leave out of date and which pipeline stages would run,
and changes nothing.

| Tool | Does |
|---|---|
| `set_param(session, scope, name, value, justification, channel)` | Changes one pipeline setting for the next update; refuses a value outside the setting's range or choices |
| `update(session)` | Reruns the stages not up to date with the settings and files, and those after them |
| `detect_channel(session, round, channel, params, existing)` | Detects a channel again on its raw image |
| `transform_channel(session, round, channel)` | Warps a channel's puncta into aligned space with the saved registration |
| `export_report(session)` | Writes the report from the files, with the session's rules (a new numbered file; reports are never overwritten) |
| `export_cells(session, ids, padding_um)` | Writes one OME-TIFF per nucleus |
| `threshold_sweep(session, round, channel, params)` | Counts puncta over a range of sensitivities and suggests one |
| `run_batch(template, output, workers)` | Runs a batch template; a dataset finished before runs only what is due |

A real run opens the session for writing, which saves it under the next session number as Load
Session does. Every change is logged in the session's `decisions.jsonl` with the justification,
the server's actor and the name of the client the call came from. Calls that write to one folder
run one after another. A batch can take a long time; a client may need a longer timeout for
`run_batch`.

Prompts: `explain_status(session)` in every mode, `triage_batch(batch)` from triage up and
`tune_channel(session, channel)` in operate mode.

## What the server will not do

- Open anything outside its roots. A session path that is absolute, goes up a folder (`..`),
  or leads out through a link is refused.
- Name a folder above a root. Paths in results are relative to the root (provenance records
  name files relative to their session folder), and a path outside every root, such as a raw
  image on another drive, is cut to its file name. Lab folder names carry sample and
  investigator names, and results may be passed to a model or across the network.
- Change a file in `query` mode. Sessions are opened read-only; the tests check that
  answering every tool leaves the folder byte-for-byte as it was. In `triage` mode only
  `propose` (the decisions log) and `flag_batch` with `write_report` (`qc_report.md`) write.
- Write outside its roots, or delete a file, in any mode. A batch template whose images or
  output folders lie outside the roots is refused.

## Tests

From the repository root:

```bash
python -m pytest packages/zfisher-mcp/tests -q
```

The tests build a small synthetic batch session, call every tool through an MCP client, and
run the `zfisher-mcp` command over stdio, once in `query` mode and once in `operate` mode.
