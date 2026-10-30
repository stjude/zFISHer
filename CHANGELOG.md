# Changelog

All notable changes to zFISHer. Versions follow [semantic versioning](https://semver.org/).

## Unreleased

### Fixes

- **The GUI's registration and warping write what batch writes.** Automated
  Registration & Warping and the manual Global Canvas step re-implemented the
  pipeline and differed from it: images were converted to 32-bit floats before
  alignment, so the aligned TIFFs differed from batch's in type; rounds whose
  nuclear channels are named differently (fixed for batch in 2.1.2) were not
  handled; puncta were warped from the layers on screen rather than from their
  files; and removing puncta outside the consensus mask also removed puncta
  placed by hand, which batch keeps. Both now use the steps batch uses, after
  saving any pending puncta edits, and on the test data write the same files.
  Segmentation in the GUI uses batch's size filter too.
- **Every file zFISHer writes is written whole or not at all**, as masks and
  the session file already were: puncta CSVs (rewritten on every edit),
  aligned images, centroids, reports, converted inputs and per-nucleus
  exports are written beside the old file and swapped in when complete, so a
  crash or a full disk mid-write leaves the previous version instead of a
  truncated one. All writing goes through `zfisher/core/outputs.py`.
- **Registering the same images twice gives the same shift.** RANSAC's random
  choice of centroid pairs was not seeded, so two registrations of the same
  rounds could keep different inliers and give slightly different shifts, and
  every aligned coordinate with them; the B-spline fit was already seeded.
  RANSAC is now seeded too, and the seed is recorded with the registration.
- New `puncta_id` values are handed out under the session's lock, so a
  detection running in the background and a point placed by hand at the same
  moment can no longer be given the same ids.
- **Batch Process leaves the session you have open alone.** Each dataset of a
  batch was run on the GUI's own session, so after a batch the window still
  showed your session while zFISHer's memory held the last dataset's: an edit
  made then was saved into that dataset's folder, and the session log
  continued in it. Every pipeline run, from the GUI, the CLI or a script, now
  runs on a session of its own, and the session log that was open before is
  opened again after it.
- **The batch template accepts the ranges the GUI does.** Its Sensitivity
  floor was 0.0 where Puncta Detection's is 0.01; a 0.0 cell is now clamped to
  0.01 with a warning, like every other value outside its range. Defaults and
  ranges for every template column now come from the same place as the GUI's.
  The colocalization Cutoff in the GUI now stops at 10 µm, the template's
  limit; it had no upper limit.
- **Export checks the puncta files against the consensus mask.** The report
  re-derives each punctum's nucleus from the mask before counting, but the
  puncta CSVs beside it could still carry other IDs: in one lab folder, 98% of
  the Cy5 puncta's `Nucleus_ID` values disagreed with the saved mask while the
  check before export reported no change. Export now reads each aligned
  puncta CSV back, saves it again from its layer if it lags behind, and stops,
  naming the file, if it still disagrees.
- **`zfisher-cli nucleus-ids <output folder>`** checks every aligned puncta
  CSV's `Nucleus_ID` against the folder's saved consensus mask and reports the
  agreement per file; with `--write` it writes a repaired copy of each,
  `*.nucleus_ids_repaired.csv`, beside the original, with every ID read from
  the mask at the punctum's nearest voxel. Originals are never changed.
- **A puncta file is never written with its values on the wrong puncta.** A
  puncta CSV joins each layer's table of values to its points row by row; a
  table one row short or long put every following value, `Nucleus_ID`
  included, on the wrong punctum, without a word. Such a table is now refused
  and the file keeps its last good version, with the reason in the log and the
  status bar. A save that failed for any reason used to write the coordinates
  alone, dropping every punctum's `Nucleus_ID`, `puncta_id` and `Source`; it
  now writes nothing. A `puncta_id` carried by more than one punctum, which
  sessions from before 2.1 can hold, is logged.
- **The report states the sigma detection used.** A batch channel with no
  row on the Puncta sheet, and every channel of `zfisher-cli run`, is detected
  without a sigma, and the detectors then smooth with sigma 1.0. The Parameters
  sheet said 0.0 and `provenance.jsonl` left it out. Detection now fills in every setting
  it is not given with the value it uses, and records those; the puncta found
  are unchanged. Reports made earlier from such channels list sigma 0.0 where
  1.0 was used. Channels set on the Puncta sheet, and detection in the GUI,
  were always reported correctly.
- **Detect Puncta on a raw image reads the files, as batch does.** It
  detected on the image and mask arrays in the window, so the provenance
  record named files the detection had not read, and GUI and batch could give
  different puncta for the same settings. It now reads the round's image file
  and the round's own nuclei mask file, through the same function as batch and
  re-detection, after saving any puncta or mask edits still waiting to be
  written. If the mask chosen in the widget is not the round's own, the status
  bar says which mask was used. Detecting on an aligned image is unchanged.
- **A round's nuclei mask edited in the GUI still counts as that round's
  mask.** Saving an edit registered it under a different kind, so re-detecting
  a channel on its original image would then have stopped with "no nuclei
  mask".
- **The session file is written whole or not at all.** It was overwritten in
  place, so a crash, a full disk or a killed process during a save left a
  truncated session file, which then would not open. It is now written beside
  the old one and swapped in, so the previous version stays until the new one
  is complete. A save that fails is no longer only logged: an action started
  from a button shows the error, and a save made in the background, after a
  puncta or mask edit, says so in the status bar. The change stays in memory
  and is written by the next save.

### Documentation

- The README names the Cellpose model zFISHer runs, Cellpose 4's default
  `cpsam`; it said `nuclei`.
- The example dataset's DOI in `examples/README.md` reads 10.5281/zenodo.20288536,
  the record its link opens; the number shown before is not a registered DOI.

### Diagnostics

- **`zfisher-cli profile <folders or log files>`** reads session logs and
  writes a one-page profile: the machines, minutes per dataset for batch runs,
  seconds per pipeline stage and per GUI action (median, 90th percentile,
  worst, total), and the longest main-thread blocks with what was running.
  Blocks while zFISHer was in the background are counted apart, since the
  system can slow an idle app's timers. `-o profile.md` writes it to a file.
  It only reads.

### Distances and report tables

- **How many puncta have a neighbour within a distance, without a rerun.**
  The Colocalization & Export page has a Distance Query section: choose a
  source and a target channel, a distance and a bin width, and Count says how
  many source puncta have a target punctum within that distance, with a
  histogram, read from the latest report. `zfisher-cli query-distances` gives
  the same answer. The count is per source punctum, of its nearest target
  punctum.
- **The Distribution sheet's bins can be set**, in the Distribution Bins field
  of the Export section and in a new, optional Distribution_Bins column of the
  batch template, as the lower edge of each bin: `0, 1, 2, 3, 11`, the default,
  gives the columns the sheet always had.
- **`zfisher-cli metrics <folder>`** gives a session's quality-control numbers,
  read from its files: how the rounds registered (method, RMSD in µm, pairs
  and inliers), per channel the puncta count, SNR quantiles and the fractions
  at low SNR (below 2), outside nuclei and placed by hand, puncta per nucleus,
  how the channel was detected, the nuclei's count and volume quantiles in
  µm³, and with `--saturation` the share of saturated voxels in each aligned
  image. Registration now records its RMSD in µm as well; the RMSD it already
  recorded mixes Z and XY pixel sizes.
- **Every batch ends with a QC report**, `qc_report.md` in the batch's output
  folder, listing the datasets that need a look and why: a registration that
  fell back to the rough shift or whose centroid pairs mostly disagree, a
  channel with no puncta or detected on the resampled R2 image, a saturated
  image, and any metric far from the rest of the batch (3.5 robust standard
  deviations from the batch median, once there are at least 4 datasets).
  `zfisher-cli flag-batch <batch folder>` writes it for an existing batch;
  the thresholds can be set in a new QC sheet of the batch template.
- Each export also writes the report's tables as CSV files beside it:
  `reports/distances.csv` and one file per sheet under `reports/report_sheets/`,
  which scripts and the queries read instead of the workbook.

### Nuclei at the image edge

- **Nuclei cut by the edge of the image can be removed automatically**, with
  their puncta, instead of deleting them by hand: tick Remove Nuclei at the
  Image Edge in Match & Merge Nuclei or Automated Registration & Warping, set
  Remove_Border_Nuclei in the batch template, or pass
  `--remove-border-nuclei` to `zfisher-cli run`. The edge is that of the region
  both rounds imaged, taken from the registration, so a nucleus cut by either
  round's field of view goes. It is off by default; the decisions log records
  how many nuclei it removed. Turning it on for a finished batch folder
  re-runs the consensus mask and what follows from it, nothing earlier.

### Puncta at the ends of the stack

- **Puncta near the first and last Z plane can be dropped at detection**,
  where spots are cut by the ends of the stack: Exclude Z Edge Planes in
  Puncta Detection, or the new Z_Edge_Exclude column of the batch template's
  Puncta sheet (the lab's rule is 5). 0, the default, keeps them all. The
  puncta kept keep the ids detection gave them, and the setting is listed
  with the others in the report's Parameters sheet when it is used.

### Choosing the sensitivity

- **Suggest Sensitivity**, in Puncta Detection, counts the puncta a raw
  channel would give over a range of sensitivities (0.02 to 0.5) with the
  other settings in the widget, finds the widest stretch where the count
  levels off, shows it and its width, and sets the sensitivity to its middle.
  Press Detect Puncta to use it. `zfisher-cli sweep <folder> "R1 - FITC"`
  prints the same curve. The curve is kept in the session file. For Local
  Maxima the whole curve costs one detection.

### Per-nucleus export

- **`zfisher-cli export-cells <folder or batch root>`** writes one OME-TIFF per
  consensus nucleus into `cells/`: every aligned and warped channel cropped
  around the nucleus (its bounding box plus `--padding` µm, 2 by default) and,
  as the last channel, that nucleus's mask, with the voxel size in µm and the
  channel names in the OME header. Beside each, `nucleus_<id>_puncta.csv`
  lists its puncta from every aligned channel in the crop's voxel coordinates
  (`--no-puncta` to skip). Given a batch root, it exports every dataset
  folder in it; `--id` exports chosen nuclei only. The session is not changed.

### Resuming a batch

- **Running batch again over a folder it finished skips what is up to date.**
  Each pipeline stage (segmentation, detection per channel, registration, the
  canvas, the consensus mask, the warp per channel, the report) records the
  settings, input images and earlier stages it used, and the files it wrote.
  A second run over the same folder runs only the stages whose settings or
  inputs changed, and what depends on them, reading the rest from disk; a run
  that finished runs nothing. A file changed by hand since (a nucleus deleted
  from the consensus mask, say) is kept, and what was made from it is made
  again. Batch used to refuse an output folder that already had a session; it
  still refuses one whose session was not made by such a run (a GUI session,
  a batch from before this version), so hand-made work there is not touched.
- New sessions' ND2 images are converted to OME-TIFF once; batch converted
  them twice.
- **Several datasets at once.** Batch Process has a Workers setting, and
  `zfisher-cli batch` a `--workers N` option: that many datasets are processed
  at the same time, each in its own process. Each holds both of its rounds in
  memory, so the number is bounded by memory; the default stays 1. The window
  stays responsive while a batch runs, and the summary at the end says which
  datasets failed and why, where before every dataset that reached the end
  was reported as a success.
- `zfisher-cli status <folder>` lists each pipeline stage as fresh, stale,
  edited or missing, and `zfisher-cli update <folder>` runs again, with the
  settings the folder was made with, the stages that are not fresh.

### Decisions log

- **Each session keeps `decisions.jsonl`**, beside `provenance.jsonl`: one
  line per parameter choice or stage run, saying who made it (`human` for the
  window and the CLI, `default` for settings nobody chose, such as a batch
  channel without a Puncta row), which login ran it, what it applied to (round
  and channel), the old and new settings and what they produced (puncta and
  nuclei found, the registration's shift and error). Detection, segmentation,
  registration, the canvas, the consensus mask and colocalization rules are
  recorded, from the GUI, batch and scripts alike.

### Questions from an MCP client

- **`packages/zfisher-mcp`: an MCP server over finished sessions.** Any client
  that speaks the Model Context Protocol, a chat assistant or a script, can
  list the sessions under a folder and ask what a session holds, what is out of
  date and why, how a file was made, who set which parameter, its
  quality-control numbers, rows of its report, its run times, and how many
  puncta have a neighbour within a distance. The server reads only: sessions
  are opened read-only, nothing outside the folders it is given can be opened,
  and no answer names a folder above them. Install it beside zFISHer with
  `pip install -e packages/zfisher-mcp` and run `zfisher-mcp --root <folder>`;
  see its README. The core API gained the calls it uses: finding the sessions
  under a folder, a page of report rows filtered by column values, a session's
  provenance records and log timing lines, and one channel's detection
  settings.
- **The MCP server can triage a batch and tune settings, when started to.**
  `zfisher-mcp --mode triage` adds flagging the datasets of a batch that need a
  look and proposing changes for a person to accept; `--mode operate` adds
  changing a pipeline setting (refused outside its range), rerunning what is
  out of date, re-detecting, warping and exporting a channel, the threshold
  sweep, per-nucleus export and running a batch template. Every one of these
  is a dry run unless the client says otherwise, and says what it would write,
  what it would leave out of date and which stages would run. Every change is
  logged in the session's decisions log with its justification, the server's
  actor and the client's name.
- **Home > Assistant Server** starts the MCP server from inside zFISHer, for a
  client on the same computer: it serves the folder holding the open session
  at `http://127.0.0.1:<port>/mcp`, behind a token made fresh for each start,
  in the mode chosen there. The open session is served from zFISHer's memory;
  a change a client makes to it runs in zFISHer after pending edits are saved,
  under a progress dialog, and the puncta layers it rewrote are reloaded on
  screen. Changes that would rewrite the open session (update, running a
  batch) are refused for it. It needs `packages/zfisher-mcp` installed.
  `zfisher-mcp --http [HOST:]PORT` serves the same way without the GUI.
- **What a settings change would rerun can be known before running it.** The
  core API can say which pipeline stages an update would run with given
  settings, without running any, from the same stage table the run uses; it
  can change one pipeline setting after checking it against the setting's
  range, and say what re-detecting, warping or exporting would write and
  leave out of date. The decisions log records proposals, and the client a
  tool server's call came from.
- **Exporting a batch session's report uses the batch's colocalization
  rules** when none are given; they were only taken from where the GUI keeps
  them, so a report exported again from a batch session had no colocalization
  sheets.

### Installation

- **The `zfisher` command works after `pip install`.** It pointed at a
  `main.py` that is not part of the installed package, so outside the source
  folder it failed with "No module named 'main'". It now starts the app from
  the package; `python -m zfisher` does the same, and `python main.py` still
  works in a source checkout.
- **`zfisher-cli`** runs the headless pipeline (`run`, `batch`, `template`),
  as `python -m zfisher.cli` did, and has two new commands: `status <folder>`
  describes a session and says what in it is out of date, reading only
  (`--json` for a machine-readable answer), and `export <folder>` writes its
  colocalization report from the files on disk, with the session's rules or
  `--rule "R1 - FITC,R2 - Cy5,1.0"`. Every command goes through the same core
  operations a script or another front end would use. Its `-v` now shows zFISHer's debug messages;
  it set the level on a logger zFISHer's messages do not pass through.

### Development

- Cellpose segmentation no longer passes the `channels` argument Cellpose 4
  deprecated; its default model ignores it, and the masks are the same with
  and without it.
- Errors zFISHer raises on purpose derive from `zfisher.core.errors`
  (`SessionError`, `InputError`, `SchemaError`, `RegistrationError`,
  `DetectionError`, `StaleError`) and keep the built-in type they had, so
  existing `except` clauses still catch them. The pipeline raises them instead
  of bare `RuntimeError`. The GUI's error handler shows them by their message,
  logs a traceback only for unexpected errors, returns `FAILED`, and re-raises
  when `ZFISHER_RAISE_ERRORS=1` is set, for tests. No handler in `zfisher.core`
  passes over an error silently any more.
- `zfisher/core/api.py` is the one set of operations every front end calls:
  open or create a session (read-only by default, writing nothing), describe
  it, report what is out of date, list its settings, query its report, check
  its nucleus IDs, detect and warp a channel, export the report, and run the
  pipeline. Results are typed models with a JSON schema. The report export
  works from the files on disk, reading each punctum's nucleus from the
  consensus mask, and a rule that names a channel both rounds have is refused
  instead of taking the first match.
- A session is now a `Session` object (`zfisher/core/session.py`) that owns
  its settings, file registry, lock and JSON file; the module-level functions
  act on one default instance, so nothing that called them changes, and the
  files written are byte for byte the same. Several sessions can live in one
  process, and `Session.batch_writes()` writes the file once for a run of
  changes instead of after each one. `session.use(s)` makes `s` the session
  the module functions act on for a block, per thread; the pipeline runs that
  way, so no core function needed a new argument. A pipeline run writes its
  session file at the end of each stage instead of after every file it
  registers: 15 writes instead of 44 for the synthetic test dataset.
- Every setting a person or a batch template chooses (detection, segmentation,
  registration, consensus, colocalization) is described once, in
  `zfisher/core/params.py`, with its default, range, choices, label and
  tooltip. Tests hold each GUI control to its field, and the models' JSON
  schema to a snapshot in `tests/snapshots/`. `pydantic` is now a declared
  dependency. The widgets for detection, registration, nuclei matching,
  automated preprocessing and colocalization build their controls from these
  models, and the duplicated detection defaults are gone from `constants.py`.
- Session files record their format, `schema_version: 1`. A session written by
  a later zFISHer in a format this one cannot read is refused with a message
  saying so, instead of being half read.
- The tests find the checkout and each other through pytest's `pythonpath`
  setting instead of editing `sys.path` in every file.
- `ruff check .` runs in CI for the errors pyflakes finds (unused and
  undefined names, imports shadowing each other); 112 such findings are fixed.
  Install with `pip install -e ".[test]"` to get pytest and ruff.
- A test now covers opening a listed file from its `.bak` when the file itself
  is missing; a docstring claimed one existed.

## 2.2.1 — 2026-10-29

### Fixes

- **Export refuses to build a report on nucleus IDs it could not re-derive.**
  The per-nucleus numbers come from each puncta layer's `Nucleus_ID`, which is
  re-derived from the consensus mask just before export. If no consensus mask is
  loaded, or a puncta layer has no `Nucleus_ID` column or a table that does not
  match its points, the check used to pass silently (`changed=0`) and the report
  could carry stale assignments. Export now stops and names the problem, and the
  log warns for each skipped layer.
- **Reset and Load Session leave the open session's files alone.** Clearing
  the canvas went through the same path as deleting a layer by hand, which
  backs the layer's file up to `.bak` and drops it from the session file. So
  Home > Reset renamed every file the open session lists (masks, puncta CSVs,
  aligned images, reports) and emptied its session file; Load Session did the
  same to the session it was leaving; and loading the open session again, from
  the session file the app was writing, found it already emptied and came back
  without its processed layers. The files and the session file now stay as
  they are, and New Session clears the canvas the same way. A session this
  happened to still has its files, renamed with a `.bak` ending, where they
  were; where an earlier session file exists in its `sessions/` folder,
  loading that one brings them back, since a listed file that is missing is
  read from its `.bak`.
- **Load Session keeps nothing of the session it replaces.** The file being
  loaded was merged into the session in memory, so a key the file did not
  have kept the value of whatever session was open before it: a session made
  before 2.1.0, opened after a newer one, inherited that one's saved
  registration, canvas scale and nuclear channel, and re-detecting a channel
  would have warped its puncta with another dataset's transform. The session
  in memory is now replaced whole.
- **Batch Process saves the open session's mask edits first.** Mask edits are
  written a moment after the last stroke; starting a batch replaced the open
  session before that moment came, so the edit was written into the first
  batch dataset's folder, or not at all. The pending edits now go to the open
  session before the batch starts, as they already did before Load Session
  and New Session.

## 2.2.0 — 2026-10-29

### Re-detecting a channel of a reopened session

- **A channel can be re-detected on its original image and warped again**,
  without registering again, in sessions made with 2.1.0 or later. In Puncta
  Detection, choose the channel's aligned or warped image layer and keep
  **Detect on original image, then warp** ticked (the method batch uses): zFISHer
  reads that channel of the round's raw image (asking where it is if it has
  moved), detects with the settings in the widget inside the round's own nuclei
  mask, warps the puncta with the session's saved registration, and replaces
  the aligned puncta layer in place. Replace keeps the puncta placed by hand,
  on the raw or the aligned image; Replace All drops them; Merge adds the new
  puncta to the old ones.
- Unticked, detection runs on the aligned image itself, as before. For R2 that
  is the image after resampling, not the method batch uses, so zFISHer asks
  first.
- The export section says when the report is out of date and why (a channel
  was re-detected, puncta were changed, the consensus mask was edited); export
  again to update it. The status bar says the same after a re-detection.
- **Puncta can be picked by hand on a channel's original image and warped.**
  Puncta Picking > Manual > **Pick on Original Image**: choose a channel and
  press **Open Original Image** to load that channel's raw image and raw puncta
  over the aligned data (the channel's aligned layers are hidden meanwhile;
  **Show the round's nuclei mask** adds the round's own mask, about 1.2 GB more
  memory). Pick or correct puncta with the Puncta Editor, Fishing Hook
  included, then press **Warp Picks into Aligned Space**: the raw puncta, picks
  included, go through the session's saved registration onto the aligned
  puncta layer. **Close Without Warping** keeps the picks in the raw puncta and
  leaves the aligned puncta marked out of date. For R2 the original image is
  placed by the rigid shift only, so it sits close to, not exactly on, the
  warped data.
- **Sessions made before 2.1.0** have no saved warp, so neither feature above
  works in them. Puncta Detection says so under the greyed-out box, and so does
  the picking page: detecting on the aligned image still works, and the report
  says so; re-running the dataset through batch with zFISHer 2.1 or later makes
  both available. A folder missing its saved warp file
  (`aligned/registration_bspline.tfm`) is named as such.
- When puncta are warped into aligned space with puncta outside nuclei
  removed, as batch does, puncta placed by hand are kept wherever they are.
- **The report's Parameters sheet says how each channel was detected**, in a
  new **Detected on** column after Channel: on the original image and then
  shifted (R1) or warped (R2) into aligned space, the batch method; or on the
  aligned image, which for R2 is the image after resampling and differs from
  batch. It is read from the record of the channel's current puncta, and says
  "Not recorded" where there is none.

### Puncta IDs and detection settings

- **A `puncta_id` is never given to a second punctum.** Re-running detection
  with Replace used to number the new puncta from 0 (or from the highest kept
  manual punctum), so the ids of the puncta it removed were reused for
  different ones. Replace, Replace All, Merge and points added by hand now take
  ids higher than any the channel has had. The next free id is kept in the
  session file (`puncta_next_id`), so this holds after reopening a session too.
  Sessions without it start above the ids in that channel's puncta CSVs.
- **One channel has one row in the report's Parameters sheet.** GUI detection
  stored its settings under the image layer's name (`Warped R2 - FITC`) while
  batch used `R2 - FITC`, so re-detecting a batch channel added a second row and
  left the old one. Both now use `R2 - FITC`. In older sessions with several
  rows for one channel, the report shows the latest.
  Batch rows now also give `z_scale`, as GUI rows did.

### Provenance

- **Exported reports are recorded in `provenance.jsonl`**, with the puncta
  CSVs and the consensus mask they were made from. Aligned puncta CSVs also
  record the hash of the raw CSV they were warped from and a fingerprint of
  their puncta. From these zFISHer tells whether a channel's aligned puncta or
  the report are out of date, and whether aligned puncta were edited by hand
  since they were made. Updating a puncta's `Nucleus_ID` on reload or export
  does not count as an edit.

### Fixes

- **The Fishing Hook lands on the plane of the spot you click.** It searched
  the line through the cursor in steps of slightly more than a voxel, and in a
  2D view that skipped the plane under the cursor: clicking a spot on its own
  plane put the punctum on the plane above or below (0.3 to 0.5 µm off in Z),
  or on background for a spot one plane thick. It now searches every half
  voxel. Puncta placed with the Fishing Hook in 2D before this may sit a plane
  off.
- **The capture arrows and the scale bar no longer crash zFISHer on a value
  they cannot draw.** A corrupt arrow endpoint, or a camera zoom of nan or
  infinity, raised inside their drawing code; Qt's painter was left open, and
  the next redraw could end zFISHer with a segmentation fault and no
  traceback. With a corrupt arrow this happened as soon as the 3D view was
  drawn, which fits the 3D-rotation crash reports. Such values are now
  skipped, and the painter is always closed.
- Undo in the Puncta Editor swaps the points through the same routine as
  every other change to a puncta layer, which clears napari's drawing cache
  first; changing the number of points against a stale cache can corrupt what
  is drawn.
- Labels with "&" show it again: "Run Analysis & Export Master Report", the
  "Colocalization & Export" page and "Match & Merge Nuclei" showed a gap
  instead, because Qt reads a single "&" as a keyboard shortcut.
- **Replace and Merge judge duplicates in micrometres.** A new punctum within
  0.3 µm of one already there is the same punctum and is dropped, along Z as
  along XY. Before, the limit was 3 voxels, which at 0.3 µm Z steps meant
  0.9 µm in Z: puncta two planes apart were taken for one. Re-detection on the
  original image uses the same rule.
- Editing a raw-space puncta layer no longer drops its `space: raw` mark in the
  session, which made the next reopen draw raw puncta over the aligned images.
- **A copied output folder uses its own files.** Opening the session of a
  copied output folder while the original folder still existed kept every
  mask, puncta CSV, aligned image and report pointing into the original: the
  GUI showed the original's layers, and removing a layer renamed the
  original's file to `.bak`. Every file inside the output folder now opens from
  the folder the session file is in, and a file missing from the copy is
  reported missing instead of being read from the original. Raw images outside
  the output folder are opened where they are, as before.

### Diagnostics

- **A native crash now leaves a Python stack.** A segfault or access violation
  inside Qt or OpenGL used to end zFISHer with nothing in the session log. Now
  the Python stack of every thread is written to `crash_<time>.log`, in the
  session's `logs/` folder (before a session is open: `~/Library/Logs/zFISHer`
  on macOS, `%LOCALAPPDATA%\zFISHer\Logs` on Windows). A run that ends normally
  leaves no crash file.
- **Errors napari catches are logged.** zFISHer hides napari's error
  pop-ups, and with them every error raised inside a napari event handler.
  Those errors, with their tracebacks, now go to the session log
  ("Error caught by napari: ..."); the pop-ups stay hidden.

### Development

- `tools/check_mouse_picking.py` picks puncta with simulated mouse clicks and
  key presses (napari's add mode, the Fishing Hook) on an original image in
  zFISHer's real window, and warps them.
- A regression test for the 3D-rotation crash (`tests/test_3d_rotation.py`)
  rotates the 3D view of a session with puncta, ID labels and capture arrows in
  zFISHer's real window and draws every frame; it is skipped on machines
  without a display. `tools/app_driver.py` drives that window from a script,
  so GUI changes can be checked without someone at the screen.
- **Releases are tagged automatically.** When a merge into main brings a
  version made with `prepare_release.py` (one with no tag yet), the release
  workflow runs the tests, checks the version and changelog, tags the merge
  commit and creates the draft release. Releasing is now: run the prepare
  script, commit and push, merge the pull request, publish the draft. Merges
  that leave the version alone release nothing, and pushing a tag by hand still
  works.
- Batch detects each channel's puncta and transforms them into aligned space
  with the same core steps (`zfisher/core/stages.py`) that re-detecting a
  channel of a reopened session will use. Its puncta CSVs, images and masks
  are unchanged.

## 2.1.2 — 2026-10-14

### Fixes

- **Batch now works when R2's nuclear channel has another name than R1's**
  (the batch template's `R2_Nuclear_Channel`, e.g. `DAPI` in R1 and `Hoechst`
  in R2). Such a dataset failed with "Consensus failed: aligned masks not
  found": R2's mask was saved under R1's channel name, and canvas generation
  pairs the two rounds' channels by name, so R2's nuclei were never aligned or
  warped. The two nuclear channels and their masks are now paired with each
  other, and R2's files keep R2's channel name (`Warped_R2_Hoechst.tif`).
  Datasets whose rounds use the same name give the same results as before.
  When such a session is reopened in the GUI, R2's nuclear layers are not
  pre-selected in the widgets; choose them by hand.

## 2.1.1 — 2026-10-13

### Raw images

- **A raw image that has moved can be found again.** When Load Session cannot
  find a round's raw image, it asks where the file is now. The chosen file is
  checked against what the session recorded when it was created (shape, data
  type, channels, voxel size and file size); if anything differs, zFISHer lists
  the differences and asks before using it. The answer is saved in the session,
  so you are asked once, and every re-point is recorded in the session file
  (`raw_repoints`). Before, a round whose file had moved was silently skipped.
  Sessions made before this version record the image's identity the first time
  the image is found.
- **ND2 sessions reopen from their `input/` copy** instead of converting the ND2
  file again.
- **Each raw round is read once** when it is shown in the viewer. New Session and
  Load Session used to read both rounds a second time just to find the nuclear
  channel, a second full read of a multi-GB file. The nuclear channel chosen is
  unchanged.
- zFISHer can now read one channel of a raw image without reading the whole
  round, which re-detecting a single channel will need.

### Fixes

- **Classical nuclei segmentation no longer crashes when it finds no nuclei.**
  It failed with a numpy "shape mismatch" error. Nuclei Mapping now shows the
  rounds where nuclei were found and a "No Nuclei Found" warning naming the
  others; batch reports the dataset as failed with the same explanation and goes
  on to the next one. Classical segmentation cannot find a nucleus in a stack
  with fewer than 29 Z planes, and the warning says so when that is the cause.

### Development

- `python .github/scripts/prepare_release.py patch` (or `minor`, `major`, or a
  version number) prepares a release: it sets the version in
  `zfisher/version.py` and `CITATION.cff`, dates the Unreleased heading in this
  changelog, checks the result the way the release workflow will, and prints
  the git commands left to run.

## 2.1.0 — 2026-10-13

Faster mask editing, safer saves, and a full record of how every puncta file
was made. Sessions made with 2.0.x open and work unchanged; the upgrade notes
below list what is new in the output folder.

### Upgrade notes

- **New dependency:** `psutil` (memory figures in the session log). Existing
  environments need `pip install psutil==7.2.2`; a new environment created from
  `zfisher_environment.yml` gets it automatically.
- **Mask files are now `<name>_masks.ome.tif`**, tiled OME-TIFFs that carry their
  voxel size in µm. This applies to the R1/R2 nuclei masks from GUI and batch
  segmentation, the consensus mask, and masks saved by the mask editor. Older
  `.tif` masks still load. After the first save of an old session's mask, the
  session points to the new `.ome.tif`, and the old `.tif` stays on disk but is no
  longer updated. In Fiji, open the masks with the Bio-Formats importer. Masks in
  `aligned/` are unchanged.
- **New files in the output folder:**
  - `reports/R1_-_<channel>_puncta.csv` and `R2_-_…`: raw-space puncta, one file
    per round and channel. Batch used to discard these, and the GUI renamed them
    to `.bak`.
  - `aligned/registration_bspline.tfm`: the fitted B-spline warp.
  - `provenance.jsonl`: a record of every puncta file written (see below).
- **Puncta CSVs** written by the GUI now use the same column order as batch
  (`puncta_id, Z, Y, X, Nucleus_ID, Intensity, SNR, Source`, then any others), and
  `Source` is always present. Scripts that read these files by column name are
  unaffected; scripts that read columns by position need updating.
- **Re-running detection with Replace now keeps puncta placed by hand.** A new
  **Replace All** button, shown only when manual puncta exist, deletes them too.
- **The B-spline fit is seeded**, so the same images now give the same warp every
  time. Warps from earlier versions were random from run to run, so a new run
  differs slightly from any earlier one.
- **Max RANSAC Distance is in micrometres** and now takes effect; it was ignored
  before. The manual Registration widget's default search radius is therefore
  100 µm rather than 100 px.
- **Report export writes a numbered copy** instead of refusing to overwrite an
  existing report.

### Faster mask editing

Deleting one nucleus took about 2.5 s of main-thread work on a typical
71 × 2044 × 2048 mask; it now takes about 0.07 s.

- Delete and merge touch only the nucleus's bounding box, and so does their undo,
  which also removes a transient 1.2 GB copy of the mask per edit.
- Nucleus-ID labels are updated only for the nuclei an edit touched: 6 ms after a
  delete instead of 1.19 s.
- The mask redraws only the edited voxels. In 3D that is 2 ms instead of 114 ms,
  with 0.4 MB sent to the GPU instead of 297 MB.
- Saves compress only the tiles an edit touched: 52–64 ms instead of 0.22 s (0.73 s
  on one core). GUI segmentation no longer writes 1.2 GB uncompressed masks.

### Safer saves

- Pending mask saves now finish before New Session, Load Session, Reset and quit.
  Before, edits from the last few seconds could be lost at quit, or written into
  the next session's folder after a session switch.
- Edited masks are written compressed to a temporary file and then swapped into
  place, so a save that fails halfway no longer leaves a truncated mask.
- A failed mask save is now reported in the status bar and the session log, and
  retried on the next edit.
- Painted edits (brush, eraser, fill) are now saved. Before, only merge, delete
  and extrude reached disk.
- A failed aligned-image save now stops that dataset with the real error,
  instead of vanishing and surfacing only when the session was reopened.
- Mask undo stays on the layer its edits came from. Before, undoing after
  switching layers wrote one layer's edits into another and corrupted it.

### Puncta provenance

- `Source` (`auto`, `manual`, or `unknown` for files from before it was kept) now
  survives the canvas step, re-detection and every save. Before, the aligned-space
  transform reset every punctum to `auto`, and the GUI dropped the column and
  renumbered `puncta_id`.
- Raw-space puncta are kept, and each aligned punctum shares its `puncta_id` with
  the raw detection it came from.
- The registration is saved: the shift, the canvas offsets, the B-spline file, and
  whether RANSAC was used or it fell back to the rough shift (with the inlier
  fraction). Puncta can be transformed into aligned space again from disk,
  without registering again.
- Every puncta file zFISHer writes is recorded in `provenance.jsonl` and logged as
  an `ARTIFACT` line in the session log. Each record holds the file's SHA-256;
  round, channel and coordinate space; the image and mask used, each with its
  SHA-256; the detection parameters; the raw file and registration an aligned
  file came from; who ran it (batch or GUI); the zFISHer version; and the puncta
  counts by `Source`. Detection on an aligned or warped image is marked as such,
  since for R2 that is a different method from detecting on the raw image and
  warping.
- Reopening a session that has aligned data no longer draws raw-space puncta over
  the aligned images.

### Other fixes

- Refilter Puncta no longer gives surviving puncta another punctum's
  `puncta_id`, `Nucleus_ID` and `Source`.
- The per-nucleus report sheets now match the consensus mask after mask edits:
  Nucleus IDs are re-derived from the mask before export and on session load.
  The per-nucleus sheet lists every nucleus; Stats adds `Extranuclear_Puncta`
  and `Total_Puncta`; Distribution adds `Total_Nuclei`.
- Re-detection with Replace no longer fails when a kept manual punctum has no
  `puncta_id`.

### Diagnostics

- The session log records per-stage wall-clock and memory (`STAGE`), the latency
  of every manual action (`ACTION`), main-thread blocks over 100 ms (`STALL`), and
  the machine and output drive (`ENV`).
- The welcome screen and Home widget show the full version. Code that isn't a
  tagged release includes its commit (`2.1.0-dev+96a0ac9`), so it can be told
  apart from a release.
- Session files record the version that created them (`created_with_zfisher`)
  and the one that last saved them (`zfisher_version`). The report's Metadata
  sheet has a "zFISHer Version" row.

### Development

- The tests run automatically on GitHub for every push and pull request.
- Pushing a version tag runs the tests, checks the tag against
  `zfisher/version.py`, `CITATION.cff` and this changelog, and creates a draft
  GitHub release with this changelog's section as its notes.

### Known limitations

- R2's batch mask records R1's voxel size in its OME metadata. zFISHer reads scale
  from the session, so no measurement is affected.
- The version string does not show uncommitted local changes.
- Re-detecting a channel and rewarping it from the GUI is not available yet; this
  release saves everything that feature needs.

## 2.0.1

Untagged. Fixed the scale bar label at 0.1× zoom.

## 1.0.0 — 2026-06-05

First tagged release.
