# Changelog

All notable changes to zFISHer. Versions follow [semantic versioning](https://semver.org/).

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
