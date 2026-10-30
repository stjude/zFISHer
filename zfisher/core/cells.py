"""One OME-TIFF per nucleus: the aligned channels cropped around it.

``export_cells`` crops every aligned and warped channel of a session, and the
consensus mask, around each nucleus (its bounding box plus a margin given in
micrometres), and writes ``cells/nucleus_<id>.ome.tif``: channels Z, C, Y, X,
the last channel being that nucleus's mask (1 inside, 0 outside), with the
physical voxel size and channel names in the OME metadata, so Fiji and other
OME readers open them at the right scale. With ``with_puncta`` a
``nucleus_<id>_puncta.csv`` beside it lists the nucleus's puncta from every
aligned channel, in the crop's voxel coordinates.

Each aligned channel is read once and cropped for every nucleus before the
next is read, so memory holds one full volume and the crops.
"""
import logging
import math
from pathlib import Path

import numpy as np

from .. import constants
from . import io, puncta, session

logger = logging.getLogger(__name__)

CELLS_DIR = "cells"


def aligned_channels():
    """``(name, path)`` of every aligned or warped channel image in the session, R1 first."""
    prefixes = (constants.ALIGNED_PREFIX + " ", constants.WARPED_PREFIX + " ")
    out = []
    for name, info in (session.get_data("processed_files", default={}) or {}).items():
        if (isinstance(info, dict) and info.get("type") == "image" and name.startswith(prefixes)
                and not name.endswith(constants.MASKS_SUFFIX) and info.get("path") and Path(info["path"]).exists()):
            out.append((name, Path(info["path"])))
    return sorted(out, key=lambda item: (puncta.channel_key(item[0]) or item[0]))


def _boxes(mask, ids, margin):
    """Each nucleus's bounding box grown by ``margin`` voxels per axis, clipped to the mask."""
    from scipy import ndimage
    objects = ndimage.find_objects(mask)
    boxes = {}
    for label in ids:
        if label < 1 or label > len(objects) or objects[label - 1] is None:
            continue
        boxes[label] = tuple(slice(max(0, s.start - m), min(n, s.stop + m))
                             for s, m, n in zip(objects[label - 1], margin, mask.shape))
    return boxes


def export_cells(ids=None, padding_um=2.0, with_puncta=True, out_dir=None, progress_callback=None):
    """Write one OME-TIFF per nucleus of the open session (all, or ``ids``);
    returns the paths written. See the module docstring."""
    import pandas as pd
    import tifffile
    from .stages import _consensus_path

    consensus_path = _consensus_path()
    if consensus_path is None:
        raise FileNotFoundError("The session has no consensus nuclei mask to cut the nuclei from.")
    scale = session.get_data("canvas_scale")
    if not scale:
        raise LookupError("The session has no canvas scale: it was not aligned.")
    channels = aligned_channels()
    if not channels:
        raise FileNotFoundError("The session has no aligned channel images to crop.")
    mask = io.read_label_tif(consensus_path)
    present = np.unique(mask)
    present = present[present > 0]
    wanted = present if ids is None else np.array([i for i in ids if i in set(present.tolist())])
    margin = [int(math.ceil(padding_um / s)) for s in scale]
    boxes = _boxes(mask, [int(i) for i in wanted], margin)
    folder = Path(out_dir) if out_dir else Path(session.get_data("output_dir")) / CELLS_DIR
    folder.mkdir(parents=True, exist_ok=True)

    crops = {label: [] for label in boxes}
    dtypes = []
    for i, (name, path) in enumerate(channels):
        if progress_callback:
            progress_callback(int(80 * i / len(channels)), f"Cropping {name}...")
        volume = tifffile.imread(path)
        if volume.shape != mask.shape:
            raise ValueError(f"{name} is {volume.shape}, the consensus mask {mask.shape}: they are not on the "
                             f"same canvas.")
        dtypes.append(volume.dtype)
        for label, box in boxes.items():
            crops[label].append(np.array(volume[box]))
        del volume
    same = len(set(dtypes)) == 1 and np.issubdtype(dtypes[0], np.integer)
    dtype = dtypes[0] if same else np.float32
    names = [name for name, _ in channels] + ["Nucleus mask"]

    tables = []
    if with_puncta:
        reports = Path(session.get_data("output_dir")) / constants.REPORTS_DIR
        layer_of = {constants.puncta_csv_path(reports, name).name: name
                    for name in (session.get_data("processed_files", default={}) or {})}
        for csv in puncta.aligned_puncta_csvs(reports):
            table = pd.read_csv(csv, float_precision="round_trip")
            if len(table):
                coords = table[["Z", "Y", "X"]].to_numpy(float)
                tables.append((layer_of.get(csv.name, csv.stem), table, puncta.nucleus_ids_from_mask(coords, mask)))

    written = []
    for n, (label, box) in enumerate(sorted(boxes.items())):
        stack = [c.astype(dtype, copy=False) for c in crops.pop(label)]
        stack.append((mask[box] == label).astype(dtype))
        data = np.stack(stack, axis=1)                          # Z, C, Y, X
        path = folder / f"nucleus_{label:05d}.ome.tif"
        tifffile.imwrite(path, data, photometric="minisblack", compression="zlib", metadata={
            "axes": "ZCYX", "PhysicalSizeZ": float(scale[0]), "PhysicalSizeY": float(scale[1]),
            "PhysicalSizeX": float(scale[2]), "PhysicalSizeZUnit": "µm", "PhysicalSizeYUnit": "µm",
            "PhysicalSizeXUnit": "µm", "Channel": {"Name": names}}, ome=True)
        written.append(path)
        if with_puncta:
            origin = np.array([s.start for s in box], dtype=float)
            rows = []
            for layer, table, nuclei in tables:
                inside = table[nuclei == label].copy()
                if len(inside):
                    inside[["Z", "Y", "X"]] = inside[["Z", "Y", "X"]].to_numpy(float) - origin
                    inside.insert(0, "layer", layer)
                    rows.append(inside)
            if rows:
                pd.concat(rows, ignore_index=True).to_csv(folder / f"nucleus_{label:05d}_puncta.csv", index=False)
        if progress_callback:
            progress_callback(80 + int(20 * (n + 1) / max(len(boxes), 1)), f"Wrote nucleus {label}")
    logger.info("Exported %d nuclei to %s (%d channels, margin %s voxels)", len(written), folder,
                len(names), margin)
    return written
