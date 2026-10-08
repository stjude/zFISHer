"""Label-volume edits limited to one label's bounding box.

Deleting or merging one nucleus used to scan, copy and diff the whole mask
(about 0.8 s on a typical 71 x 2044 x 2048 volume). Restricted to the label's
bounding box the same work takes milliseconds. :class:`LabelBoxes` holds the
boxes for one volume, built once with ``find_objects`` and then kept current by
the edits themselves.

A box only ever grows. An edit that removes voxels leaves the boxes as they
were, so each box stays a superset of its label's voxels whatever later puts
old values back (the editor's undo, napari's own undo and redo). The cost is
boxes that can be larger than needed, which only makes an edit slower, never
wrong. Label 0 is background and has no box.
"""
import logging

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def index_extent(indices):
    """Bounding box (tuple of slices) of a fancy-index tuple, or None if empty."""
    if not indices or np.size(indices[0]) == 0:
        return None
    return tuple(slice(int(ix.min()), int(ix.max()) + 1) for ix in indices)


class LabelBoxes:
    """Per-label bounding boxes of one label volume; boxes only grow."""

    def __init__(self, data):
        self.shape = data.shape
        self._lo = {}
        self._hi = {}
        for i, region in enumerate(ndimage.find_objects(data)):
            if region is not None:
                self._lo[i + 1] = np.array([s.start for s in region])
                self._hi[i + 1] = np.array([s.stop for s in region])

    def get(self, label):
        """The box of ``label`` as a tuple of slices, or None if it has none."""
        label = int(label)
        if label not in self._lo:
            return None
        return tuple(slice(int(a), int(b)) for a, b in zip(self._lo[label], self._hi[label]))

    def extend(self, label, region):
        """Grow the box of ``label`` to cover ``region`` (a tuple of slices)."""
        label = int(label)
        if label <= 0 or region is None:
            return
        lo = np.array([s.start for s in region])
        hi = np.array([s.stop for s in region])
        if label in self._lo:
            np.minimum(self._lo[label], lo, out=self._lo[label])
            np.maximum(self._hi[label], hi, out=self._hi[label])
        else:
            self._lo[label] = lo
            self._hi[label] = hi

    def extend_values(self, indices, values):
        """Grow the boxes of every label in ``values`` to cover ``indices``.

        ``values`` is one label or an array matching ``indices``, as in a
        napari history atom. Each label gets the extent of all the indices,
        which is a superset of its own.
        """
        region = index_extent(indices)
        if region is None:
            return
        for label in np.unique(np.asarray(values)):
            self.extend(label, region)


def _search_region(data, boxes, label):
    region = boxes.get(label)
    if region is None:
        return tuple(slice(0, n) for n in data.shape), False
    return region, True


def count_label(data, boxes, label):
    """Number of voxels of ``label``, counted inside its box only."""
    region, _ = _search_region(data, boxes, label)
    return int(np.count_nonzero(data[region] == label))


def replace_label(data, boxes, old, new):
    """Set every voxel of ``old`` to ``new`` in place, touching only old's box.

    Returns the changed indices as a fancy-index tuple into ``data``, or None
    if ``old`` has no voxels. Every changed voxel held ``old``, so the undo
    record is ``(indices, old)``. A label without a box is searched in the
    whole volume, which costs what the edit cost before boxes.
    """
    region, boxed = _search_region(data, boxes, old)
    sub = data[region]
    hit = sub == old
    local = np.nonzero(hit)
    if local[0].size == 0:
        return None
    if not boxed:
        logger.warning("Label %d had no bounding box but has %d voxels; searched the "
                       "whole volume", int(old), local[0].size)
    sub[hit] = new
    indices = tuple(ix + s.start for ix, s in zip(local, region))
    boxes.extend(new, index_extent(indices))
    return indices
