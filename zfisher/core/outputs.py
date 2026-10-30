"""Every file zFISHer writes goes through here, and is written whole or not at all.

``atomic_write(path)`` hands out a temporary path beside ``path`` (same folder,
same extension, so writers that go by extension still work), and only when
the block ends without an error swaps the finished file into place with
``os.replace``. A crash, a full disk or a killed process mid-write leaves the
previous file as it was, never a truncated one; a failed write leaves no
temporary file behind. On Windows, where another program (a virus scanner,
a sync client) can hold a file open for a moment, the swap is retried.

The helpers below (CSV, TIFF, NumPy, Excel, text) are the writers the rest of
zFISHer calls. Masks have their own tiled writer (``io.write_label_tif``),
which follows the same rule.
"""
import os
import time
from contextlib import contextmanager
from pathlib import Path

REPLACE_ATTEMPTS = 5
REPLACE_WAIT_S = 0.05


def replace(source, target):
    """``os.replace(source, target)``, retried while ``target`` is briefly locked."""
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_WAIT_S)


@contextmanager
def atomic_write(path):
    """Yield a temporary path to write ``path``'s new contents to; swap it in
    when the block succeeds, delete it when the block fails."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ends with the real name, so writers that decide by extension (tifffile's
    # OME-TIFF for ".ome.tif", pandas' Excel engine) treat it as the real file.
    tmp = path.with_name(f".{os.getpid()}.tmp.{path.name}")
    try:
        yield tmp
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text(path, text, fsync=False):
    """Write ``text`` (UTF-8). ``fsync`` forces it to disk before the swap."""
    with atomic_write(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
            if fsync:
                f.flush()
                os.fsync(f.fileno())
    return Path(path)


def write_csv(frame, path, **kwargs):
    """Write a pandas DataFrame as CSV (``index=False`` unless given)."""
    kwargs.setdefault("index", False)
    with atomic_write(path) as tmp:
        frame.to_csv(tmp, **kwargs)
    return Path(path)


def write_tiff(path, data, **kwargs):
    """Write an array with ``tifffile.imwrite`` (its keywords pass through)."""
    import tifffile
    with atomic_write(path) as tmp:
        tifffile.imwrite(tmp, data, **kwargs)
    return Path(path)


def write_npy(path, array):
    """Write an array as ``.npy``."""
    import numpy as np
    with atomic_write(path) as tmp:
        with open(tmp, "wb") as f:
            np.save(f, array)
    return Path(path)


@contextmanager
def excel_writer(path, **kwargs):
    """A ``pandas.ExcelWriter`` (openpyxl) on a temporary file, swapped into
    place when the block ends without an error."""
    import pandas as pd
    kwargs.setdefault("engine", "openpyxl")
    with atomic_write(path) as tmp:
        with pd.ExcelWriter(tmp, **kwargs) as writer:
            yield writer
