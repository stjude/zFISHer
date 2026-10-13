"""Provenance of the files a session writes.

Each time zFISHer writes an artifact it appends one record to
``<output_dir>/provenance.jsonl`` and logs the same event as an ``ARTIFACT``
line in the session log. A record says what the file is and how it came to be:
its SHA-256, round, channel, coordinate space, the inputs it was made from (each
with its own SHA-256), parameters, what it was derived from, who ran it (batch
or gui), the software version and the time.

The file is append-only. A new record for a file carries forward the fields of
that file's previous record, so the latest record for a path is its complete
description and the earlier ones are its history. Paths inside the output
folder are stored relative to it, so the record survives the folder moving.

Files are identified by SHA-256, not by size and modification time, which
change when a folder is copied. ``verify`` reports a file changed outside
zFISHer: every zFISHer write of an artifact appends a record with its new hash.
"""
import datetime
import hashlib
import json
import logging
import threading
from pathlib import Path

from . import session

logger = logging.getLogger(__name__)

PROVENANCE_FILE = "provenance.jsonl"

_lock = threading.Lock()
_software_version = None


def software_version():
    """The zFISHer version with, for anything but a tagged release, the commit
    (``2.1.0-dev+96a0ac9``). Computed once per run."""
    global _software_version
    if _software_version is None:
        from ..version import get_full_version
        _software_version = get_full_version()
    return _software_version


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_sha256(path):
    """SHA-256 of a file. While a session is open the hash is kept in its
    ``file_hashes``, keyed by path and valid while size and modification time
    are unchanged, so a multi-GB raw image is hashed once, not at every save."""
    path = Path(path)
    stat = path.stat()
    if not session.get_data("output_dir"):
        return _sha256(path)
    key = str(path.resolve())
    cache = session.get_data("file_hashes", default={}) or {}
    hit = cache.get(key)
    if hit and hit.get("size") == stat.st_size and hit.get("mtime_ns") == stat.st_mtime_ns:
        return hit["sha256"]
    digest = _sha256(path)
    session.update_data("file_hashes", {**cache, key: {
        "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}})
    return digest


def file_identity(path):
    """Path, size, modification time and SHA-256 of an input file, to tell later
    whether it is still the same file. None for no path."""
    if not path:
        return None
    path = Path(path)
    try:
        stat = path.stat()
        digest = file_sha256(path)
    except OSError:
        return {"path": _stored(path), "missing": True}
    return {"path": _stored(path), "size_bytes": stat.st_size, "sha256": digest,
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")}


def _output_dir():
    out = session.get_data("output_dir")
    return Path(out) if out else None


def _stored(path):
    """``path`` relative to the output folder when inside it, else absolute."""
    path = Path(path)
    out = _output_dir()
    if out is not None:
        try:
            return path.resolve().relative_to(out.resolve()).as_posix()
        except ValueError:
            pass
    return str(path)


def provenance_file():
    """``<output_dir>/provenance.jsonl``, or None without an open session."""
    out = _output_dir()
    return out / PROVENANCE_FILE if out is not None else None


def records(path=None):
    """All records, oldest first; only those of ``path`` if given."""
    target = provenance_file()
    if target is None or not target.exists():
        return []
    key = _stored(path) if path is not None else None
    out = []
    with open(target, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                logger.warning("Skipping unreadable line %d of %s", n, target.name)
                continue
            if key is None or rec.get("path") == key:
                out.append(rec)
    return out


def latest(path):
    """The latest record of ``path``, or None."""
    found = records(path)
    return found[-1] if found else None


def record_artifact(path, event, **fields):
    """Append a record for the file at ``path``, which zFISHer just wrote, and
    log it as an ``ARTIFACT`` line. ``fields`` are merged over the file's
    previous record (None values are skipped). Returns the record, or None
    without an open session."""
    target = provenance_file()
    if target is None:
        return None
    path = Path(path)
    with _lock:
        previous = latest(path) or {}
        record = {k: v for k, v in previous.items()
                  if k not in ("event", "time", "sha256", "size_bytes", "software")}
        record.update({k: v for k, v in fields.items() if v is not None})
        record.update(path=_stored(path), event=event, sha256=_sha256(path),
                      size_bytes=path.stat().st_size, run_by=fields.get("run_by", record.get("run_by")),
                      software={"zfisher": software_version()},
                      time=datetime.datetime.now().isoformat(timespec="seconds"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, sort_keys=True) + "\n")
    summary = " ".join(f"{k}={record[k]}" for k in ("path", "event", "sha256", "round", "channel",
                                                    "space", "derived_from", "run_by")
                       if record.get(k) is not None)
    logger.info("ARTIFACT %s", summary.replace("\n", " "))
    return record


def has_record(path):
    return latest(path) is not None


def verify(path):
    """Whether the file at ``path`` is still the one its latest record
    describes: True if the hashes match, False if not (it was changed outside
    zFISHer, or is gone), None if it has no record."""
    rec = latest(path)
    if rec is None:
        return None
    path = Path(path)
    return path.exists() and _sha256(path) == rec.get("sha256")
