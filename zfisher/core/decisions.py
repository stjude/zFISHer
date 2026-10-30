"""The decisions log: who set which parameter, from what to what, and why.

``<output_dir>/decisions.jsonl`` sits beside ``provenance.jsonl``. Provenance
says how each file was made; this log says who chose the settings: one record
per parameter change or stage run, with the time, the **actor** (``human``,
``default`` for a value nobody chose, or ``agent:<model>``), the **user** (the
OS login, or the person a shared service authenticated), the scope (round,
channel, rule), the setting's name, its old and new values, and a
justification. A tool server also stamps **via**, the client its calls came
from (``mcp:<client name>``). A record with ``status: "proposed"`` is a
change suggested for a person to accept or dismiss, not one made.

The actor and the user are stamped by the front end that runs the code, never
taken from an argument a caller (a model, say) could set: the GUI marks its
process ``human``, the CLI marks its commands ``human``, the pipeline marks a
channel left to the built-in settings ``default``, and a tool server marks its
calls with the agent it serves (``use_actor``).
"""
import contextvars
import datetime
import getpass
import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path

from . import session

logger = logging.getLogger(__name__)

DECISIONS_FILE = "decisions.jsonl"
HUMAN, DEFAULT = "human", "default"

_actor = contextvars.ContextVar("zfisher_actor", default=None)
_user = contextvars.ContextVar("zfisher_user", default=None)
_via = contextvars.ContextVar("zfisher_via", default=None)
_process_actor = [DEFAULT]
_lock = threading.Lock()


def set_process_actor(actor):
    """The actor for everything this process runs outside ``use_actor``: the
    GUI sets ``human`` at start-up."""
    _process_actor[0] = actor


def current_actor():
    return _actor.get() or _process_actor[0]


def current_user():
    if _user.get():
        return _user.get()
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


@contextmanager
def use_actor(actor, user=None, via=None):
    """Run a block with ``actor`` (and ``user``, when a service authenticated
    one, and ``via``, the client a tool server's call came from) stamped on
    its decisions."""
    actor_token = _actor.set(actor)
    user_token = _user.set(user) if user is not None else None
    via_token = _via.set(via) if via is not None else None
    try:
        yield
    finally:
        _actor.reset(actor_token)
        if user_token is not None:
            _user.reset(user_token)
        if via_token is not None:
            _via.reset(via_token)


def decisions_file():
    """``<output_dir>/decisions.jsonl``, or None without a session folder."""
    out = session.get_data("output_dir")
    return Path(out) / DECISIONS_FILE if out else None


def record(name, new, old=None, scope=None, justification=None, metrics_after=None, status=None,
           even_read_only=False):
    """Append a decision for setting ``name`` (``new`` replacing ``old``) to
    the session's log. ``scope`` says what it applies to (``{"round": "R1",
    "channel": "FITC"}``); ``status`` is ``"proposed"`` for a suggestion.
    Returns the record, or None without a session folder or in a read-only
    session, unless ``even_read_only`` (a proposal adds to the log without
    changing the session)."""
    target = decisions_file()
    if target is None or (session.current().read_only and not even_read_only):
        return None
    from .provenance import software_version
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "actor": current_actor(),
             "user": current_user(), "scope": scope or {}, "name": name, "old": old, "new": new,
             "justification": justification, "metrics_after": metrics_after,
             "software": {"zfisher": software_version()}}
    if _via.get():
        entry["via"] = _via.get()
    if status:
        entry["status"] = status
    if status == "proposed":
        import uuid
        entry["id"] = uuid.uuid4().hex[:12]           # what a review of it names
    with _lock:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
    logger.info("DECISION %s by %s (%s): %s", name, entry["actor"], entry["user"],
                ", ".join(f"{k}={v}" for k, v in entry["scope"].items()) or "session")
    return entry


def records(name=None):
    """Every decision of the session, oldest first; only those for ``name`` if given."""
    target = decisions_file()
    if target is None or not target.exists():
        return []
    out = []
    with open(target, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unreadable line %d of %s", n, target.name)
                continue
            if name is None or entry.get("name") == name:
                out.append(entry)
    return out
