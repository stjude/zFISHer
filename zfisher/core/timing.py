"""
Timing lines for the session log.

Three record types share one ``key=value`` grammar so a single parser reads
them all:

``STAGE``  one per pipeline stage (``stage_timer`` / ``timed_stage``):
           wall-clock, resident memory at start and end, the peak seen by a
           shared 0.1 s sampler, and the process high-water mark.
``ACTION`` one per manual GUI operation or deferred refresh
           (``action_timer`` / ``timed_action``): wall-clock only.
``ENV``    one per session log (``log_environment``): the machine and the
           storage the numbers came from.

``STALL`` lines come from ``zfisher.ui.stall_monitor``, which needs Qt; this
module stays free of Qt, napari and zfisher imports so core can use it.

Timing never breaks the work it measures: a failure inside a timer is logged
at DEBUG and swallowed, while an exception from the timed code propagates and
the line records ``status=error error=<Type>``.
"""

import functools
import inspect
import logging
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import psutil
except Exception:  # pragma: no cover - psutil ships with the environment
    psutil = None

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_S = 0.1
_MB = 1024 * 1024

# ---------------------------------------------------------------------------
# Line format
# ---------------------------------------------------------------------------

TIMING_LINE_RE = re.compile(r"\b(STAGE|ACTION|STALL|ENV)((?: [^\s=]+=\S*)+)\s*$")
STAGE_LINE_RE = re.compile(r"\bSTAGE((?: [^\s=]+=\S*)+)\s*$")


def _token(value):
    """One whitespace-free token for a key or value."""
    if isinstance(value, float):
        value = "nan" if math.isnan(value) else f"{value:g}"
    text = re.sub(r"\s+", "_", str(value).strip())
    return text.replace("=", ":") or "_"


def format_timing_line(kind, fixed, extras=None):
    """``KIND k=v k=v ...``: the ``fixed`` pairs in order, then the extras."""
    parts = [kind]
    for key, value in fixed:
        parts.append(f"{_token(key)}={_token(value)}")
    for key, value in (extras or {}).items():
        parts.append(f"{_token(key)}={_token(value)}")
    return " ".join(parts)


def format_stage_line(stage, duration_s, rss_start_mb, rss_end_mb, peak_rss_mb,
                      proc_peak_mb, status, extras=None):
    return format_timing_line("STAGE", [
        ("stage", stage),
        ("duration_s", f"{duration_s:.3f}"),
        ("rss_start_mb", f"{rss_start_mb:.1f}"),
        ("rss_end_mb", f"{rss_end_mb:.1f}"),
        ("peak_rss_mb", f"{peak_rss_mb:.1f}"),
        ("proc_peak_mb", f"{proc_peak_mb:.1f}"),
        ("status", status),
    ], extras)


def format_action_line(action, duration_s, status, thread, extras=None):
    return format_timing_line("ACTION", [
        ("action", action),
        ("duration_s", f"{duration_s:.3f}"),
        ("status", status),
        ("thread", thread),
    ], extras)


def _pairs(body):
    fields = {}
    for pair in body.split():
        key, _, value = pair.partition("=")
        fields[key] = value
    return fields


def parse_timing_line(line):
    """Return ``(kind, fields)`` for a timing line, or None. Values are strings."""
    match = TIMING_LINE_RE.search(line)
    if not match:
        return None
    return match.group(1), _pairs(match.group(2))


def parse_stage_line(line):
    """Return the fields of a ``STAGE`` line, or None."""
    match = STAGE_LINE_RE.search(line)
    return _pairs(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _rss_mb():
    try:
        return psutil.Process().memory_info().rss / _MB
    except Exception:
        return float("nan")


def _proc_peak_mb():
    """Process high-water mark: peak working set on Windows, max RSS elsewhere."""
    try:
        if sys.platform == "win32":
            return psutil.Process().memory_info().peak_wset / _MB
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Bytes on macOS, kilobytes on Linux.
        return peak / _MB if sys.platform == "darwin" else peak / 1024
    except Exception:
        return float("nan")


class _Sampler:
    """One daemon thread that samples RSS for every active stage."""

    def __init__(self, interval=SAMPLE_INTERVAL_S):
        self._interval = interval
        self._lock = threading.Lock()
        self._active = set()
        self._wake = threading.Event()
        self._thread = None

    def register(self, timer):
        with self._lock:
            self._active.add(timer)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="zfisher-rss-sampler", daemon=True)
                self._thread.start()
        self._wake.set()

    def unregister(self, timer):
        with self._lock:
            self._active.discard(timer)

    def _run(self):
        while True:
            with self._lock:
                active = list(self._active)
            if not active:
                self._wake.clear()
                with self._lock:
                    idle = not self._active
                if idle:
                    self._wake.wait()
                continue
            rss = _rss_mb()
            for timer in active:
                timer._observe(rss)
            time.sleep(self._interval)


_SAMPLER = _Sampler()

# ---------------------------------------------------------------------------
# Activity slots, read by the stall monitor
# ---------------------------------------------------------------------------

_local = threading.local()
_main_activity = []          # names of the stages and actions open on the main thread
_last_main_activity = None   # (name, perf_counter at exit) of the last one to close


def _on_main_thread():
    return threading.current_thread() is threading.main_thread()


def _thread_label():
    return "main" if _on_main_thread() else threading.current_thread().name


def _stack():
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def current_activity():
    """Innermost stage or action open on the main thread, or None."""
    return _main_activity[-1] if _main_activity else None


def last_activity():
    """``(name, ended_at)`` of the last main-thread stage or action, or None."""
    return _last_main_activity


def _push_activity(name):
    if _on_main_thread():
        _main_activity.append(name)


def _pop_activity(name):
    global _last_main_activity
    if not _on_main_thread():
        return
    for i in range(len(_main_activity) - 1, -1, -1):
        if _main_activity[i] == name:
            del _main_activity[i]
            break
    _last_main_activity = (name, time.perf_counter())


# ---------------------------------------------------------------------------
# Stage and action timers
# ---------------------------------------------------------------------------

def _drop_from_stack(timer):
    stack = _stack()
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is timer:
            del stack[i]
            break


class stage_timer:
    """Context manager: one ``STAGE`` line with wall-clock and memory."""

    kind = "stage"

    def __init__(self, stage, **fields):
        self.name = stage
        self.fields = dict(fields)
        self._t0 = None
        self._rss_start = float("nan")
        self._peak = float("nan")
        self._waited = 0.0

    def add_fields(self, **fields):
        self.fields.update(fields)

    def _observe(self, rss):
        if not math.isnan(rss) and (math.isnan(self._peak) or rss > self._peak):
            self._peak = rss

    def _start(self):
        self._rss_start = _rss_mb()
        self._peak = self._rss_start
        _SAMPLER.register(self)

    def _finish(self, duration, status):
        _SAMPLER.unregister(self)
        rss_end = _rss_mb()
        self._observe(rss_end)
        logger.info(format_stage_line(
            self.name, duration, self._rss_start, rss_end, self._peak,
            _proc_peak_mb(), status, self.fields))

    def __enter__(self):
        self._t0 = time.perf_counter()
        try:
            _stack().append(self)
            _push_activity(self.name)
            self._start()
        except Exception:
            logger.debug("Timer start failed for %s", self.name, exc_info=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            duration = time.perf_counter() - self._t0 - self._waited
            if self._waited:
                self.fields["wait_s"] = f"{self._waited:.3f}"
            _drop_from_stack(self)
            _pop_activity(self.name)
            status = "ok"
            if exc_type is not None:
                status = "error"
                self.fields["error"] = exc_type.__name__
            self._finish(duration, status)
        except Exception:
            logger.debug("Timer stop failed for %s", self.name, exc_info=True)
        return False


class action_timer(stage_timer):
    """Context manager: one ``ACTION`` line, wall-clock only."""

    kind = "action"

    def __init__(self, action, **fields):
        super().__init__(action, **fields)
        self._thread = None

    def _observe(self, rss):
        pass

    def _start(self):
        self._thread = _thread_label()

    def _finish(self, duration, status):
        logger.info(format_action_line(self.name, duration, status, self._thread, self.fields))


class user_wait:
    """Context manager around a modal dialog.

    Time spent inside is left out of ``duration_s`` of every stage and action
    open on this thread and reported as their ``wait_s``, so a confirmation or
    a "done" popup does not count as compute time.
    """

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            waited = time.perf_counter() - self._t0
            for timer in _stack():
                timer._waited += waited
        except Exception:
            logger.debug("user_wait failed", exc_info=True)
        return False


def add_stage_fields(**fields):
    """Attach fields to the innermost stage open on the current thread."""
    try:
        for timer in reversed(_stack()):
            if timer.kind == "stage":
                timer.add_fields(**fields)
                return
    except Exception:
        logger.debug("add_stage_fields failed", exc_info=True)


def _timed(timer_cls, name, fields, result_fields):
    def decorator(func):
        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            extra = {}
            if fields is not None and signature is not None:
                try:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    extra = fields(dict(bound.arguments)) or {}
                except Exception:
                    logger.debug("Field extraction failed for %s", name, exc_info=True)
            with timer_cls(name, **extra) as timer:
                result = func(*args, **kwargs)
                if result_fields is not None:
                    try:
                        timer.add_fields(**(result_fields(result) or {}))
                    except Exception:
                        logger.debug("Result fields failed for %s", name, exc_info=True)
                return result
        return wrapper
    return decorator


def timed_stage(stage, fields=None, result_fields=None):
    """Decorator form of ``stage_timer``.

    ``fields`` gets the bound arguments as a dict, ``result_fields`` gets the
    return value; each returns a dict of extra fields for the line.
    """
    return _timed(stage_timer, stage, fields, result_fields)


def timed_action(action, fields=None, result_fields=None):
    """Decorator form of ``action_timer``."""
    return _timed(action_timer, action, fields, result_fields)


def n_rows(result):
    """``n_spots`` for a result that is an array (or None)."""
    return {"n_spots": 0 if result is None else len(result)}


class HoverAggregate:
    """Collapses a per-mouse-move handler into one ``ACTION`` line per window.

    The line is written only for windows whose slowest call exceeded
    ``threshold_ms``. A window is closed by the first call after it ends.
    """

    def __init__(self, action, window_s=5.0, threshold_ms=20.0):
        self.action = action
        self.window_s = window_s
        self.threshold_ms = threshold_ms
        self._reset(time.perf_counter())

    def _reset(self, now):
        self._start = now
        self._n = 0
        self._total = 0.0
        self._max = 0.0

    def add(self, duration_s):
        try:
            now = time.perf_counter()
            if now - self._start >= self.window_s:
                self.flush(now)
            self._n += 1
            self._total += duration_s
            self._max = max(self._max, duration_s)
        except Exception:
            logger.debug("Hover aggregate failed", exc_info=True)

    def flush(self, now=None):
        now = time.perf_counter() if now is None else now
        if self._n and self._max * 1000 > self.threshold_ms:
            logger.info(format_action_line(self.action, self._total, "ok", _thread_label(), {
                "n": self._n,
                "max_ms": f"{self._max * 1000:.1f}",
                "window_s": f"{self.window_s:g}",
            }))
        self._reset(now)


# ---------------------------------------------------------------------------
# Environment line
# ---------------------------------------------------------------------------

_NETWORK_FSTYPES = {
    "nfs", "nfs4", "smbfs", "cifs", "smb", "smb2", "smb3", "afpfs", "webdav",
    "sshfs", "fuse.sshfs", "ncpfs", "9p", "davfs", "fuse.davfs",
}
_WINDOWS_DRIVE_TYPES = {2: "removable", 3: "local_fixed", 4: "network", 5: "optical", 6: "ramdisk"}
WRITE_PROBE_BYTES = 8 * _MB


def _windows_drive_type(root):
    import ctypes
    return _WINDOWS_DRIVE_TYPES.get(ctypes.windll.kernel32.GetDriveTypeW(root), "unknown")


def _posix_fstype(path):
    """Filesystem type of the mount holding ``path`` (longest mountpoint match)."""
    best, fstype = "", None
    for part in psutil.disk_partitions(all=True):
        mount = part.mountpoint.rstrip("/") or "/"
        if (path == mount or path.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
            best, fstype = mount, part.fstype
    return fstype


def drive_type(path):
    """``local_fixed``, ``removable``, ``network``, ... or ``unknown``."""
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return "network"  # UNC path
    try:
        resolved = os.path.abspath(text)
        if sys.platform == "win32":
            drive = os.path.splitdrive(resolved)[0]
            if drive.startswith("\\\\"):
                return "network"
            return _windows_drive_type(drive + "\\")
        fstype = (_posix_fstype(os.path.realpath(resolved)) or "").lower()
        if not fstype:
            return "unknown"
        return "network" if fstype in _NETWORK_FSTYPES else "local_fixed"
    except Exception:
        logger.debug("Drive type lookup failed for %s", text, exc_info=True)
        return "unknown"


def measure_write_speed(folder, nbytes=WRITE_PROBE_BYTES):
    """MB/s for a synced write of ``nbytes`` to a temporary file in ``folder``.

    Returns the string ``"skipped"`` when the folder cannot be written.
    """
    probe = Path(folder) / ".zfisher_write_probe.tmp"
    try:
        payload = os.urandom(nbytes)
        t0 = time.perf_counter()
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0))
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        elapsed = time.perf_counter() - t0
        return f"{nbytes / _MB / max(elapsed, 1e-6):.0f}"
    except OSError:
        return "skipped"
    finally:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass


def _cpu_model():
    try:
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=2)
            return out.stdout.strip() or platform.processor()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_fields():
    """GPU name and memory from torch when it is loaded, else from nvidia-smi."""
    fields = {"gpu": "unknown", "vram_gb": "unknown", "torch_cuda": "not_loaded"}
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            fields["torch_cuda"] = str(bool(torch.cuda.is_available()))
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                fields["gpu"] = props.name
                fields["vram_gb"] = f"{props.total_memory / 1024**3:.0f}"
                return fields
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                fields["torch_mps"] = "True"
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            first = out.stdout.strip().splitlines()[0]
            name, mem_mib = [s.strip() for s in first.split(",")]
            fields["gpu"] = name
            fields["vram_gb"] = f"{float(mem_mib) / 1024:.0f}"
        except Exception:
            pass
    return fields


def _package_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def environment_fields(output_dir):
    fields = {
        "os": platform.platform(terse=True),
        "cpu": _cpu_model(),
        "cores": os.cpu_count() or "unknown",
        "ram_gb": "unknown",
    }
    try:
        fields["ram_gb"] = f"{psutil.virtual_memory().total / 1024**3:.0f}"
    except Exception:
        pass
    fields.update(_gpu_fields())
    fields["python"] = platform.python_version()
    fields["napari"] = _package_version("napari")
    fields["qt"] = f"PySide6_{_package_version('PySide6')}"
    fields["out_drive"] = drive_type(output_dir)
    fields["write_mb_s"] = measure_write_speed(output_dir)
    return fields


def log_environment(output_dir):
    """Write the ``ENV`` line for a session whose outputs go to ``output_dir``."""
    try:
        logger.info(format_timing_line("ENV", list(environment_fields(output_dir).items())))
    except Exception:
        logger.debug("Environment line failed", exc_info=True)
