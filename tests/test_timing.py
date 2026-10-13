"""Headless tests for the session-log timing lines (``zfisher.core.timing``).

``STAGE`` lines time pipeline stages with memory, ``ACTION`` lines time manual
GUI operations, and one ``ENV`` line per session log records the machine and
storage. Timing must never change the outcome of the code it wraps.

``caplog`` cannot see these records because the ``zfisher`` logger does not
propagate, so the fixture attaches its own handler.

Run:  pytest tests/test_timing.py -q
"""
import io
import logging
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from zfisher.core import timing


@pytest.fixture
def timing_log():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    log = logging.getLogger("zfisher.core.timing")
    log.addHandler(handler)
    try:
        yield lambda: [timing.parse_timing_line(l) for l in stream.getvalue().splitlines()
                       if timing.parse_timing_line(l)]
    finally:
        log.removeHandler(handler)


def _only(records, kind):
    return [fields for k, fields in records if k == kind]


# --- STAGE ---------------------------------------------------------------------

def test_stage_line_parses_with_sanitised_extras(timing_log):
    with timing.stage_timer("puncta_detection", layer="R1 - FITC_puncta", note="a=b"):
        time.sleep(0.05)
    (line,) = _only(timing_log(), "STAGE")
    assert line["stage"] == "puncta_detection"
    assert float(line["duration_s"]) >= 0.05
    assert line["status"] == "ok"
    assert line["layer"] == "R1_-_FITC_puncta"
    assert line["note"] == "a:b"
    for key in ("rss_start_mb", "rss_end_mb", "peak_rss_mb", "proc_peak_mb"):
        float(line[key])  # numeric or nan, never missing


def test_parse_stage_line_reads_a_log_formatted_record():
    raw = ("2026-10-01 10:00:00 | zfisher.core.timing [INFO] STAGE stage=x duration_s=1.000 "
           "rss_start_mb=1.0 rss_end_mb=2.0 peak_rss_mb=3.0 proc_peak_mb=4.0 status=ok n_spots=7")
    fields = timing.parse_stage_line(raw)
    assert fields["stage"] == "x" and fields["n_spots"] == "7"
    assert timing.parse_timing_line(raw)[0] == "STAGE"
    assert timing.parse_stage_line("no timing here") is None


def test_stage_error_is_reraised_and_marked(timing_log):
    with pytest.raises(ValueError):
        with timing.stage_timer("boom"):
            raise ValueError("bad")
    (line,) = _only(timing_log(), "STAGE")
    assert line["status"] == "error" and line["error"] == "ValueError"


def test_timing_failures_never_break_the_stage(timing_log, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("timing is broken")
    monkeypatch.setattr(timing._SAMPLER, "register", broken)
    if timing.psutil is not None:
        monkeypatch.setattr(timing.psutil, "Process", broken)

    @timing.timed_stage("work")
    def work(x):
        return x * 2

    assert work(21) == 42
    (line,) = _only(timing_log(), "STAGE")
    assert line["status"] == "ok"
    if timing.psutil is not None:
        assert line["rss_start_mb"] == "nan"


def test_decorator_binds_positional_and_keyword_args_and_result(timing_log):
    @timing.timed_stage("detect",
                        fields=lambda a: {"layer": a["layer_name"], "method": a["params"]["method"]},
                        result_fields=timing.n_rows)
    def detect(image, params=None, layer_name=None):
        return np.zeros((3, 7))

    detect(None, {"method": "LoG"}, "R2 - Cy5")
    detect(None, params={"method": "DoG"}, layer_name="R1")
    first, second = _only(timing_log(), "STAGE")
    assert (first["layer"], first["method"], first["n_spots"]) == ("R2_-_Cy5", "LoG", "3")
    assert (second["layer"], second["method"]) == ("R1", "DoG")


def test_decorator_keeps_the_wrapped_signature():
    import inspect

    @timing.timed_stage("s")
    def f(a, b=2):
        return a + b

    assert list(inspect.signature(f).parameters) == ["a", "b"]
    assert f.__name__ == "f"


@pytest.mark.skipif(timing.psutil is None, reason="psutil not installed")
def test_sampler_sees_a_transient_allocation(timing_log):
    with timing.stage_timer("alloc"):
        block = np.ones(100 * 1024 * 1024, dtype=np.uint8)  # touched, so resident
        time.sleep(0.4)
        del block
    (line,) = _only(timing_log(), "STAGE")
    assert float(line["peak_rss_mb"]) - float(line["rss_start_mb"]) >= 50


def test_add_stage_fields_targets_innermost_stage_on_this_thread(timing_log):
    def worker():
        with timing.stage_timer("worker_stage"):
            timing.add_stage_fields(where="worker")

    with timing.stage_timer("outer"):
        with timing.stage_timer("inner"):
            timing.add_stage_fields(device="cpu")
            t = threading.Thread(target=worker)
            t.start()
            t.join()
        with timing.action_timer("an_action"):
            timing.add_stage_fields(after="inner")  # skips the action, lands on outer
    stages = {f["stage"]: f for f in _only(timing_log(), "STAGE")}
    assert stages["inner"]["device"] == "cpu"
    assert stages["worker_stage"]["where"] == "worker"
    assert "where" not in stages["inner"] and "where" not in stages["outer"]
    assert stages["outer"]["after"] == "inner" and "device" not in stages["outer"]


def test_pipeline_stage_lands_in_the_session_log(tmp_path):
    from zfisher.core import log_config, puncta, session
    session.clear_session()
    try:
        log_config.attach_session_log(tmp_path)
        res = puncta.process_puncta_detection(
            np.zeros((4, 16, 16), dtype=np.float32),
            params={"method": "Local Maxima", "threshold_rel": 0.5, "min_distance": 3},
            layer_name="R1 - FITC",
        )
    finally:
        session.clear_session()
    assert res.shape == (0, 7)
    text = "\n".join(p.read_text() for p in (tmp_path / "logs").glob("*.log"))
    records = [timing.parse_timing_line(l) for l in text.splitlines()]
    records = [r for r in records if r]
    (stage,) = [f for k, f in records if k == "STAGE" and f["stage"] == "puncta_detection"]
    assert stage["status"] == "ok" and stage["n_spots"] == "0" and stage["layer"] == "R1_-_FITC"
    (env,) = _only(records, "ENV")
    assert env["out_drive"] in ("local_fixed", "network", "removable", "ramdisk", "unknown")


# --- ACTION ---------------------------------------------------------------------

def test_action_line_parses(timing_log):
    with timing.action_timer("mask_delete", layer="Consensus Nuclei_masks", label=57):
        time.sleep(0.02)
    (line,) = _only(timing_log(), "ACTION")
    assert line["action"] == "mask_delete"
    assert float(line["duration_s"]) >= 0.02
    assert line["status"] == "ok" and line["thread"] == "main"
    assert line["layer"] == "Consensus_Nuclei_masks" and line["label"] == "57"
    assert "rss_start_mb" not in line


def test_action_error_is_reraised_and_marked(timing_log):
    @timing.timed_action("mask_merge")
    def merge():
        raise KeyError("nope")

    with pytest.raises(KeyError):
        merge()
    (line,) = _only(timing_log(), "ACTION")
    assert line["status"] == "error" and line["error"] == "KeyError"


def test_timed_action_returns_the_wrapped_value():
    @timing.timed_action("x")
    def f(a, *, b):
        return a, b

    assert f(1, b=2) == (1, 2)


def test_activity_slots_track_nesting_and_exceptions():
    assert timing.current_activity() is None
    with timing.action_timer("outer"):
        assert timing.current_activity() == "outer"
        with timing.stage_timer("inner"):
            assert timing.current_activity() == "inner"
        assert timing.current_activity() == "outer"
        assert timing.last_activity()[0] == "inner"
    assert timing.current_activity() is None
    assert timing.last_activity()[0] == "outer"

    with pytest.raises(RuntimeError):
        with timing.action_timer("fails"):
            raise RuntimeError
    assert timing.current_activity() is None
    assert timing.last_activity()[0] == "fails"


def test_worker_thread_actions_do_not_touch_main_slots():
    before = timing.last_activity()

    def worker():
        with timing.action_timer("in_worker"):
            pass

    t = threading.Thread(target=worker, name="detect-worker")
    t.start()
    t.join()
    assert timing.current_activity() is None
    assert timing.last_activity() == before


def test_user_wait_is_left_out_of_every_open_timer(timing_log):
    with timing.stage_timer("outer"):
        with timing.action_timer("mask_erase_all"):
            with timing.user_wait():  # e.g. the confirmation dialog
                time.sleep(0.2)
            time.sleep(0.02)
    records = timing_log()
    (action,) = _only(records, "ACTION")
    (stage,) = _only(records, "STAGE")
    for line in (action, stage):
        assert float(line["duration_s"]) < 0.15
        assert float(line["wait_s"]) >= 0.2
    with timing.action_timer("no_dialog"):
        pass
    assert "wait_s" not in _only(timing_log(), "ACTION")[-1]


def test_hover_aggregate_is_silent_under_threshold(timing_log):
    agg = timing.HoverAggregate("mask_hover", window_s=0.0, threshold_ms=20)
    for _ in range(5):
        agg.add(0.001)
    agg.flush()
    assert _only(timing_log(), "ACTION") == []


def test_hover_aggregate_writes_one_line_per_slow_window(timing_log):
    agg = timing.HoverAggregate("mask_hover", window_s=0.05, threshold_ms=20)
    agg.add(0.030)
    agg.add(0.002)
    time.sleep(0.06)
    agg.add(0.001)  # closes the first window
    (line,) = _only(timing_log(), "ACTION")
    assert line["action"] == "mask_hover"
    assert line["n"] == "2" and float(line["max_ms"]) == pytest.approx(30.0)


# --- ENV ------------------------------------------------------------------------

def test_env_line_parses(tmp_path, timing_log):
    timing.log_environment(tmp_path)
    (env,) = _only(timing_log(), "ENV")
    for key in ("os", "cpu", "cores", "ram_gb", "gpu", "torch_cuda", "napari", "qt",
                "out_drive", "write_mb_s"):
        assert key in env, key
    assert float(env["write_mb_s"]) > 0
    assert not list(tmp_path.iterdir())  # the probe file is removed


def test_drive_type_unc_path_is_network():
    assert timing.drive_type(r"\\labserver\share\session") == "network"
    assert timing.drive_type("//labserver/share/session") == "network"


def test_drive_type_reports_a_remote_drive(tmp_path, monkeypatch):
    if sys.platform == "win32":
        monkeypatch.setattr(timing, "_windows_drive_type", lambda root: "network")
    else:
        monkeypatch.setattr(timing, "_posix_fstype", lambda path: "smbfs")
    assert timing.drive_type(tmp_path) == "network"


def test_drive_type_local_folder_is_not_network(tmp_path):
    assert timing.drive_type(tmp_path) in ("local_fixed", "unknown")


def test_write_probe_is_skipped_on_an_unwritable_folder(tmp_path, monkeypatch):
    def refuse(*_a, **_k):
        raise PermissionError("read-only")
    monkeypatch.setattr(timing.os, "open", refuse)
    assert timing.measure_write_speed(tmp_path) == "skipped"


@pytest.mark.skipif(sys.platform == "win32" or getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="needs POSIX permissions and a non-root user")
def test_write_probe_is_skipped_on_a_read_only_folder(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        assert timing.measure_write_speed(ro) == "skipped"
    finally:
        ro.chmod(0o755)
