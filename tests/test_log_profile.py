"""The one-page profile read from session logs (scope plan M1, gate G0).

``zfisher-cli profile`` reads the STAGE, ACTION, STALL and ENV lines of
session logs and writes minutes per dataset, seconds per stage and per GUI
action, and the worst main-thread blocks. Blocks while zFISHer was in the
background are counted apart.

Run:  python -m pytest tests/test_log_profile.py -q
"""
from zfisher import cli
from zfisher.core import log_profile, timing

T = "2026-10-16 12:00:{:02d} | zfisher.core.timing [INFO] "


def _write(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(T.format(i) + line for i, line in enumerate(lines)) + "\n")
    return path


def _stage(name, seconds, peak=1024.0, status="ok"):
    return timing.format_stage_line(name, seconds, 100.0, 200.0, peak, peak, status)


def _logs(tmp_path):
    batch = _write(tmp_path / "ds1" / "logs" / "zfisher_2026-10-16_120000.log", [
        "Output directory: /data/ds1",
        "ENV os=Windows-10 cpu=Xeon cores=24 ram_gb=62 gpu=RTX_4070 vram_gb=16 torch_cuda=False out_drive=network",
        _stage("segment_nuclei:cellpose", 300.0, peak=8192.0),
        _stage("puncta_detection", 20.0),
        _stage("puncta_detection", 40.0, status="error"),
        _stage("pipeline_total", 600.0, peak=8192.0),
    ])
    gui = _write(tmp_path / "ds2" / "logs" / "zfisher_2026-10-16_130000.log", [
        "Output directory: /data/ds2",
        timing.format_action_line("widget:puncta_detection", 12.0, "ok", "main"),
        timing.format_action_line("puncta_save", 0.02, "ok", "main"),
        timing.format_timing_line("STALL", [("blocked_ms", 5000), ("during", "widget:automated_registration"),
                                            ("app_active", 1)]),
        timing.format_timing_line("STALL", [("blocked_ms", 900000), ("during", "untracked"), ("app_active", 0)]),
    ])
    return batch, gui


def test_logs_are_read_and_told_apart(tmp_path):
    batch, gui = _logs(tmp_path)
    runs = {r.path: r for r in log_profile.profile([tmp_path])}
    assert runs[batch].kind == "batch" and runs[gui].kind == "gui"
    assert runs[batch].output_dir == "/data/ds1" and runs[batch].env["cores"] == "24"
    assert [s[0] for s in runs[batch].stages][-1] == "pipeline_total"
    assert runs[gui].stalls[1][3] is False                       # in the background


def test_the_profile_has_minutes_per_dataset_and_leaves_background_blocks_out(tmp_path):
    _logs(tmp_path)
    text = log_profile.format_markdown(log_profile.profile([tmp_path]))
    assert "| ds1 | 10.0 | 8.0 |" in text                         # pipeline_total minutes, peak GB
    assert "| puncta_detection | 2 | 30.0 |" in text and "| 1 |" in text.split("| puncta_detection |")[1].split("\n")[0]
    assert "| widget:automated_registration | 1 | 5.0 | 5.0 |" in text
    assert "1 blocks over 100 ms while zFISHer was the active app" in text and "Another 1 (900 s)" in text
    assert "1 runs without the GUI, 1 GUI sessions" in text


def test_the_cli_writes_the_profile_and_fails_without_logs(tmp_path, capsys):
    _logs(tmp_path)
    out = tmp_path / "profile.md"
    assert cli.main(["profile", str(tmp_path), "-o", str(out)]) == 0
    assert out.read_text().startswith("# zFISHer profile")
    assert cli.main(["profile", str(tmp_path / "empty")]) == 1
    assert "No session logs" in capsys.readouterr().out
