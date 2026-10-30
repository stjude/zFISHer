"""Closing a session log puts zFISHer's logger levels back.

Attaching a session log opens the zfisher loggers up to DEBUG so the file gets
everything. Detaching left them there, so after a session closed the console,
and whatever ran next, logged at DEBUG.

Run:  python -m pytest tests/test_session_log_levels.py -q
"""
import logging

from zfisher.core import log_config


def test_detaching_restores_the_levels_attaching_changed(tmp_path):
    names = ("zfisher", "stdout", "stderr")
    before = {n: logging.getLogger(n).level for n in names}
    log_config.attach_session_log(tmp_path)
    try:
        assert logging.getLogger("zfisher").level == logging.DEBUG
        log_config.attach_session_log(tmp_path)              # attaching again detaches first
        assert logging.getLogger("zfisher").level == logging.DEBUG
    finally:
        log_config.detach_session_log()
    assert {n: logging.getLogger(n).level for n in names} == before
