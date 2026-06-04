"""Headless tests for session-file location (new `sessions/` subfolder) with
backward-compatible loading of legacy root-level session files.

Run:  python -m pytest tests/test_session_location.py -q
or:   python tests/test_session_location.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zfisher import constants
from zfisher.core import session


def test_save_writes_into_sessions_subfolder():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            session.clear_session()
            session.update_data("output_dir", str(d))  # triggers a save
            assert (d / constants.SESSIONS_DIR / constants.SESSION_FILENAME).exists()
            # NOT in the root
            assert not (d / constants.SESSION_FILENAME).exists()
        finally:
            session.clear_session()  # detach log handle before tempdir cleanup


def test_load_legacy_root_session_writes_new_into_sessions():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            legacy1 = d / "zfisher_session_1.json"
            legacy1.write_text(json.dumps({"output_dir": str(d), "processed_files": {}}))
            # a legacy session_2 also sits in the root → next free number is 3
            (d / "zfisher_session_2.json").write_text(json.dumps({"output_dir": str(d)}))

            session.clear_session()
            session.load_session_file(legacy1)

            assert session.get_data("session_filename") == "zfisher_session_3.json"
            # new file lives in sessions/, numbered across BOTH layouts
            assert (d / constants.SESSIONS_DIR / "zfisher_session_3.json").exists()
            assert not (d / "zfisher_session_3.json").exists()
            # the original legacy file is left untouched
            assert legacy1.exists()
        finally:
            session.clear_session()


def test_load_from_subfolder_without_output_dir_derives_root():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        try:
            sess = d / constants.SESSIONS_DIR
            sess.mkdir()
            f = sess / "zfisher_session_1.json"
            f.write_text(json.dumps({"processed_files": {}}))  # NO output_dir stored

            session.clear_session()
            session.load_session_file(f)

            # root derived as the grandparent of a file under sessions/
            assert Path(session.get_data("output_dir")) == d
            assert (sess / "zfisher_session_2.json").exists()
        finally:
            session.clear_session()


def test_session_exists_checks_both_locations():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        assert not session._session_exists(d, "zfisher_session_1.json")
        (d / "zfisher_session_1.json").write_text("{}")  # legacy root
        assert session._session_exists(d, "zfisher_session_1.json")
        (d / constants.SESSIONS_DIR).mkdir()
        (d / constants.SESSIONS_DIR / "zfisher_session_5.json").write_text("{}")
        assert session._session_exists(d, "zfisher_session_5.json")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed")
