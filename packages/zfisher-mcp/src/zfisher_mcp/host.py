"""How the server reaches sessions and runs changes.

By default (``DiskHost``) every call opens its session from disk and a change
runs on the thread that serves the call. A program that runs the server
inside itself, zFISHer's GUI, passes a host of its own: its open session is
used in place of a copy read from disk, and changes to it are handed to the
program, which runs them when nothing else is writing and shows the result.
"""
from zfisher.core import api


class DiskHost:
    """Sessions from disk; changes run where they are called."""

    def session(self, folder, writable=False):
        """The session in ``folder``. ``writable`` opens it for writing, which
        saves it under the next session number, as Load Session does."""
        return api.open_session(folder, read_only=not writable)

    def run_write(self, what, folder, fn):
        """Run ``fn``, a change described by ``what`` to the files in ``folder``,
        and return its result."""
        return fn()
