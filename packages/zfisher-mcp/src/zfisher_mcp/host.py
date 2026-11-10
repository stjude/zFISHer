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
        """The session in ``folder``. ``writable`` opens it for writing; its
        changes are saved back into its newest session file, not a new
        numbered one, so a client's many small changes leave one file."""
        return api.open_session(folder, read_only=not writable, new_file=False)

    def run_write(self, what, folder, fn):
        """Run ``fn``, a change described by ``what`` to the files in ``folder``,
        and return its result."""
        return fn()
