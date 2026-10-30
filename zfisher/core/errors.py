"""The errors zFISHer raises on purpose, so callers can tell them from bugs.

Every class derives from ``ZFisherError`` and, where code already caught a
built-in type, from that type as well (a missing raw image is still a
``FileNotFoundError``), so nothing that caught the built-in breaks. Their
messages are written for the person running zFISHer: the GUI shows them as
they are, the CLI prints them, and other front ends pass them on unchanged.
"""


class ZFisherError(Exception):
    """An expected failure, with a message for the person running zFISHer."""


class SessionError(ZFisherError, RuntimeError):
    """A session cannot be created, opened, resumed or saved."""


class InputError(ZFisherError):
    """An input image or file is missing, unreadable or not what it should be."""


class SchemaError(ZFisherError, ValueError):
    """A file, table or setting is not in the form zFISHer reads and writes."""


class RegistrationError(ZFisherError, RuntimeError):
    """The rounds could not be registered or warped."""


class DetectionError(ZFisherError, RuntimeError):
    """Segmentation or puncta detection could not produce a result."""


class StaleError(ZFisherError, RuntimeError):
    """An output was asked for that is out of date with what it is made from."""
