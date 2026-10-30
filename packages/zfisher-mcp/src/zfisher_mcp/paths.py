"""The folders the server may read, and how paths cross its boundary.

A client names a session by its path relative to a configured root, and
results name files the same way. Nothing above a root ever leaves the server:
the lab's folder names carry sample and investigator names, and a result may
be sent to a model or across the network. A path in a result that lies under
no root (a raw image on another drive, say) is cut to its file name.

With one root, paths are relative to it (``"batch_03/dataset_A"``). With
several, each root has a name and paths start with it
(``"screen/batch_03/dataset_A"``).
"""
import re
from pathlib import Path, PurePosixPath


class OutsideRoots(ValueError):
    """A path that is absolute, goes up a folder, or leads outside every root."""


# Absolute paths inside free text (error messages): POSIX, drive letter or UNC.
# Each is cut to its last component. Folder names may hold spaces, but not at
# either end, so "copy /a/x.csv to /b/y.csv" is two paths; a file name ends at
# the first space. Whole-string paths, the usual case, are handled exactly.
_NOT_IN_NAME = r"\\/'\"<>|,;()\[\]\n"
_FOLDER = rf"[^\s{_NOT_IN_NAME}](?:[^{_NOT_IN_NAME}]*[^\s{_NOT_IN_NAME}])?"
_ABSOLUTE_IN_TEXT = re.compile(
    rf"(?<![\w.~])(?:/|[A-Za-z]:[\\/]|\\\\)(?:{_FOLDER}[\\/])+([^\s{_NOT_IN_NAME}]+)")
_DRIVE = re.compile(r"^[A-Za-z]:")


def _looks_absolute(text):
    return text.startswith(("/", "\\\\")) or bool(re.match(r"^[A-Za-z]:[\\/]", text))


class Roots:
    """The configured roots, by name; a single root has the name ``""``."""

    def __init__(self, roots):
        if not roots:
            raise ValueError("At least one root folder is needed.")
        self._roots = {}
        self._spellings = []                       # (absolute spelling, name), longest first
        for name, folder in roots.items():
            given = Path(folder).expanduser().absolute()
            resolved = given.resolve()
            if not resolved.is_dir():
                raise ValueError(f"The root {folder} is not a folder.")
            self._roots[name] = resolved
            for spelling in {str(given), str(resolved), given.as_posix(), resolved.as_posix()}:
                self._spellings.append((spelling.rstrip("/\\"), name))
        self._spellings.sort(key=lambda item: len(item[0]), reverse=True)
        # A root's spelling inside text, ending at a separator or the end of
        # the path (so "/data/batch" does not match "/data/batch2").
        self._in_text = [(re.compile(re.escape(s) + r"(?:[\\/]|(?=[\s'\",;:)\]]|\.(?:\s|$)|$))"), name)
                         for s, name in self._spellings]

    @classmethod
    def parse(cls, specs):
        """Roots from ``--root`` values, each ``PATH`` or ``NAME=PATH``. With
        several roots, a root given without a name is named after its folder."""
        named = {}
        for spec in specs:
            name, sep, folder = spec.partition("=")
            if not sep or not name or any(c in name for c in "/\\:"):
                name, folder = "", spec                     # a path that happens to hold "="
            named.setdefault(name, []).append(folder)
        if len(specs) == 1:
            (folders,) = named.values()
            return cls({"": folders[0]})
        roots = {}
        for name, folders in named.items():
            for folder in folders:
                label = name or Path(folder).expanduser().resolve().name
                if not label or "/" in label or "\\" in label or label in roots:
                    raise ValueError(f"Give the root {folder} a name of its own, as NAME={folder}.")
                roots[label] = folder
        return cls(roots)

    @property
    def names(self):
        return sorted(self._roots)

    @property
    def single(self):
        return list(self._roots) == [""]

    def folders(self):
        """``(name, folder)`` of every root."""
        return sorted(self._roots.items())

    def resolve(self, relative):
        """The folder or file ``relative`` names, which must lie inside a root
        (links are followed before the check)."""
        text = (relative or "").strip().replace("\\", "/")
        if _looks_absolute(text) or _DRIVE.match(text) or text.startswith("~"):
            raise OutsideRoots("Name the session by its path relative to the root, as list_sessions gives it, "
                               "not by an absolute path.")
        parts = [p for p in PurePosixPath(text).parts if p not in ("", ".")]
        if ".." in parts:
            raise OutsideRoots("Paths may not go up a folder ('..').")
        if self.single:
            name, rest = "", parts
        else:
            if not parts or parts[0] not in self._roots:
                raise OutsideRoots(f"Start the path with the name of a root: {', '.join(self.names)}.")
            name, rest = parts[0], parts[1:]
        root = self._roots[name]
        target = root.joinpath(*rest).resolve()
        if target != root and root not in target.parents:
            raise OutsideRoots("That path leads outside the root.")
        return target

    def relative(self, path):
        """``path`` as the server names it (relative to its root, prefixed with
        the root's name when there are several), or None outside every root."""
        text = str(path)
        for spelling, name in self._spellings:
            if text == spelling or text.startswith(spelling + "/") or text.startswith(spelling + "\\"):
                rest = text[len(spelling):].lstrip("/\\").replace("\\", "/")
                joined = "/".join(p for p in (name, rest) if p)
                return joined or "."
        resolved = Path(text).resolve() if _looks_absolute(text) else None
        if resolved is not None and str(resolved) != text:
            return self.relative(resolved)
        return None

    def scrub_text(self, text):
        """``text`` with every absolute path made relative to its root, or cut
        to its file name when it lies under no root."""
        if not text:
            return text
        if _looks_absolute(text) and "\n" not in text:
            inside = self.relative(text)
            if inside is not None:
                return inside
            return re.split(r"[\\/]", text.rstrip("/\\"))[-1]
        for pattern, name in self._in_text:
            text = pattern.sub(lambda m, name=name: (f"{name}/" if name else "") if m.group(0)[-1] in "/\\"
                               else (name or "."), text)
        return _ABSOLUTE_IN_TEXT.sub(lambda m: m.group(1), text)

    def scrub(self, value):
        """``value`` (dicts, lists and strings, nested) with ``scrub_text``
        applied to every string, keys included."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {self.scrub(k) if isinstance(k, str) else k: self.scrub(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.scrub(v) for v in value]
        return value
