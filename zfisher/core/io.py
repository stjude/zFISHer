import logging
import os
import zlib
import nd2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import tifffile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from .. import constants
from .timing import timed_stage

logger = logging.getLogger(__name__)

@dataclass
class FISHSession:
    """
    A data class to hold information from a single imaging session.
    """
    data: np.ndarray      # (Z, C, Y, X)
    voxels: tuple        # (dz, dy, dx) in microns
    channels: list       # ['DAPI', 'FISH1', ...]
    path: str
    metadata: dict = None # Store raw metadata

_OME_NS = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}


def _tiff_default_metadata(tif, n_channels, path):
    """Channel names and voxel size from ImageJ metadata, or generic names and
    1.0 µm when there is none."""
    try:
        ij_meta = tif.imagej_metadata
        if ij_meta and 'Labels' in ij_meta:
            channels = ij_meta['Labels'][:n_channels]
            logger.info("Metadata fallback: loaded channel names %s from ImageJ Labels.", channels)
            voxels = (float(ij_meta['spacing']), 1.0, 1.0) if 'spacing' in ij_meta else (1.0, 1.0, 1.0)
            return channels, voxels
    except Exception as e:
        logger.warning("Could not parse ImageJ metadata from %s: %s", path, e)
    channels = [f"Ch{i+1}" for i in range(n_channels)]
    logger.warning("No channel metadata found for %s. Using generic names: %s. Voxel size defaulting to 1.0 µm — physical measurements may be inaccurate.", path, channels)
    return channels, (1.0, 1.0, 1.0)


def _tiff_metadata(tif, n_channels, path):
    """``(channels, voxels)`` of an open TIFF holding ``n_channels`` channels:
    from OME-XML when present, else ImageJ metadata, else defaults. The one
    reading of TIFF metadata, shared by the full loader and ``read_channel``
    so a channel name means the same channel in both."""
    if not tif.ome_metadata:
        return _tiff_default_metadata(tif, n_channels, path)
    try:
        root = ET.fromstring(tif.ome_metadata)
        found = root.findall('.//ome:Channel', _OME_NS)
        if found:
            channels = [ch.get('Name', f"Ch{i}") for i, ch in enumerate(found)][:n_channels]
            voxels = (1.0, 1.0, 1.0)
        else:
            channels, voxels = _tiff_default_metadata(tif, n_channels, path)
        pixels_element = root.find('.//ome:Pixels', _OME_NS)
        if pixels_element is not None:
            raw_voxels = tuple(float(pixels_element.get(f'PhysicalSize{a}', 1.0)) for a in "ZYX")
            voxels = tuple(
                s if isinstance(s, (int, float)) and not np.isnan(s) and s > 0 else 1.0
                for s in raw_voxels
            )
            if voxels != raw_voxels:
                logger.warning("Some voxel sizes were invalid in OME metadata (raw: %s), defaulting to 1.0 µm where needed.", raw_voxels)
        return channels, voxels
    except Exception as e:
        logger.warning("Could not parse OME-XML from %s: %s", path, e)
        return _tiff_default_metadata(tif, n_channels, path)


def _tiff_layout(tif):
    """Where the channels are in a TIFF round: ``(lead_shape, channel_axis, n_channels)``.

    ``lead_shape`` is the series shape before Y and X. A 3D stack has one channel
    (``channel_axis`` None). For 4D the channel axis is the smaller of the first
    two, the same rule the full loader applies to the array."""
    shape = tif.series[0].shape
    if len(shape) == 2:
        raise ValueError(
            "zFISHer requires 3D volumetric data (Z-stacks). "
            "The loaded file is a 2D image (e.g., MIP or EDF). "
            "Please provide the original Z-stack instead."
        )
    lead = tuple(shape[:-2])
    if len(lead) == 1:
        return lead, None, 1
    if len(lead) == 2:
        c_axis = 0 if lead[0] < lead[1] else 1
        return lead, c_axis, lead[c_axis]
    raise ValueError(f"Cannot read channels from a TIFF of shape {shape}")


class TiffSession:
    """
    A helper class to wrap TIFF data with an interface similar to FISHSession.
    """
    def __init__(self, path: Path):
        self.path = str(path)
        self.data = self._load_data(self.path)
        with tifffile.TiffFile(self.path) as tif:
            self.ome_metadata = tif.ome_metadata
            self.channels, self.voxels = _tiff_metadata(tif, self.data.shape[1], self.path)

    def _load_data(self, path_str: str):
        """Loads image data and normalizes dimensions to (Z, C, Y, X)."""
        with tifffile.TiffFile(path_str) as tif:
            data = tif.asarray()

        # Normalize to (Z, C, Y, X) format:
        # - 2D (Y, X) → (1, 1, Y, X): single slice, single channel
        # - 3D (Z, Y, X) → (Z, 1, Y, X): Z-stack, single channel
        # - 4D: if shape[0] < shape[1], assume (C, Z, Y, X) → swap to (Z, C, Y, X)
        #   (C dimension is typically smallest; Z-stacks have ~50 slices, channels ~3-5)
        if data.ndim == 2:
            raise ValueError(
                "zFISHer requires 3D volumetric data (Z-stacks). "
                "The loaded file is a 2D image (e.g., MIP or EDF). "
                "Please provide the original Z-stack instead."
            )
        elif data.ndim == 3: # (Z, Y, X) -> (Z, 1, Y, X)
            return data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            if data.shape[0] < data.shape[1]: # (C, Z, Y, X) -> (Z, C, Y, X)
                return np.moveaxis(data, 0, 1)
            return data
        return data

# ... [Remaining functions load_nd2_session, load_image_session, etc. remain unchanged] ...
def load_nd2_session(path: str) -> FISHSession:
    """
    Loads an ND2 file into a FISHSession object.

    Parameters
    ----------
    path : str
        The file path to the .nd2 file.

    Returns
    -------
    FISHSession
        An object containing the data and metadata from the ND2 file.
    """
    with nd2.ND2File(path) as f:
        # Read data and axis order
        img = f.asarray()

        # The `sizes` attribute is a reliable way to get the axis names and order,
        # which is more robust across different versions of the nd2 library.
        if not hasattr(f, 'sizes') or not f.sizes:
            raise AttributeError("Could not determine axis order from nd2 file. The 'sizes' attribute is missing or empty.")
        original_axes = "".join(f.sizes.keys())
        order = list(f.sizes.keys())

        # If a T axis is present and singular, squeeze it
        if 'T' in order:
            t_idx = order.index('T')
            if img.shape[t_idx] == 1:
                img = np.squeeze(img, axis=t_idx)
                order.pop(t_idx)
        
        # Add singleton C axis if missing
        if 'C' not in order:
            # Find where to insert C. Usually after Z.
            if 'Z' in order:
                c_idx = order.index('Z') + 1
            else:
                c_idx = 0
            img = np.expand_dims(img, axis=c_idx)
            order.insert(c_idx, 'C')

        # Reject 2D data (no Z axis = MIP/EDF, not a Z-stack)
        if 'Z' not in order:
            raise ValueError(
                "zFISHer requires 3D volumetric data (Z-stacks). "
                "The loaded ND2 file has no Z axis (e.g., MIP or EDF). "
                "Please provide the original Z-stack instead."
            )

        # Now we should have at least Z, C, Y, X. Let's enforce the order.
        target_order = ['Z', 'C', 'Y', 'X']
        
        if not all(axis in order for axis in target_order):
             raise ValueError(f"Could not normalize axes. Original: {original_axes}, Current: {order}")

        current_indices = [order.index(axis) for axis in target_order]
        img = np.transpose(img, current_indices)

        # Voxel size handling
        v_size_raw = (f.voxel_size().z, f.voxel_size().y, f.voxel_size().x)
        # Sanitize voxel sizes: replace None, zero, negative, or NaN with 1.0
        v_size = tuple(
            s if isinstance(s, (int, float)) and not np.isnan(s) and s > 0 else 1.0
            for s in v_size_raw
        )
        if v_size != v_size_raw:
            logger.warning("Some voxel sizes were invalid in ND2 metadata (raw: %s), defaulting to 1.0 µm where needed.", v_size_raw)
        try:
            ch_names = [c.channel.name for c in f.metadata.channels]
        except (AttributeError, TypeError):
            ch_names = [f"Channel_{i+1}" for i in range(img.shape[1])]
            logger.warning("Could not read channel names from ND2 metadata. Using generic names: %s", ch_names)

    return FISHSession(data=img, voxels=v_size, channels=ch_names, path=path, metadata=f.metadata)

def peek_channel_names(path):
    """Read channel names from an image file without loading pixel data.

    Supports .nd2 and TIFF/OME-TIFF files.  Returns a list of channel
    name strings, or ``None`` if the metadata cannot be read.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == '.nd2':
            with nd2.ND2File(str(path)) as f:
                try:
                    return [c.channel.name for c in f.metadata.channels]
                except (AttributeError, TypeError):
                    # Fall back to sizes dict
                    n_ch = f.sizes.get('C', 1)
                    return [f"Channel_{i+1}" for i in range(n_ch)]

        # TIFF / OME-TIFF
        with tifffile.TiffFile(str(path)) as tif:
            # Try OME-XML first
            if tif.ome_metadata:
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                channels = root.findall('.//ome:Channel', ns)
                if channels:
                    return [ch.get('Name', f"Ch{i}") for i, ch in enumerate(channels)]

            # ImageJ metadata fallback
            ij_meta = tif.imagej_metadata
            if ij_meta and 'Labels' in ij_meta:
                return list(ij_meta['Labels'])

        return None
    except Exception as exc:
        logger.debug("peek_channel_names failed for %s: %s", path, exc)
        return None


def peek_z_depth(path):
    """Read the Z depth from an image file without loading pixel data.

    Returns the number of Z slices as an int, or ``None`` if it cannot
    be determined from metadata alone.
    """
    path = Path(path)
    try:
        if path.suffix.lower() == '.nd2':
            with nd2.ND2File(str(path)) as f:
                return f.sizes.get('Z', None)

        with tifffile.TiffFile(str(path)) as tif:
            if tif.ome_metadata:
                root = ET.fromstring(tif.ome_metadata)
                ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                pixels = root.find('.//ome:Pixels', ns)
                if pixels is not None:
                    sz = pixels.get('SizeZ')
                    if sz is not None:
                        return int(sz)
            # Fallback: check shape from series
            if tif.series:
                shape = tif.series[0].shape
                if len(shape) >= 3:
                    return shape[0] if len(shape) == 3 else shape[-3]
        return None
    except Exception as exc:
        logger.debug("peek_z_depth failed for %s: %s", path, exc)
        return None


@timed_stage("load_image", fields=lambda a: {"file": Path(a["path"]).name})
def load_image_session(path: Path):
    """
    Loads an image file (ND2 or TIFF) and returns a session object.

    This function acts as a factory, dispatching to the appropriate loader
    based on the file extension.

    Parameters
    ----------
    path : Path
        The path to the image file.

    Returns
    -------
    FISHSession or TiffSession
        An object containing the loaded image data and metadata.
    """
    if path.suffix.lower() == '.nd2':
        return load_nd2_session(str(path))
    else:
        return TiffSession(path)
    
def find_nuclear_channel(channels):
    """
    Auto-detect the nuclear stain channel from a list of channel names.

    Checks each channel against ``constants.NUCLEAR_STAIN_NAMES`` using
    case-insensitive substring matching.

    Parameters
    ----------
    channels : list[str]
        Channel names from the imaging session.

    Returns
    -------
    str or None
        The first matching channel name, or None if no match is found.
    """
    for ch in channels:
        for nuc_name in constants.NUCLEAR_STAIN_NAMES:
            if nuc_name.upper() in ch.upper():
                return ch
    return None


def get_channel_data(session, target_name=constants.DAPI_CHANNEL_NAME):
    """Finds a channel by name in the session metadata and returns the 3D array."""
    try:
        # session.channels is populated by load_nd2_session
        idx = session.channels.index(target_name)
        return session.data[:, idx, :, :]
    except ValueError:
        logger.warning("Channel '%s' not found. Defaulting to index 0.", target_name)
        return session.data[:, 0, :, :]


# --- One channel from a round image, and finding a round image again --------
#
# Re-detecting one channel of a reopened session needs that channel's raw image,
# not the whole round. read_channel reads one channel without loading the
# others. resolve_raw_path finds the round's image: the input/ copy written for
# ND2 inputs, then the path stored in the session, then a file the user points
# to, checked against what the session recorded about the image when it was
# created.

def _channel_index(channel, names):
    if isinstance(channel, (int, np.integer)):
        if not 0 <= int(channel) < len(names):
            raise IndexError(f"Channel {channel} out of range; the image has {len(names)} channels")
        return int(channel)
    if channel in names:
        return names.index(channel)
    raise KeyError(f"No channel named {channel!r}; the image has {names}")


def _nd2_layout(f):
    """Axis order of an open ND2 file and its channel names, refusing what the
    full loader also refuses."""
    sizes = dict(f.sizes)
    if 'Z' not in sizes:
        raise ValueError(
            "zFISHer requires 3D volumetric data (Z-stacks). "
            "The loaded ND2 file has no Z axis (e.g., MIP or EDF). "
            "Please provide the original Z-stack instead."
        )
    extra = {ax: n for ax, n in sizes.items() if ax not in "ZCYX" and n != 1}
    if extra:
        raise ValueError(f"Cannot read channels from an ND2 file with axes {sizes}")
    try:
        names = [c.channel.name for c in f.metadata.channels]
    except (AttributeError, TypeError):
        names = [f"Channel_{i+1}" for i in range(sizes.get('C', 1))]
    return sizes, names


def read_channel(path, channel):
    """One channel of a round image as a (Z, Y, X) array, read without loading
    the round's other channels.

    ``channel`` is a channel name, or an index. The result equals that channel
    of ``load_image_session(path)``; an unknown name raises ``KeyError`` rather
    than falling back to the first channel.
    """
    path = Path(path)
    if path.suffix.lower() == '.nd2':
        with nd2.ND2File(str(path)) as f:
            sizes, names = _nd2_layout(f)
            idx = _channel_index(channel, names)
            key = tuple(idx if ax == 'C' else slice(None) if ax in "ZYX" else 0 for ax in sizes)
            data = np.asarray(f.to_dask()[key])
            kept = [ax for ax in sizes if ax in "ZYX"]
            return np.transpose(data, [kept.index(ax) for ax in "ZYX"])

    with tifffile.TiffFile(str(path)) as tif:
        lead, c_axis, n_channels = _tiff_layout(tif)
        names, _voxels = _tiff_metadata(tif, n_channels, str(path))
        idx = _channel_index(channel, names)
        series = tif.series[0]
        if c_axis is None:
            return series.asarray()
        n_z = lead[1 - c_axis]
        if tuple(series.keyframe.shape) == tuple(series.shape[-2:]):
            # One page per (Z, C) plane: read this channel's pages only.
            pages = [int(np.ravel_multi_index((idx, z) if c_axis == 0 else (z, idx), lead))
                     for z in range(n_z)]
            return tif.asarray(key=pages, series=0).reshape((n_z,) + tuple(series.shape[-2:]))
        logger.warning("%s does not store one plane per page; reading the whole round "
                       "to take channel %r", path.name, channel)
        data = series.asarray()
        return np.take(data, idx, axis=c_axis)


def image_identity(path):
    """What a round image is, read from its metadata: file name and size, shape
    (Z, C, Y, X), dtype, channel names and voxel size. Recorded when a session
    is created, so a file pointed to later can be checked against it."""
    path = Path(path)
    if path.suffix.lower() == '.nd2':
        with nd2.ND2File(str(path)) as f:
            sizes, names = _nd2_layout(f)
            shape = [sizes['Z'], sizes.get('C', 1), sizes['Y'], sizes['X']]
            dtype = str(np.dtype(f.dtype))
            vs = f.voxel_size()
            voxels = [float(v) if v and v > 0 else 1.0 for v in (vs.z, vs.y, vs.x)]
    else:
        with tifffile.TiffFile(str(path)) as tif:
            lead, c_axis, n_channels = _tiff_layout(tif)
            names, voxels = _tiff_metadata(tif, n_channels, str(path))
            z = lead[0] if c_axis is None else lead[1 - c_axis]
            shape = [z, n_channels] + list(tif.series[0].shape[-2:])
            dtype = str(tif.series[0].dtype)
            voxels = [float(v) for v in voxels]
    return {"file": path.name, "size_bytes": path.stat().st_size, "shape_zcyx": [int(v) for v in shape],
            "dtype": dtype, "channels": list(names), "voxels": voxels}


def identity_differences(recorded, found):
    """How ``found`` differs from the ``recorded`` identity of a round image,
    as readable lines; empty if they agree. A different file size alone is
    listed but expected for a re-exported file."""
    labels = {"shape_zcyx": "shape (Z, C, Y, X)", "dtype": "data type", "channels": "channels",
              "voxels": "voxel size (µm)", "size_bytes": "file size (bytes)"}
    out = []
    for key, label in labels.items():
        a, b = recorded.get(key), found.get(key)
        if key == "voxels" and a is not None and b is not None:
            same = np.allclose(a, b, rtol=1e-6)
        else:
            same = a == b
        if a is not None and not same:
            out.append(f"{label}: recorded {a}, this file has {b}")
    return out


class RawImageNotFound(FileNotFoundError):
    """A round's raw image could not be found and nobody was asked for it."""


def _input_copy(round_id):
    from . import session
    out = session.get_data("output_dir")
    if not out:
        return None
    path = Path(out) / constants.INPUT_DIR / f"{round_id}_converted.ome.tif"
    return path if path.exists() else None


def record_raw_identity(round_id, path):
    """Store ``image_identity(path)`` as the session's record of ``round_id``'s
    raw image, unless one is already stored. Never raises: a file whose
    metadata cannot be read is simply not recorded."""
    from . import session
    identities = dict(session.get_data("raw_identity", default={}) or {})
    if round_id in identities:
        return
    try:
        identities[round_id] = image_identity(path)
    except Exception as exc:
        logger.warning("Could not record the identity of %s image %s: %s", round_id, path, exc)
        return
    session.update_data("raw_identity", identities)


def resolve_raw_path(round_id, chooser=None, confirm=None):
    """The raw image of ``round_id`` ('R1' or 'R2') for this session.

    Tried in order: the ``input/`` copy written for ND2 inputs, the path stored
    in the session, then ``chooser(round_id, missing_path)``, which returns a
    path or None (the GUI passes a file dialog). A chosen file is compared with
    the identity recorded when the session was created; if they differ,
    ``confirm(round_id, path, differences)`` decides whether to use it (default:
    use it). An accepted file is stored in the session, so the user is asked
    once, and the re-point is recorded. Without a chooser, or if the user
    cancels, raises ``RawImageNotFound`` naming the round and the file.
    """
    from . import session
    key = "r1_path" if round_id == "R1" else "r2_path"
    stored = session.get_data(key)
    if stored and Path(stored).exists():
        record_raw_identity(round_id, stored)      # sessions made before identities were kept
    copy = _input_copy(round_id)
    if copy is not None:
        return copy
    if stored and Path(stored).exists():
        return Path(stored)
    if chooser is None:
        raise RawImageNotFound(
            f"The {round_id} raw image {stored or '(none recorded)'} cannot be found. "
            f"Give the image's new location for {round_id}.")
    chosen = chooser(round_id, stored)
    if not chosen:
        raise RawImageNotFound(f"No file was chosen for the {round_id} raw image {stored}.")
    chosen = Path(chosen)
    recorded = (session.get_data("raw_identity", default={}) or {}).get(round_id)
    differences = []
    if recorded:
        try:
            differences = identity_differences(recorded, image_identity(chosen))
        except Exception as exc:
            differences = [f"could not read the file's metadata: {exc}"]
    else:
        logger.info("No recorded identity for %s; %s is used without a check", round_id, chosen.name)
    if differences and confirm is not None and not confirm(round_id, chosen, differences):
        raise RawImageNotFound(f"The file chosen for {round_id}, {chosen}, was not accepted.")
    import datetime
    repoints = list(session.get_data("raw_repoints", default=[]) or [])
    repoints.append({"round": round_id, "from": stored, "to": str(chosen), "differences": differences,
                     "time": datetime.datetime.now().isoformat(timespec="seconds")})
    session.update_data("raw_repoints", repoints)
    session.update_data(key, str(chosen))
    logger.info("Raw image for %s re-pointed from %s to %s%s", round_id, stored, chosen,
                f" ({len(differences)} difference(s) accepted)" if differences else "")
    return chosen
    
@timed_stage("convert_nd2_to_ome", fields=lambda a: {"round": a["prefix"]})
def convert_nd2_to_ome(
    nd2_session: FISHSession,
    output_dir: Path,
    prefix: str,
    progress_callback=None
):
    """
    Converts an ND2 session to OME-TIFF and saves associated metadata.

    This function saves the image data as a multi-channel OME-TIFF file
    and also exports the OME-XML and raw ND2 metadata to separate files.

    Parameters
    ----------
    nd2_session : FISHSession
        The session object loaded from an ND2 file.
    output_dir : Path
        The directory where the converted files will be saved.
    prefix : str
        A prefix (e.g., 'R1', 'R2') to use for the output filenames.
    progress_callback : callable, optional
        A function to call with progress messages.
    """
    if not output_dir:
        return

    try:
        if progress_callback:
            progress_callback(f"Converting {prefix} to OME-TIFF...")

        ome_tif_path = output_dir / f"{prefix}_converted.ome.tif"
        logger.info("Converting %s to OME-TIFF...", nd2_session.path)

        metadata = {
            'axes': 'ZCYX',
            'Channel': {'Name': nd2_session.channels},
            'PhysicalSizeX': nd2_session.voxels[2], 'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': nd2_session.voxels[1], 'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': nd2_session.voxels[0], 'PhysicalSizeZUnit': 'µm',
        }

        tifffile.imwrite(ome_tif_path, nd2_session.data, metadata=metadata, photometric='minisblack')
        logger.info("Saved OME-TIFF to %s", ome_tif_path)

        if progress_callback:
            progress_callback(f"Extracting OME-XML for {prefix}...")

        with tifffile.TiffFile(ome_tif_path) as tf:
            if tf.ome_metadata:
                xml_path = output_dir / f"{prefix}_metadata.ome.xml"
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(tf.ome_metadata)
                logger.info("Exported OME-XML metadata to %s", xml_path)

        if hasattr(nd2_session, 'metadata') and nd2_session.metadata:
            json_path = output_dir / f"{prefix}_metadata.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(nd2_session.metadata, f, indent=4, default=str)
            logger.info("Exported ND2 metadata to %s", json_path)

    except Exception as e:
        logger.error("Failed to convert ND2 to OME-TIFF: %s", e)


MASK_EXTENSION = ".ome.tif"
_LEGACY_MASK_EXTENSION = ".tif"


def mask_path(directory, name):
    """The path a mask named ``name`` is written to in ``directory``."""
    return Path(directory) / f"{name}{MASK_EXTENSION}"


def find_mask(directory, name):
    """The mask file named ``name`` in ``directory``: ``<name>.ome.tif``, or
    ``<name>.tif`` as written before masks were OME-TIFF; None if neither exists."""
    for ext in (MASK_EXTENSION, _LEGACY_MASK_EXTENSION):
        path = Path(directory) / f"{name}{ext}"
        if path.exists():
            return path
    return None


class LabelTiles:
    """The compressed tiles of one label volume, kept between saves so that a
    save compresses again only the tiles an edit touched.

    A mask is written as a tiled OME-TIFF: every plane is cut into 256 x 256
    tiles, each compressed on its own. Compressing the whole of a typical
    71 x 2044 x 2048 mask takes about 0.2 s on 11 cores and 0.9 s on one; one
    nucleus touches about a hundred of its 4,544 tiles, which take a few ms.
    The compressed tiles of such a mask take about 19 MB.

    ``mark`` and ``mark_all`` record what changed since the last save. A change
    that is not marked leaves its tiles stale in the file until a tile is
    marked again or the whole volume is, so callers mark every in-place write.
    """

    TILE = 256

    def __init__(self, data):
        self.shape = data.shape
        if data.ndim != 3:
            raise ValueError(f"A mask must be a (Z, Y, X) volume, not {data.ndim}D")
        dtype = np.dtype(data.dtype)
        if dtype.kind not in "iu":
            raise ValueError(f"Labels must be integers, not {dtype}")
        # OME has no 64-bit integers; label values fit uint32 in practice and
        # encoding checks every tile.
        self.dtype = np.dtype(np.uint32) if dtype.itemsize == 8 else dtype.newbyteorder("=")
        self._ny = -(-self.shape[1] // self.TILE)
        self._nx = -(-self.shape[2] // self.TILE)
        self._tiles = [None] * (self.shape[0] * self._ny * self._nx)
        self._dirty = set()
        self._all_dirty = True

    def fits(self, data):
        """Whether these tiles describe an array of ``data``'s shape."""
        return data.shape == self.shape

    def mark(self, region):
        """Record that the voxels in ``region`` (a tuple of slices over all
        axes of the volume, as from ``label_edits.index_extent``) changed."""
        if region is None or self._all_dirty:
            return
        rows = range(region[1].start // self.TILE, (region[1].stop - 1) // self.TILE + 1)
        cols = range(region[2].start // self.TILE, (region[2].stop - 1) // self.TILE + 1)
        for z in range(region[0].start, region[0].stop):
            for ty in rows:
                for tx in cols:
                    self._dirty.add((z * self._ny + ty) * self._nx + tx)

    def mark_all(self):
        """Record that the whole volume changed."""
        self._all_dirty = True
        self._dirty.clear()

    def _encode(self, data, index):
        z, rest = divmod(index, self._ny * self._nx)
        ty, tx = divmod(rest, self._nx)
        tile = data[z, ty * self.TILE:(ty + 1) * self.TILE, tx * self.TILE:(tx + 1) * self.TILE]
        if tile.dtype.itemsize == 8 and tile.size and (tile.min() < 0 or tile.max() > 0xFFFFFFFF):
            raise ValueError("Label values do not fit in 32 bits, which OME-TIFF needs")
        tile = np.ascontiguousarray(tile, dtype=self.dtype)
        if tile.shape != (self.TILE, self.TILE):
            tile = np.pad(tile, ((0, self.TILE - tile.shape[0]), (0, self.TILE - tile.shape[1])))
        # zlib level 1: as small as the default level here and much faster.
        return zlib.compress(tile, 1)

    def update(self, data):
        """Compress the changed tiles of ``data`` again. Returns how many."""
        if not self.fits(data):
            raise ValueError(f"Tiles of shape {self.shape} cannot hold data of shape {data.shape}")
        if self._all_dirty:
            indices = range(len(self._tiles))
            # zlib releases the GIL, so threads compress in parallel.
            with ThreadPoolExecutor(os.cpu_count() or 1) as pool:
                self._tiles = list(pool.map(lambda i: self._encode(data, i), indices))
        else:
            indices = sorted(self._dirty)
            for i in indices:
                self._tiles[i] = self._encode(data, i)
        self._all_dirty = False
        self._dirty.clear()
        return len(indices)

    @property
    def pending(self):
        """How many tiles the next ``update`` compresses."""
        return len(self._tiles) if self._all_dirty else len(self._dirty)

    @property
    def all_pending(self):
        """Whether the next ``update`` compresses every tile."""
        return self._all_dirty

    def __iter__(self):
        return iter(self._tiles)

    @property
    def nbytes(self):
        return sum(len(t) for t in self._tiles if t is not None)


def _ome_metadata(voxel_size):
    """OME metadata for a (Z, Y, X) mask; ``voxel_size`` is (z, y, x) in µm."""
    meta = {"axes": "ZYX"}
    if voxel_size is not None:
        for axis, size in zip("ZYX", voxel_size):
            size = float(size)
            if np.isfinite(size) and size > 0:
                meta[f"PhysicalSize{axis}"] = size
                meta[f"PhysicalSize{axis}Unit"] = "µm"
    return meta


def read_label_tif(path):
    """Read a mask as (Z, Y, X): one written by ``write_label_tif``, or a plain
    TIFF mask from before masks were OME-TIFF.

    ``tifffile.imread`` drops axes of size 1 from an OME-TIFF, so a one-plane
    mask would come back 2D; this keeps Z and drops only OME's T, C and S."""
    with tifffile.TiffFile(path) as tif:
        if not tif.is_ome:
            return tif.asarray()
        axes = tif.series[0].get_axes(False)
        data = tif.asarray(squeeze=False)
    return data[tuple(slice(None) if axis in "ZYX" else 0 for axis in axes)]


def write_label_tif(path, data, voxel_size=None, tiles=None):
    """
    Write a label mask as a tiled, compressed OME-TIFF without ever truncating
    the file that is already there.

    ``voxel_size`` is (z, y, x) in µm and goes into the OME metadata, so the
    mask opens at its true scale elsewhere (Fiji through Bio-Formats). Pass the
    ``LabelTiles`` kept for ``data`` as ``tiles`` to compress only the tiles
    marked since the last save; without it every tile is compressed.

    The mask goes to ``<name>.tmp`` first and is then swapped into place, so a
    write that dies halfway leaves the previous file intact. Windows refuses
    the swap while another program holds the file open; the file is then
    overwritten in place and a warning is logged. If that fails as well, the
    complete ``.tmp`` copy is left beside it.

    Returns ``"atomic"`` or ``"direct"``, whichever path wrote the file.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tiles is None:
        tiles = LabelTiles(data)
    tiles.update(data)

    def _write(target):
        # minisblack: without it tifffile stores a stack of exactly 3 or 4
        # slices as the planes of a colour image. ome=True because the
        # temporary name does not end in .ome.tif.
        tifffile.imwrite(target, iter(tiles), shape=tiles.shape, dtype=tiles.dtype,
                         tile=(LabelTiles.TILE, LabelTiles.TILE), compression='zlib',
                         photometric='minisblack', ome=True,
                         metadata=_ome_metadata(voxel_size))

    try:
        _write(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    try:
        os.replace(tmp, path)
        return "atomic"
    except OSError as e:
        logger.warning("Could not swap %s into place (%s); overwriting it directly.", path.name, e)
    _write(path)
    tmp.unlink(missing_ok=True)
    return "direct"

