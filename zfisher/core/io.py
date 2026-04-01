import logging
import nd2
import numpy as np
from dataclasses import dataclass
import tifffile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from .. import constants

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

class TiffSession:
    """
    A helper class to wrap TIFF data with an interface similar to FISHSession.
    """
    def __init__(self, path: Path):
        self.path = str(path)
        self.ome_metadata = None
        self.data = self._load_data(self.path)

        # First attempt to parse OME-XML
        if self.ome_metadata:
            try:
                self._parse_ome_metadata()
            except Exception as e:
                logger.warning("Could not parse OME-XML from %s: %s", path, e)
                self._set_default_metadata()
        else:
            self._set_default_metadata()

    def _set_default_metadata(self):
        """
        Sets default metadata and looks for 'Labels' in ImageJ metadata as a fallback.
        """
        self.voxels = (1.0, 1.0, 1.0)
        
        # ImageJ metadata fallback: Some TIFF files store channel names
        # in ImageJ 'Labels' metadata instead of OME-XML.
        try:
            with tifffile.TiffFile(self.path) as tif:
                ij_meta = tif.imagej_metadata
                if ij_meta and 'Labels' in ij_meta:
                    # 'Labels' is a list of channel names
                    self.channels = ij_meta['Labels'][:self.data.shape[1]]
                    logger.info("Metadata fallback: loaded channel names %s from ImageJ Labels.", self.channels)
                    
                    # Try to extract Z-spacing if available
                    if 'spacing' in ij_meta:
                        self.voxels = (float(ij_meta['spacing']), 1.0, 1.0)
                    return
        except Exception as e:
            logger.warning("Could not parse ImageJ metadata from %s: %s", self.path, e)

        # Final fallback to generic naming if no labels are found
        self.channels = [f"Ch{i+1}" for i in range(self.data.shape[1])]
        logger.warning("No channel metadata found for %s. Using generic names: %s. Voxel size defaulting to 1.0 µm — physical measurements may be inaccurate.", self.path, self.channels)

    def _parse_ome_metadata(self):
        """Parses channel names and voxel sizes from OME-XML string."""
        root = ET.fromstring(self.ome_metadata)
        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}

        # Extract Channel Names
        channels = root.findall('.//ome:Channel', ns)
        if channels:
            num_ch_data = self.data.shape[1]
            self.channels = [ch.get('Name', f"Ch{i}") for i, ch in enumerate(channels)][:num_ch_data]
        else:
            self._set_default_metadata()

        # Extract Voxel Sizes
        pixels_element = root.find('.//ome:Pixels', ns)
        if pixels_element is not None:
            dz = float(pixels_element.get('PhysicalSizeZ', 1.0))
            dy = float(pixels_element.get('PhysicalSizeY', 1.0))
            dx = float(pixels_element.get('PhysicalSizeX', 1.0))
            raw_voxels = (dz, dy, dx)
            self.voxels = tuple(
                s if isinstance(s, (int, float)) and not np.isnan(s) and s > 0 else 1.0
                for s in raw_voxels
            )
            if self.voxels != raw_voxels:
                logger.warning("Some voxel sizes were invalid in OME metadata (raw: %s), defaulting to 1.0 µm where needed.", raw_voxels)

    def _load_data(self, path_str: str):
        """Loads image data and normalizes dimensions to (Z, C, Y, X)."""
        with tifffile.TiffFile(path_str) as tif:
            data = tif.asarray()
            self.ome_metadata = tif.ome_metadata

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

