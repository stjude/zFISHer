import logging
import nd2
import numpy as np
from dataclasses import dataclass
import tifffile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from collections import defaultdict
import os
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
        
        # Check for ImageJ metadata fallback (Fix for cropper script)
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
        except Exception:
            pass

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

        if data.ndim == 2:   # (Y, X) -> (1, 1, Y, X)
            return data[np.newaxis, np.newaxis, :, :]
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

        # Add singleton Z axis if missing
        if 'Z' not in order:
            img = np.expand_dims(img, axis=0)
            order.insert(0, 'Z')

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

def discover_nd2_pairs(input_dir: Path):
    """
    Discovers pairs of R1 and R2 .nd2 files in a directory.
    Assumes a naming convention where files are differentiated by 'R1' and 'R2'.
    Example: 'FOV1_R1.nd2' and 'FOV1_R2.nd2'.
    """
    files = list(input_dir.glob('*.nd2'))
    groups = defaultdict(dict)

    # Try to group by replacing R1/R2 identifiers
    for f in files:
        # Create a base name by removing r1/r2 and common separators
        base_name = re.sub(r'[._-]?r[12][._-]?', '', f.name, flags=re.IGNORECASE)
        if 'r1' in f.name.lower():
            groups[base_name]['r1'] = f
        elif 'r2' in f.name.lower():
            groups[base_name]['r2'] = f

    paired_list = []
    for base, paths in groups.items():
        if 'r1' in paths and 'r2' in paths:
            # Create a clean FOV name from the R1 file
            fov_name = re.sub(r'[._-]?r1[._-]?', '', paths['r1'].stem, flags=re.IGNORECASE)
            paired_list.append({'name': fov_name, 'r1': paths['r1'], 'r2': paths['r2']})

    # Fallback for non-standard names: pair by sorted order
    if not paired_list and len(files) >= 2:
        logger.warning("No R1/R2 pairs found by name. Assuming sorted file order represents pairs.")
        sorted_files = sorted(files)
        # If odd number of files, the last one is ignored
        for i in range(0, (len(sorted_files) // 2) * 2, 2):
            r1_path = sorted_files[i]
            r2_path = sorted_files[i+1]
            common_prefix = os.path.commonprefix([r1_path.stem, r2_path.stem]).strip('-_')
            fov_name = common_prefix if common_prefix else r1_path.stem
            paired_list.append({'name': fov_name, 'r1': r1_path, 'r2': r2_path})

    logger.info("Discovered %d file pairs for batch processing.", len(paired_list))
    return paired_list