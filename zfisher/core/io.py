import nd2
import numpy as np
from dataclasses import dataclass
import tifffile
import json
from pathlib import Path

@dataclass
class FISHSession:
    """
    A data class to hold information from a single imaging session,
    typically loaded from an .nd2 file.

    Attributes
    ----------
    data : np.ndarray
        The image data in (Z, C, Y, X) order.
    voxels : tuple
        A tuple representing voxel size (dz, dy, dx) in microns.
    channels : list
        A list of channel names as strings.
    path : str
        The file path to the original image file.
    metadata : dict, optional
        A dictionary containing raw metadata from the file.
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
        # Default metadata (generic names and 1.0 scale)
        self.channels = [f"Ch{i+1}" for i in range(self.data.shape[1])]
        self.voxels = (1.0, 1.0, 1.0) # (dz, dy, dx)

    def _load_data(self, path_str: str):
        """
        Loads image data from a TIFF file and normalizes its dimensions.

        Parameters
        ----------
        path_str : str
            The file path to the TIFF image.

        Returns
        -------
        np.ndarray
            The image data normalized to (Z, C, Y, X) order.
        """
        with tifffile.TiffFile(path_str) as tif:
            data = tif.asarray()
            self.ome_metadata = tif.ome_metadata

        # Normalize to (Z, C, Y, X)
        if data.ndim == 2:   # (Y, X) -> (1, 1, Y, X)
            return data[np.newaxis, np.newaxis, :, :]
        elif data.ndim == 3: # (Z, Y, X) -> (Z, 1, Y, X)
            return data[:, np.newaxis, :, :]
        elif data.ndim == 4:
            # Heuristic: If dim0 < dim1, assume (C, Z, Y, X) and swap to (Z, C, Y, X)
            if data.shape[0] < data.shape[1]:
                return np.moveaxis(data, 0, 1)
            return data
        return data

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
        # Data is loaded as (C, Z, Y, X) or (Z, Y, X) etc.
        # f.asarray() brings it to a standard order, often TZCYX or TCYX
        img = f.asarray()

        # Ensure data is at least 4D (Z, C, Y, X)
        if img.ndim == 3: # (Z, Y, X) -> (Z, 1, Y, X)
            img = img[:, np.newaxis, :, :]
        elif img.ndim == 2: # (Y, X) -> (1, 1, Y, X)
            img = img[np.newaxis, np.newaxis, :, :]

        # Voxel size handling
        v_size = (f.voxel_size().z, f.voxel_size().y, f.voxel_size().x)
        try:
            ch_names = [c.channel.name for c in f.metadata.channels]
        except (AttributeError, TypeError):
            ch_names = [f"Channel_{i+1}" for i in range(img.shape[1])]

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
        print(f"Converting {nd2_session.path} to OME-TIFF...")

        metadata = {
            'axes': 'ZCYX',
            'Channel': {'Name': nd2_session.channels},
            'PhysicalSizeX': nd2_session.voxels[2], 'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeY': nd2_session.voxels[1], 'PhysicalSizeYUnit': 'µm',
            'PhysicalSizeZ': nd2_session.voxels[0], 'PhysicalSizeZUnit': 'µm',
        }

        tifffile.imwrite(ome_tif_path, nd2_session.data, metadata=metadata, photometric='minisblack')
        print(f"Saved OME-TIFF to {ome_tif_path}")

        if progress_callback:
            progress_callback(f"Extracting OME-XML for {prefix}...")

        with tifffile.TiffFile(ome_tif_path) as tf:
            if tf.ome_metadata:
                xml_path = output_dir / f"{prefix}_metadata.ome.xml"
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(tf.ome_metadata)
                print(f"Exported OME-XML metadata to {xml_path}")

        if hasattr(nd2_session, 'metadata') and nd2_session.metadata:
            json_path = output_dir / f"{prefix}_metadata.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(nd2_session.metadata, f, indent=4, default=str)
            print(f"Exported ND2 metadata to {json_path}")

    except Exception as e:
        print(f"Failed to convert ND2 to OME-TIFF: {e}")