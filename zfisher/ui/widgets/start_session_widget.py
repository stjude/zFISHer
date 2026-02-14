from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari
from zfisher.core import session
from qtpy.QtCore import Qt
from .. import popups
from ._shared import load_raw_data_into_viewer
import zfisher.core.session as session
from ..constants import CHANNEL_COLORS
import numpy as np
import tifffile
from packaging.version import parse as parse_version
from qtpy.QtWidgets import QFrame

class StartSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        # The main container uses a simple vertical layout (labels=False).
        # This allows headers to span the full width.
        super().__init__(labels=False)
        self._viewer = viewer

        # --- Widgets ---
       # Create a native Qt horizontal line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("margin: 15px 0; color: #555;") # Adds spacing and color
            
        # Use HTML to center the headers. A block-level tag like <h3> is needed
        # for the alignment to apply across the widget's width.
        self._new_session_header = Label(value="<b>New Session</b>")
        
        
        # Create the separator using a styled Container
        self._separator = Container(labels=False)
        self._separator.native.setStyleSheet("""
            margin: 10px 0px;
            border-bottom: 1px solid #555;
            max-height: 1px;
            min-height: 1px;
        """)
        
        
        self._load_session_header = Label(value="<b>Load Previous Session</b>")

        # --- Alignment (now works because labels=False gives each widget full width) ---
        self._new_session_header.native.setAlignment(Qt.AlignCenter)
        self._load_session_header.native.setAlignment(Qt.AlignCenter)

        # --- "New Session" Sub-Container (uses default form layout with labels=True) ---
        self._new_session_container = Container()
        self._round1_path = FileEdit(label="Round 1", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2"))
        self._round2_path = FileEdit(label="Round 2", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2"))
        self._output_dir = FileEdit(label="Output Directory", mode="d", value=Path.home() / "zFISHer_Output")
        self._new_session_btn = PushButton(text="New Session")
        self._new_session_container.extend([
            self._round1_path,
            self._round2_path,
            self._output_dir,
            self._new_session_btn,
        ])

        # --- "Load Session" Sub-Container (uses default form layout with labels=True) ---
        self._load_session_container = Container()
        self._load_session_file = FileEdit(label="Session File (.json)", filter="*.json")
        self._load_session_btn = PushButton(text="Load Session")
        self._load_session_container.extend([
            self._load_session_file,
            self._load_session_btn,
        ])

        # --- Main Layout ---
        self.extend([
            self._new_session_header,
            self._new_session_container,
            self._separator,
            self._load_session_header,
            self._load_session_container,
        ])

        # --- Connections ---
        self._new_session_btn.clicked.connect(self._on_new_session)
        self._load_session_btn.clicked.connect(self._on_load_session)

    def _on_new_session(self):
        round1_path = self._round1_path.value
        round2_path = self._round2_path.value
        output_dir = self._output_dir.value

        session_file = output_dir / "zfisher_session.json"
        if session_file.exists():
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Session Already Exists",
                f"""A session already exists in this directory.

{output_dir}

Please choose a different output directory, or use the 'Load Session' button to continue your previous analysis."""
            )
            return

        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        (output_dir / "segmentation").mkdir(exist_ok=True)
        (output_dir / "aligned").mkdir(exist_ok=True)

        self._viewer.layers.clear()
        session.clear_session()
        session.update_data("output_dir", str(output_dir))
        session.update_data("r1_path", str(round1_path))
        session.update_data("r2_path", str(round2_path))
        session.save_session()

        with popups.ProgressDialog(self._viewer.window._qt_window, title="Loading Data...") as dialog:
            load_raw_data_into_viewer(
                self._viewer,
                round1_path,
                round2_path,
                output_dir=output_dir,
                progress_callback=lambda p, t: dialog.update_progress(p, t)
            )
            dialog.update_progress(100, "Done.")

        if hasattr(self._viewer.window, 'custom_scale_bar'):
            self._viewer.window.custom_scale_bar.show()
            self._viewer.window.custom_scale_bar.move_to_bottom_right()

    def _on_load_session(self):
        session_file = self._load_session_file.value
        if not session_file.exists() or session_file.is_dir():
            if session_file.is_dir():
                self._viewer.status = "Error: Please select a session file, not a directory."
            return

        dialog = popups.ProgressDialog(self._viewer.window._qt_window, "Loading Session...")
        session.set_loading(True)
        try:
            self._viewer.layers.clear()
            
            dialog.update_progress(10, "Loading session file...")
            session.load_session_file(session_file)
            
            shift = session.get_data("shift")
            if shift:
                print(f"Restored Shift: {shift}")

            r1_path = session.get_data("r1_path")
            r2_path = session.get_data("r2_path")
            if r1_path and r2_path:
                def progress_callback(p, text):
                    scaled_progress = 10 + int(p * 0.6) 
                    dialog.update_progress(scaled_progress, text)

                load_raw_data_into_viewer(
                    self._viewer, 
                    r1_path, 
                    r2_path,
                    progress_callback=progress_callback
                )

            scale = (1, 1, 1)
            for layer in self._viewer.layers:
                if isinstance(layer, napari.layers.Image):
                    scale = layer.scale
                    break

            processed_files = session.get_data("processed_files").items()
            num_files = len(processed_files)
            if num_files == 0:
                num_files = 1

            for i, (name, path_str) in enumerate(processed_files):
                progress = 70 + int(25 * (i + 1) / num_files)
                dialog.update_progress(progress, f"Loading: {name}")

                path = Path(path_str)
                if path.exists():
                    if path.suffix == '.npy':
                        data = np.load(path, allow_pickle=True)

                        if data.dtype.names:
                            coords = np.vstack(data['coord'])
                            properties = {name: data[name] for name in data.dtype.names if name != 'coord'}
                            
                            if "consensus_nuclei_masks_ids" in name.lower():
                                text_params = {
                                    'string': '{label}',
                                    'size': 10,
                                    'color': 'cyan',
                                    'translation': np.array([0, -5, 0])
                                }
                                self._viewer.add_points(
                                    coords,
                                    name=name,
                                    size=0,
                                    scale=scale,
                                    properties=properties,
                                    text=text_params,
                                    blending='translucent_no_depth'
                                )
                            else:
                                 self._viewer.add_points(
                                    coords,
                                    name=name,
                                    scale=scale,
                                    properties=properties
                                )

                        else:
                            if name == "Arrows":
                                vector_params = {
                                    'data': data,
                                    'name': name,
                                    'opacity': 1.0,
                                    'edge_width': 1,
                                    'length': 1,
                                    'edge_color': 'white',
                                }
                                if parse_version(napari.__version__) >= parse_version("0.7.0"):
                                    vector_params['head_width'] = 4
                                    vector_params['head_length'] = 6
                                
                                self._viewer.add_vectors(**vector_params)
                            elif "centroids" in name.lower():
                                self._viewer.add_points(data, name=name, size=5, face_color='orange', scale=scale)
                            else:
                                properties = {'id': np.arange(len(data)) + 1}
                                text_params = {
                                    'string': '{id}',
                                    'size': 8,
                                    'color': 'white',
                                    'translation': np.array([0, 5, 5]),
                                }
                                self._viewer.add_points(
                                    data, 
                                    name=name, 
                                    size=3, 
                                    face_color='yellow', 
                                    scale=scale,
                                    properties=properties,
                                    text=text_params
                                )
                    elif path.suffix in ['.tif', '.tiff']:
                        data = tifffile.imread(path)
                        if "masks" in name.lower():
                            self._viewer.add_labels(data, name=name, opacity=0.3, visible=False, scale=scale)
                        else:
                            c_map = 'gray'
                            for ch, color in CHANNEL_COLORS.items():
                                if ch.upper() in name.upper():
                                    c_map = color
                                    break
                            self._viewer.add_image(data, name=name, blending='additive', scale=scale, colormap=c_map)
                    print(f"Restored layer: {name}")
            
            dialog.update_progress(95, "Finalizing...")
            self._viewer.status = "Session Restored."

            if hasattr(self._viewer.window, 'custom_scale_bar'):
                self._viewer.window.custom_scale_bar.show()
                self._viewer.window.custom_scale_bar.move_to_bottom_right()
            
            dialog.update_progress(100, "Done.")

        finally:
            session.set_loading(False)
            dialog.close()
