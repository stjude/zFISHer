from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import napari
import os
import gc
import tifffile

from qtpy.QtWidgets import QFrame

from ...core import session, io, segmentation, registration, puncta
from .. import popups, viewer_helpers
from ..decorators import error_handler
from ._shared import load_raw_data_into_viewer
from ..style import COLORS
from ... import constants

class NewSessionWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        """Initializes all UI widgets for the new session functionality."""
        self._header = Label(value="New Session")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(value="<i>Start a new zFISHer session from raw data.</i>")
        self._info.native.setObjectName("widgetInfo")
        self._round1_path = FileEdit(label="R1:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-19-24Fdecon.nd2"))
        self._round2_path = FileEdit(label="R2:", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/Desktop/ND2_FILE_INPUTS/1-17-24Adecon.nd2"))
     #   self._round1_path = FileEdit(label="Round 1", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R1_micro.tif"))
     #   self._round2_path = FileEdit(label="Round 2", filter="*.nd2 *.tif *.tiff *.ome.tif", value=Path("/Users/sstaller/zFISHer_MicroTests/R2_micro.tif"))

        self._output_dir = FileEdit(label="Output:", mode="d", value=Path.home() / "zFISHer_Output")
        self._new_session_btn = PushButton(text="Start New Session")
        self._autorun_btn = PushButton(text="Autorun")

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        """Arranges all widgets in the container using native layout."""
        # Form container with labels for the file inputs
        self._form = Container(labels=True)
        self._form.extend([self._round1_path, self._round2_path, self._output_dir])

        _layout = self.native.layout()
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._form.native)
        _layout.addWidget(self._new_session_btn.native)
        _layout.addWidget(self._autorun_btn.native)

    def _connect_signals(self):
        """Connects widget signals to their corresponding slots."""
        self._new_session_btn.clicked.connect(self._on_new_session)
        self._autorun_btn.clicked.connect(self._on_autorun)

    def _validate_input_files(self, r1_path, r2_path):
        """Checks if files exist, are files, and are readable."""
        error_messages = []
        for path, name in [(r1_path, "Round 1"), (r2_path, "Round 2")]:
            if not path.is_file():
                error_messages.append(f"• {name} file does not exist or is a directory:\n  {path}")
            elif not os.access(path, os.R_OK):
                error_messages.append(f"• {name} file is not readable (check permissions):\n  {path}")

        if error_messages:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Invalid Input Files",
                "Please correct the following issues:\n\n" + "\n\n".join(error_messages)
            )
            return False
        return True

    @error_handler("New Session Failed")
    def _on_new_session(self):
        r1_val = self._round1_path.value
        r2_val = self._round2_path.value
        out_val = self._output_dir.value

        if not self._validate_input_files(r1_val, r2_val):
            return

        with popups.ProgressDialog(self._viewer.window._qt_window, title="Initializing...") as dialog:
            success = session.initialize_new_session(out_val, r1_val, r2_val, progress_callback=dialog.update_progress)
            if not success:
                popups.show_error_popup(self._viewer.window._qt_window, "Session Exists", "A session already exists in the selected output directory. Please choose a different directory or load the existing session.")
                return

            self._viewer.layers.clear()
            load_raw_data_into_viewer(self._viewer, session.get_data("r1_path"), session.get_data("r2_path"), output_dir=session.get_data("output_dir"), progress_callback=dialog.update_progress)

    @error_handler("Autorun Failed")
    def _on_autorun(self):
        r1_val = self._round1_path.value
        r2_val = self._round2_path.value
        out_val = Path(self._output_dir.value)

        if not self._validate_input_files(r1_val, r2_val):
            return

        with popups.ProgressDialog(self._viewer.window._qt_window, title="Autorun Pipeline") as dialog:

            # --- 1. Session Init ---
            dialog.update_progress(2, "Initializing session...")
            if not session.initialize_new_session(out_val, r1_val, r2_val):
                popups.show_error_popup(
                    self._viewer.window._qt_window, "Session Exists",
                    "A session already exists in the selected output directory. Please choose a different directory."
                )
                return

            # --- 2. Load & Convert Raw Images ---
            dialog.update_progress(5, "Loading Round 1...")
            r1_sess = io.load_image_session(r1_val)
            dialog.update_progress(8, "Loading Round 2...")
            r2_sess = io.load_image_session(r2_val)

            input_dir = out_val / constants.INPUT_DIR
            if str(r1_val).lower().endswith('.nd2'):
                dialog.update_progress(10, "Converting R1 to OME-TIF...")
                io.convert_nd2_to_ome(r1_sess, input_dir, "R1", lambda msg: dialog.update_progress(11, msg))
            if str(r2_val).lower().endswith('.nd2'):
                dialog.update_progress(12, "Converting R2 to OME-TIF...")
                io.convert_nd2_to_ome(r2_sess, input_dir, "R2", lambda msg: dialog.update_progress(13, msg))

            # --- 3. DAPI Segmentation ---
            r1_dapi = io.get_channel_data(r1_sess, constants.DAPI_CHANNEL_NAME)
            r2_dapi = io.get_channel_data(r2_sess, constants.DAPI_CHANNEL_NAME)
            seg_results = segmentation.process_session_dapi(
                r1_data=r1_dapi, r2_data=r2_dapi, output_dir=out_val,
                progress_callback=lambda p, t: dialog.update_progress(15 + int(p * 0.2), t)
            )

            # --- 4. Registration ---
            shift, _ = registration.calculate_session_registration(
                seg_results['R1'][1], seg_results['R2'][1],
                voxels=r1_sess.voxels,
                progress_callback=lambda p, t: dialog.update_progress(35 + int(p * 0.1), t)
            )
            if shift is None:
                popups.show_error_popup(
                    self._viewer.window._qt_window, "Registration Failed",
                    "Could not calculate a valid shift. Check that both rounds have sufficient DAPI signal."
                )
                return

            # --- 5. Canvas Generation ---
            r1_layers = [{'name': f"R1 - {ch}", 'data': r1_sess.data[:, i, :, :], 'scale': r1_sess.voxels, 'is_label': False}
                         for i, ch in enumerate(r1_sess.channels)]
            r2_layers = [{'name': f"R2 - {ch}", 'data': r2_sess.data[:, i, :, :], 'scale': r2_sess.voxels, 'is_label': False}
                         for i, ch in enumerate(r2_sess.channels)]

            seg_dir = out_val / constants.SEGMENTATION_DIR
            for prefix, sess_obj, layer_list in [("R1", r1_sess, r1_layers), ("R2", r2_sess, r2_layers)]:
                mask_name = f"{prefix} - {constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}"
                mask_path = seg_dir / f"{mask_name}.tif"
                if mask_path.exists():
                    layer_list.append({'name': mask_name, 'data': tifffile.imread(mask_path), 'scale': sess_obj.voxels, 'is_label': True})

            aligned_dir = out_val / constants.ALIGNED_DIR
            registration.generate_global_canvas(
                r1_layers_data=r1_layers, r2_layers_data=r2_layers,
                shift=shift, output_dir=aligned_dir, apply_warp=True,
                progress_callback=lambda p, t: dialog.update_progress(45 + int(p * 0.15), t)
            )
            session.update_data("canvas_scale", r1_sess.voxels)

            # --- 6. Consensus Nuclei ---
            r1_aligned_mask = aligned_dir / f"Aligned_R1_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
            r2_warped_mask = aligned_dir / f"Warped_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"
            if not r2_warped_mask.exists():
                r2_warped_mask = aligned_dir / f"Aligned_R2_{constants.DAPI_CHANNEL_NAME}{constants.MASKS_SUFFIX}.tif"

            if not (r1_aligned_mask.exists() and r2_warped_mask.exists()):
                popups.show_error_popup(
                    self._viewer.window._qt_window, "Consensus Failed",
                    "Could not find aligned masks. Canvas generation may have failed."
                )
                return

            merged_mask, _ = segmentation.process_consensus_nuclei(
                mask1=tifffile.imread(r1_aligned_mask), mask2=tifffile.imread(r2_warped_mask),
                output_dir=out_val, threshold=20.0, method="Intersection",
                progress_callback=lambda p, t: dialog.update_progress(60 + int(p * 0.1), t)
            )

            # --- 7. Puncta Detection ---
            puncta_channels = [ch for ch in r1_sess.channels if ch.upper() != constants.DAPI_CHANNEL_NAME.upper()]
            puncta_params = {
                'threshold_rel': constants.PUNCTA_THRESHOLD_REL,
                'min_distance': constants.PUNCTA_MIN_DISTANCE,
                'method': "Local Maxima"
            }
            channel_jobs = [(rnd, pfx) for rnd, pfx in [("R1", "Aligned"), ("R2", "Warped")] for _ in puncta_channels]
            job_count = max(len(channel_jobs), 1)
            job_i = 0
            for rnd, prefix_str in [("R1", "Aligned"), ("R2", "Warped")]:
                for ch in puncta_channels:
                    ch_path = aligned_dir / f"{prefix_str}_{rnd}_{ch}.tif"
                    if not ch_path.exists():
                        ch_path = aligned_dir / f"Aligned_{rnd}_{ch}.tif"
                    if ch_path.exists():
                        job_base = 70 + int((job_i / job_count) * 25)
                        job_span = max(int(25 / job_count), 1)
                        dialog.update_progress(job_base, f"Detecting puncta: {rnd} {ch}...")
                        csv_out = seg_dir / f"{prefix_str}_{rnd}_{ch}{constants.PUNCTA_SUFFIX}.csv"
                        puncta_layer_name = f"{prefix_str} {rnd} - {ch}{constants.PUNCTA_SUFFIX}"
                        puncta.process_puncta_detection(
                            image_data=tifffile.imread(ch_path),
                            mask_data=merged_mask,
                            voxels=r1_sess.voxels,
                            params=puncta_params,
                            output_path=csv_out,
                            layer_name=puncta_layer_name,
                            progress_callback=lambda p, t, _b=job_base, _s=job_span: dialog.update_progress(
                                _b + int(p / 100 * _s), f"{rnd} {ch}: {t}"
                            )
                        )
                    job_i += 1

            # --- 8. Load Results into Viewer ---
            dialog.update_progress(95, "Loading results into viewer...")
            voxels = r1_sess.voxels
            del r1_sess, r2_sess, r1_dapi, r2_dapi, merged_mask
            gc.collect()

            self._viewer.layers.clear()

            def _stem_to_layer_name(stem):
                """Convert file stems like 'Aligned_R1_FITC' to layer names like 'Aligned R1 - FITC'."""
                parts = stem.split("_", 2)
                if len(parts) >= 3:
                    return f"{parts[0]} {parts[1]} - {parts[2]}"
                return stem

            for tif_path in sorted(aligned_dir.glob("*.tif")):
                name = _stem_to_layer_name(tif_path.stem)
                data = tifffile.imread(tif_path)
                if constants.MASKS_SUFFIX.lower() in name.lower():
                    lyr = self._viewer.add_labels(data, name=name, opacity=0.3, scale=voxels, visible=False)
                    lyr.rendering = 'iso_categorical'
                else:
                    cmap = next((c for k, c in constants.CHANNEL_COLORS.items() if k.upper() in name.upper()), 'gray')
                    self._viewer.add_image(data, name=name, blending='additive', colormap=cmap, scale=voxels,
                                           visible=constants.DAPI_CHANNEL_NAME.upper() in name.upper())

            # Deformation field (.npy vectors) from aligned_dir
            deform_path = aligned_dir / f"{constants.DEFORMATION_FIELD_NAME}.npy"
            if deform_path.exists():
                viewer_helpers._load_vectors_layer(self._viewer, constants.DEFORMATION_FIELD_NAME,
                                                   deform_path, voxels, {}, [0, 0, 0])

            consensus_path = seg_dir / f"{constants.CONSENSUS_MASKS_NAME}.tif"
            if consensus_path.exists():
                data = tifffile.imread(consensus_path)
                lyr = self._viewer.add_labels(data, name=constants.CONSENSUS_MASKS_NAME, opacity=0.5, scale=voxels, visible=False)
                lyr.rendering = 'iso_categorical'

            # Consensus nuclei ID points (.npy structured array)
            ids_path = seg_dir / f"{constants.CONSENSUS_MASKS_NAME}{constants.CONSENSUS_IDS_SUFFIX}.npy"
            if ids_path.exists():
                viewer_helpers._load_points_layer(self._viewer, f"{constants.CONSENSUS_MASKS_NAME}{constants.CONSENSUS_IDS_SUFFIX}",
                                                  ids_path, voxels, {'subtype': 'structured_ids'}, [0, 0, 0])

            for csv_path in sorted(seg_dir.glob(f"*{constants.PUNCTA_SUFFIX}.csv")):
                layer_name = _stem_to_layer_name(csv_path.stem)
                viewer_helpers._load_points_layer(self._viewer, layer_name, csv_path, voxels, {'subtype': 'puncta_csv'}, [0, 0, 0])

            self._viewer.dims.axis_labels = ("z", "y", "x")
            self._viewer.reset_view()
            dialog.update_progress(100, "Done.")

        popups.show_info_popup(
            self._viewer.window._qt_window, "Autorun Complete",
            f"Pipeline complete through puncta detection.\nResults saved to:\n{out_val}"
        )
