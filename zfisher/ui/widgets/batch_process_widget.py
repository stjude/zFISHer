from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import logging
import napari
import pandas as pd
from qtpy.QtWidgets import QFrame, QFileDialog

from ...core import pipeline
from ...core.generate_batch_template import (
    build_template_sheets,
    add_dropdown_validations,
    parse_batch_config,
    validate_channels,
    resolve_puncta_for_dataset,
    resolve_coloc_for_dataset,
)
from .. import popups
from ..decorators import error_handler
from ..style import COLORS

logger = logging.getLogger(__name__)


class BatchProcessWidget(Container):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(labels=False)
        self._viewer = viewer

        self._init_widgets()
        self._init_layout()
        self._connect_signals()

    def _init_widgets(self):
        self._header = Label(value="Batch Process")
        self._header.native.setObjectName("widgetHeader")
        self._info = Label(
            value="<i>Run a complete zFISHer analysis on one or more datasets from a configuration template.</i>"
        )
        self._info.native.setObjectName("widgetInfo")
        self._info.native.setWordWrap(True)
        from qtpy.QtWidgets import QSizePolicy
        self._info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._batch_file = FileEdit(
            label="Batch File:",
            filter="*.xlsx *.xls",
            tooltip="Excel file (.xlsx) containing batch configuration. Use Generate Template to create one."
        )
        self._batch_output_dir = FileEdit(
            label="Output:",
            mode='d',
            value=Path.home() / "zFISHer_Batch_Output",
            tooltip="Base output directory. Each dataset gets its own subfolder."
        )
        self._generate_template_btn = PushButton(text="Generate Template")
        self._generate_template_btn.tooltip = "Save a pre-formatted Excel template with Datasets, Puncta, and Colocalization sheets."
        self._generate_template_btn.native.setMinimumWidth(0)
        self._batch_run_btn = PushButton(text="Run Batch Processing")
        self._batch_run_btn.tooltip = "Validate the batch file and run the full pipeline on all datasets."
        self._batch_run_btn.native.setMinimumWidth(0)

    @staticmethod
    def _make_divider():
        line = QFrame()
        line.setFixedHeight(2)
        line.setStyleSheet(f"background-color: {COLORS['separator_color']}; border: none; margin: 8px 0px;")
        return line

    def _init_layout(self):
        from qtpy.QtWidgets import QLabel, QSpacerItem, QSizePolicy

        self._batch_form = Container(labels=True)
        self._batch_form.extend([self._batch_file])
        self._batch_form.native.layout().setContentsMargins(0, 10, 0, 0)

        self._out_form = Container(labels=True)
        self._out_form.extend([self._batch_output_dir])
        self._out_form.native.layout().setContentsMargins(0, 10, 0, 20)

        self._desc = QLabel(
            "Use Generate Template to create a pre-formatted workbook. "
            "Fill out the configuration in the workbook with Datasets, Puncta, and Colocalization sheets. "
            "Load the completed workbook and run to automate processing of all datasets defined in the "
            "workbook according to the configuration specified in each sheet."
        )
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color: white; margin: 4px 2px;")

        spacer = lambda: QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Fixed)

        _layout = self.native.layout()
        _layout.setSpacing(2)
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.addWidget(self._header.native)
        _layout.addWidget(self._info.native)
        _layout.addWidget(self._make_divider())
        _layout.addWidget(self._desc)
        _layout.addWidget(self._batch_form.native)
        _layout.addWidget(self._out_form.native)
        _layout.addWidget(self._generate_template_btn.native)
        _layout.addSpacerItem(spacer())
        _layout.addWidget(self._batch_run_btn.native)
        _layout.addStretch(1)

    def _connect_signals(self):
        self._generate_template_btn.clicked.connect(self._on_generate_template)
        self._batch_run_btn.clicked.connect(self._on_batch_run)

    def _on_generate_template(self):
        """Save a multi-sheet batch template Excel file to a user-chosen location."""
        save_path, _ = QFileDialog.getSaveFileName(
            self.native,
            "Save Batch Template",
            str(Path.home() / "zFISHer_batch_template.xlsx"),
            "Excel Files (*.xlsx)",
        )
        if not save_path:
            return

        sheets = build_template_sheets()
        with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
            for name, df in sheets.items():
                df.to_excel(writer, sheet_name=name, index=False)
            add_dropdown_validations(writer.book)

            # Auto-size the Instructions columns for readability
            ws = writer.book["Instructions"]
            ws.column_dimensions['A'].width = 22
            ws.column_dimensions['B'].width = 110

        logger.info("Batch template saved to %s", save_path)
        popups.show_info_popup(
            self.native,
            "Template Saved",
            f"Batch template saved to:\n{save_path}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    @error_handler("Batch Processing Failed")
    def _on_batch_run(self):
        excel_path = Path(self._batch_file.value)
        output_base = Path(self._batch_output_dir.value)

        if not excel_path.is_file():
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "No Excel File",
                "Please select a valid batch Excel file (.xlsx)."
            )
            return

        if not output_base:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "No Output Directory",
                "Please select an output directory."
            )
            return

        # Parse and validate all sheets
        config, error = parse_batch_config(excel_path)
        if error:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Batch Validation Failed",
                error
            )
            return

        # Deep validation: check file channels match config
        channel_warnings = validate_channels(config)

        if channel_warnings:
            msg = (
                "The following issues were found during validation:\n\n"
                + "\n".join(f"  - {w}" for w in channel_warnings)
                + "\n\nDo you want to continue anyway?"
            )
            if not popups.show_yes_no_popup(
                self._viewer.window._qt_window,
                "Batch Validation Warnings",
                msg,
            ):
                return
        else:
            # No warnings — confirm start
            if not popups.show_yes_no_popup(
                self._viewer.window._qt_window,
                "Batch Validation Passed",
                f"All {len(config['datasets'])} dataset(s) validated successfully.\n\n"
                "Start batch processing?",
            ):
                return

        output_base.mkdir(parents=True, exist_ok=True)

        num_items = len(config["datasets"])
        results = []

        with popups.BatchProgressDialog(
            self._viewer.window._qt_window,
            title="Batch Processing",
            text="Preparing..."
        ) as dialog:
            for i, ds in enumerate(config["datasets"]):
                name = ds["name"]
                item_output = ds["output_dir"] or (output_base / name)

                batch_pct = int((i / num_items) * 100)
                dialog.update_batch_progress(
                    batch_pct,
                    f"Batch: {i + 1} / {num_items} — {name}"
                )
                dialog.update_item_progress(0, f"Starting {name}...")

                # Resolve per-dataset puncta and colocalization config
                puncta_config = resolve_puncta_for_dataset(name, config["puncta_rules"])
                pairwise_rules, tri_rules = resolve_coloc_for_dataset(name, config["coloc_rules"])

                try:
                    pipeline.run_full_zfisher_pipeline(
                        r1_path=ds["r1"],
                        r2_path=ds["r2"],
                        output_dir=item_output,
                        seg_method=ds["seg_method"],
                        merge_splits=ds["merge_splits"],
                        r1_nuclear_channel=ds["r1_nuclear_channel"],
                        r2_nuclear_channel=ds["r2_nuclear_channel"],
                        puncta_config=puncta_config,
                        pairwise_rules=pairwise_rules,
                        tri_rules=tri_rules,
                        progress_callback=dialog.update_item_progress,
                    )
                    results.append((name, "Success"))
                    logger.info("Batch item '%s' completed successfully.", name)
                except Exception as exc:
                    logger.error("Batch item '%s' failed: %s", name, exc, exc_info=True)
                    results.append((name, f"Failed: {exc}"))

            dialog.update_batch_progress(100, "Batch complete.")

        # Summary popup
        lines = [f"  {name}: {status}" for name, status in results]
        summary = "\n".join(lines)
        popups.show_info_popup(
            self._viewer.window._qt_window,
            "Batch Processing Complete",
            f"Processed {num_items} item(s):\n\n{summary}"
        )
