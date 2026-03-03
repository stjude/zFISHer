from magicgui.widgets import Container, PushButton, FileEdit, Label
from pathlib import Path
import os
import logging
import napari
import pandas as pd

from ...core import pipeline
from .. import popups
from ..decorators import error_handler

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.nd2', '.tif', '.tiff'}


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
            value="<i>Run the full pipeline on multiple datasets from an Excel file.<br>"
                  "Excel columns: <b>Name</b>, <b>R1</b>, <b>R2</b></i>"
        )
        self._batch_file = FileEdit(
            label="Batch Excel File",
            filter="*.xlsx *.xls"
        )
        self._batch_output_dir = FileEdit(
            label="Output Directory",
            mode='d',
            value=Path.home() / "zFISHer_Batch_Output"
        )
        self._batch_run_btn = PushButton(text="Run Batch Processing")

    def _init_layout(self):
        self.extend([
            self._header,
            self._info,
            self._batch_file,
            self._batch_output_dir,
            self._batch_run_btn,
        ])

    def _connect_signals(self):
        self._batch_run_btn.clicked.connect(self._on_batch_run)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _read_and_validate_batch(self, excel_path):
        """
        Reads the batch Excel file and validates every row.

        Returns
        -------
        (DataFrame, None) on success, or (None, error_string) on failure.
        """
        try:
            df = pd.read_excel(excel_path)
        except Exception as exc:
            return None, f"Could not read Excel file:\n{exc}"

        # --- column check ---
        required = {"Name", "R1", "R2"}
        missing = required - set(df.columns)
        if missing:
            return None, f"Missing required columns: {', '.join(sorted(missing))}"

        if df.empty:
            return None, "The Excel file has no data rows."

        # --- per-row validation ---
        errors = []
        for idx, row in df.iterrows():
            row_num = idx + 2  # 1-indexed + header row
            name = str(row['Name']).strip()
            if not name:
                errors.append(f"Row {row_num}: Name is empty.")
                continue

            for col in ('R1', 'R2'):
                raw = str(row[col]).strip()
                if not raw:
                    errors.append(f"Row {row_num} ({name}): {col} path is empty.")
                    continue

                p = Path(raw)
                if not p.is_file():
                    errors.append(f"Row {row_num} ({name}): {col} file not found:\n  {p}")
                elif not os.access(p, os.R_OK):
                    errors.append(f"Row {row_num} ({name}): {col} file not readable:\n  {p}")
                elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    errors.append(
                        f"Row {row_num} ({name}): {col} unsupported file type "
                        f"({p.suffix}). Expected: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                    )

        if errors:
            return None, "\n".join(errors)

        return df, None

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

        # Validate all rows up-front
        df, error = self._read_and_validate_batch(excel_path)
        if error:
            popups.show_error_popup(
                self._viewer.window._qt_window,
                "Batch Validation Failed",
                error
            )
            return

        output_base.mkdir(parents=True, exist_ok=True)

        num_items = len(df)
        results = []

        with popups.BatchProgressDialog(
            self._viewer.window._qt_window,
            title="Batch Processing",
            text="Preparing..."
        ) as dialog:
            for i, (_, row) in enumerate(df.iterrows()):
                name = str(row['Name']).strip()
                r1 = Path(str(row['R1']).strip())
                r2 = Path(str(row['R2']).strip())
                item_output = output_base / name

                batch_pct = int((i / num_items) * 100)
                dialog.update_batch_progress(
                    batch_pct,
                    f"Batch: {i + 1} / {num_items} — {name}"
                )
                dialog.update_item_progress(0, f"Starting {name}...")

                try:
                    pipeline.run_full_zfisher_pipeline(
                        r1_path=r1,
                        r2_path=r2,
                        output_dir=item_output,
                        progress_callback=dialog.update_item_progress
                    )
                    results.append((name, "Success"))
                    logger.info("Batch item '%s' completed successfully.", name)
                except Exception as exc:
                    logger.error("Batch item '%s' failed: %s", name, exc)
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
