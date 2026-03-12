import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from .. import constants
from . import session

logger = logging.getLogger(__name__)


def export_report(df, save_path, r1_path=None, r2_path=None, output_dir=None, coloc_df=None, coloc_meta=None, tri_coloc_df=None, tri_coloc_meta=None, per_nucleus_df=None, stats_df=None, distribution_df=None):
    """
    Exports pre-computed analysis DataFrames to an Excel file with multiple sheets.

    The output file will contain sheets for:
    - 'Distances': The raw nearest-neighbor data.
    - 'Colocalization': Pre-computed pairwise colocalization hits (if any).
    - 'Tri-Colocalization': Pre-computed tri-coloc hits (if any).
    - 'Metadata': Information about the analysis session.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame of distances from `calculate_distances`.
    save_path : Path
        The path where the Excel file will be saved.
    r1_path, r2_path, output_dir : str, optional
        Session metadata to include in the report.
    coloc_df : pd.DataFrame, optional
        Pre-computed pairwise colocalization hits from `calculate_pairwise_colocalization`.
    coloc_meta : list[dict], optional
        Metadata entries from `calculate_pairwise_colocalization`.
    tri_coloc_df : pd.DataFrame, optional
        Pre-computed tri-colocalization hits from `calculate_tri_colocalization`.
    tri_coloc_meta : list[dict], optional
        Metadata entries from `calculate_tri_colocalization`.

    Returns
    -------
    Path
        The final path of the saved report file.
    """
    save_path = Path(save_path)

    try:
        if not str(save_path).endswith(constants.EXCEL_SUFFIX):
            save_path = save_path.with_suffix(constants.EXCEL_SUFFIX)

        # Prepare Metadata
        r1_p = Path(r1_path) if r1_path else Path("Not Set")
        r2_p = Path(r2_path) if r2_path else Path("Not Set")
        session_filename = session.get_data("session_filename", constants.SESSION_FILENAME)

        meta_list = [
            {"Key": "R1 File Name", "Value": r1_p.name},
            {"Key": "R2 File Name", "Value": r2_p.name},
            {"Key": "R1 File Path", "Value": str(r1_p)},
            {"Key": "R2 File Path", "Value": str(r2_p)},
            {"Key": "Analysis Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Key": "Output Folder", "Value": str(output_dir) if output_dir else "Not Set"},
            {"Key": "Session JSON Name", "Value": session_filename}
        ]

        if coloc_meta:
            meta_list.extend(coloc_meta)
        if tri_coloc_meta:
            meta_list.extend(tri_coloc_meta)

        df_meta = pd.DataFrame(meta_list)

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=constants.DISTANCES_SHEET, index=False)
            if coloc_df is not None and not coloc_df.empty:
                coloc_df.to_excel(writer, sheet_name=constants.COLOCALIZATION_SHEET, index=False)
            if tri_coloc_df is not None and not tri_coloc_df.empty:
                tri_coloc_df.to_excel(writer, sheet_name=constants.TRI_COLOCALIZATION_SHEET, index=False)
            if per_nucleus_df is not None and not per_nucleus_df.empty:
                per_nucleus_df.to_excel(writer, sheet_name=constants.PER_NUCLEI_SHEET, index=False)
            if stats_df is not None and not stats_df.empty:
                stats_df.to_excel(writer, sheet_name=constants.STATS_SHEET, index=False)
            if distribution_df is not None and not distribution_df.empty:
                distribution_df.to_excel(writer, sheet_name=constants.DISTRIBUTION_SHEET, index=False)
            df_meta.to_excel(writer, sheet_name=constants.METADATA_SHEET, index=False)

        return save_path

    except (ImportError, ModuleNotFoundError):
        logger.warning("Excel export failed (missing openpyxl). Falling back to CSV.")
        save_path = save_path.with_suffix(".csv")
        df.to_csv(save_path, index=False)
        return save_path
