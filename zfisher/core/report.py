import logging
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from datetime import datetime

from .. import constants

logger = logging.getLogger(__name__)

def calculate_distances(points_layers_data):
    """
    Calculates nearest neighbor distances between all pairs of points layers.
    
    Parameters
    ----------
    points_layers_data : list[dict]
        A list of dictionaries, where each dictionary represents a points layer
        and contains 'name', 'data' (in pixels), and 'scale' keys.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the nearest neighbor analysis, with columns
        for source/target layers, IDs, distance, and coordinates.
    """
    results = []
    
    # Iterate all pairs (A -> B)
    for source in points_layers_data:
        for target in points_layers_data:
            if source['name'] == target['name']:
                continue
            
            # Get data and scale to microns
            # Napari data is (Z, Y, X)
            src_data = source['data'] * source['scale']
            tgt_data = target['data'] * target['scale']
            
            if len(src_data) == 0 or len(tgt_data) == 0:
                continue
                
            # Build KDTree for target
            tree = cKDTree(tgt_data)
            
            # Query nearest neighbors
            dists, idxs = tree.query(src_data)
            
            # Collect results
            for i, (d, idx) in enumerate(zip(dists, idxs)):
                results.append({
                    "Source_Layer": source['name'],
                    "Source_ID": i,
                    "Target_Layer": target['name'],
                    "Target_ID": idx,
                    "Distance_um": d,
                    "Z": src_data[i][0],
                    "Y": src_data[i][1],
                    "X": src_data[i][2]
                })
    
    return pd.DataFrame(results)

def export_report(df, save_path, r1_path=None, r2_path=None, output_dir=None, coloc_rules=None):
    """
    Exports a distances DataFrame to an Excel file with multiple sheets.

    The output file will contain sheets for:
    - 'Distances': The raw nearest-neighbor data.
    - 'Colocalization': A filtered view based on colocalization rules.
    - 'Metadata': Information about the analysis session.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame of distances from `calculate_distances`.
    save_path : Path
        The path where the Excel file will be saved.
    r1_path, r2_path, output_dir : str, optional
        Session metadata to include in the report.
    coloc_rules : list[dict], optional
        A list of colocalization rules to apply for the 'Colocalization' sheet.

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
        
        meta_list = [
            {"Key": "R1 File Name", "Value": r1_p.name},
            {"Key": "R2 File Name", "Value": r2_p.name},
            {"Key": "R1 File Path", "Value": str(r1_p)},
            {"Key": "R2 File Path", "Value": str(r2_p)},
            {"Key": "Analysis Date", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Key": "Output Folder", "Value": str(output_dir) if output_dir else "Not Set"},
            {"Key": "Session JSON Name", "Value": "zfisher_session.json"}
        ]
        
        # Handle Colocalization Rules
        df_coloc = pd.DataFrame()
        if coloc_rules:
            coloc_rows = []
            for rule in coloc_rules:
                src = rule['source']
                tgt = rule['target']
                thresh = rule['threshold']
                
                # Add to metadata
                meta_list.append({"Key": f"Coloc Rule: {src} -> {tgt}", "Value": f"<= {thresh} um"})
                
                # Filter Data
                mask = (df['Source_Layer'] == src) & \
                       (df['Target_Layer'] == tgt) & \
                       (df['Distance_um'] <= thresh)
                
                subset = df[mask].copy()
                subset['Coloc_Threshold_um'] = thresh
                coloc_rows.append(subset)
            
            if coloc_rows:
                df_coloc = pd.concat(coloc_rows)

        df_meta = pd.DataFrame(meta_list)
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=constants.DISTANCES_SHEET, index=False)
            if not df_coloc.empty:
                df_coloc.to_excel(writer, sheet_name=constants.COLOCALIZATION_SHEET, index=False)
            df_meta.to_excel(writer, sheet_name=constants.METADATA_SHEET, index=False)
            
        return save_path
        
    except (ImportError, ModuleNotFoundError):
        logger.warning("Excel export failed (missing openpyxl). Falling back to CSV.")
        save_path = save_path.with_suffix(".csv")
        df.to_csv(save_path, index=False)
        return save_path