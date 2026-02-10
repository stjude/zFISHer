import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
from datetime import datetime

def calculate_distances(points_layers_data):
    """
    Calculates nearest neighbor distances between all pairs of points layers.
    
    Args:
        points_layers_data: List of dictionaries, each containing:
            - 'name': str
            - 'data': np.ndarray (N, 3) in pixel coordinates
            - 'scale': tuple (3,) scale factors
            
    Returns:
        pd.DataFrame: Results dataframe
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
    """Exports the dataframe to Excel with metadata."""
    save_path = Path(save_path)
    
    try:
        if not str(save_path).endswith(".xlsx"):
            save_path = save_path.with_suffix(".xlsx")
        
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
            df.to_excel(writer, sheet_name='Distances', index=False)
            if not df_coloc.empty:
                df_coloc.to_excel(writer, sheet_name='Colocalization', index=False)
            df_meta.to_excel(writer, sheet_name='Metadata', index=False)
            
        return save_path
        
    except (ImportError, ModuleNotFoundError):
        print("Excel export failed (missing openpyxl). Falling back to CSV.")
        save_path = save_path.with_suffix(".csv")
        df.to_csv(save_path, index=False)
        return save_path