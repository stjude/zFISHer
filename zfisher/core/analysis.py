# zfisher/core/analysis.py
import pandas as pd
from pathlib import Path
from .report import calculate_distances, export_report
from . import session
from .. import constants

def run_colocalization_analysis(layers_data, rules, filename, r1_path, r2_path, output_dir):
    """
    Core Orchestrator for Step 7 & 8.
    Processes puncta distances and exports the master report.
    """
    # 1. Math: Calculate Nearest Neighbors via cKDTree
    # 'layers_data' should be a list of dicts: [{'name': str, 'data': array, 'scale': tuple}]
    df = calculate_distances(layers_data)
    
    if df.empty:
        print("Warning: No puncta found to analyze.")
        return None

    # 2. File I/O: Generate Multi-Sheet Excel Report
    save_path = Path(output_dir) / filename
    final_path = export_report(
        df, 
        save_path, 
        r1_path=r1_path,
        r2_path=r2_path,
        output_dir=output_dir,
        coloc_rules=rules
    )

    # 3. Session Tracking (Crucial for Headless)
    if final_path.exists():
        session.set_processed_file(
            layer_name="Master_Analysis_Report",
            path=str(final_path),
            layer_type="report"
        )
    
    return final_path