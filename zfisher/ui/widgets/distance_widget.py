import napari
from pathlib import Path
from magicgui import magicgui

import zfisher.core.session as session
from .. import popups
from ..decorators import require_active_session, error_handler
from zfisher.core.report import calculate_distances, export_report

@magicgui(
    call_button="Calculate & Export Distances",
    output_filename={"label": "Filename (.xlsx)", "value": "puncta_distances.xlsx"}
)
@require_active_session("Please start or load a session before calculating distances.")
@error_handler("Distance Calculation Failed")
def distance_widget(output_filename: str = "puncta_distances.xlsx"):
    """Calculates nearest neighbor distances between all puncta layers."""
    viewer = napari.current_viewer()

    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    
    if len(points_layers) < 2:
        raise ValueError("Need at least 2 points layers to calculate distances.")

    with popups.ProgressDialog(viewer.window._qt_window, "Calculating Distances...") as dialog:
        viewer.status = "Calculating distances..."
        dialog.update_progress(10, "Extracting points data...")
        points_data = [{'name': l.name, 'data': l.data, 'scale': l.scale} for l in points_layers]

        dialog.update_progress(30, "Calculating nearest neighbors...")
        df = calculate_distances(points_data)
                    
        if df.empty:
            viewer.status = "No distances calculated."
            return
            
        dialog.update_progress(70, "Exporting report...")
        save_path = Path(session.get_data("output_dir", default=Path.home())) / output_filename
            
        final_path = export_report(
            df, 
            save_path, 
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=session.get_data("output_dir")
        )
        
        if session.get_data("output_dir"):
             session.set_processed_file("Distance_Report", str(final_path))
        
        popups.show_info_popup(
            viewer.window._qt_window,
            "Export Complete",
            f"Analysis exported successfully.\n\nFile: {final_path.name}\nPath: {final_path}"
        )
        viewer.status = f"Exported: {final_path.name}"
        dialog.update_progress(100, "Done.")
