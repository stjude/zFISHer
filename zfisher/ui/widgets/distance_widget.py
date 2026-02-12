import napari
from pathlib import Path
from magicgui import magicgui

import zfisher.core.session as session
from zfisher.core.report import calculate_distances, export_report
from .. import popups

@magicgui(
    call_button="Calculate & Export Distances",
    output_filename={"label": "Filename (.xlsx)", "value": "puncta_distances.xlsx"}
)
def distance_widget(output_filename: str = "puncta_distances.xlsx"):
    """Calculates nearest neighbor distances between all puncta layers."""
    viewer = napari.current_viewer()
    
    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    
    if len(points_layers) < 2:
        viewer.status = "Need at least 2 points layers."
        print("Error: Not enough points layers found.")
        return

    viewer.status = "Calculating distances..."
    dialog = popups.show_busy_popup(viewer.window._qt_window, "Calculating Distances...")
    
    try:
        points_data = [{'name': l.name, 'data': l.data, 'scale': l.scale} for l in points_layers]
        df = calculate_distances(points_data)
                    
        if df.empty:
            viewer.status = "No distances calculated."
            return
            
        save_path = Path(session.get_data("output_dir", Path.home())) / output_filename
            
        final_path = export_report(
            df, 
            save_path, 
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=session.get_data("output_dir")
        )
        
        print(f"Saved distances to {final_path}")
        viewer.status = f"Exported: {final_path.name}"
        
        if session.get_data("output_dir"):
             session.set_processed_file("Distance_Report", str(final_path))
             session.save_session()
        
        popups.show_info_popup(
            viewer.window._qt_window,
            "Export Complete",
            f"Analysis exported successfully.\n\nFile: {final_path.name}\nPath: {final_path}"
        )
                 
    except Exception as e:
        print(f"Export failed: {e}")
        viewer.status = "Export failed (check console)."
        popups.show_error_popup(
            viewer.window._qt_window,
            "Export Failed",
            f"An error occurred during export.\n\nError: {e}"
        )
    finally:
        dialog.close()
