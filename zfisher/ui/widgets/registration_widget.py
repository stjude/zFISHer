import napari
from magicgui import magicgui, widgets

import zfisher.core.session as session
from zfisher.core.registration import align_centroids_ransac
from .. import popups

@magicgui(
    call_button="Calculate Shift (RANSAC)",
    r1_points={"label": "R1 Centroids"},
    r2_points={"label": "R2 Centroids"}
)
def registration_widget(
    r1_points: "napari.layers.Points",
    r2_points: "napari.layers.Points"
):
    """Calculates the XYZ shift between two point clouds."""
    viewer = napari.current_viewer()
    
    # --- Session Check ---
    output_dir = session.get_data("output_dir")
    if not output_dir:
        popups.show_error_popup(
            viewer.window._qt_window,
            "No Active Session",
            "Please start or load a session before running registration."
        )
        return

    if r1_points is None or r2_points is None:
        viewer.status = "Please select both centroid layers."
        return

    p1 = r1_points.data
    p2 = r2_points.data
    
    viewer.status = "Running RANSAC..."
    
    with popups.ProgressDialog(viewer.window._qt_window, title="Calculating Registration (RANSAC)...") as dialog:
        
        def on_progress(value, text):
            dialog.update_progress(value, text)

        shift = align_centroids_ransac(p1, p2, progress_callback=on_progress)
        
        session.update_data("shift", shift.tolist())
        session.save_session()
        
        msg = f"Calculated Shift: Z={shift[0]:.2f}, Y={shift[1]:.2f}, X={shift[2]:.2f}"
        print(msg)
        viewer.status = msg
        
        registration_widget.result_label.value = f"<b>{msg}</b>"
        dialog.update_progress(100, "Done.")

registration_widget.result_label = widgets.Label(value="")
registration_widget.append(registration_widget.result_label)
