
from magicgui import magicgui
from napari.layers import Points
import napari
from pathlib import Path

from ...core import session
from .. import popups
from ..decorators import require_active_session
from ...core.report import calculate_distances, export_report

@magicgui(
    call_button="Export",
    layout="vertical",
    source_layer={"label": "Source Layer"},
    target_layer={"label": "Target Layer"},
    threshold={"label": "Distance Threshold (um)"}
)
@require_active_session("Please start or load a session to enable exports.")
def export_widget(
    viewer: "napari.Viewer",
    source_layer: Points,
    target_layer: Points,
    threshold: float = 2.0,
):
    """
    A widget to calculate distances between points layers and export the results.
    """

    if source_layer is None or target_layer is None:
        viewer.status = "Please select both a source and target layer."
        return

    points_layers_data = [
        {
            "name": source_layer.name,
            "data": source_layer.data,
            "scale": source_layer.scale,
        },
        {
            "name": target_layer.name,
            "data": target_layer.data,
            "scale": target_layer.scale,
        },
    ]

    df = calculate_distances(points_layers_data)

    coloc_rules = [
        {
            "source": source_layer.name,
            "target": target_layer.name,
            "threshold": threshold,
        }
    ]

    reports_dir = Path(session.get_data("output_dir")) / "reports"
    reports_dir.mkdir(exist_ok=True)
    save_path = reports_dir / f"{source_layer.name}_{target_layer.name}_analysis.xlsx"

    export_report(df, save_path, coloc_rules=coloc_rules, output_dir=reports_dir)

    viewer.status = f"Report exported to {save_path}"
    print(f"Report exported to {save_path}")
