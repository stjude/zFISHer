import napari
from pathlib import Path
from magicgui import magicgui, widgets

import zfisher.core.session as session
from .. import popups
from ..decorators import require_active_session, error_handler
from zfisher.core.report import calculate_distances, export_report

@magicgui(
    call_button="Add Rule",
    source_layer={"label": "Source Layer"},
    target_layer={"label": "Target Layer"},
    cutoff={"label": "Cutoff (um)", "min": 0.1, "step": 0.1}
)
@require_active_session("Please start or load a session before adding rules.")
def colocalization_widget(
    source_layer: "napari.layers.Points", 
    target_layer: "napari.layers.Points", 
    cutoff: float = 1.0
):
    """Defines colocalization rules and exports report."""
    viewer = napari.current_viewer()

    if not source_layer or not target_layer:
        return
        
    rule = {
        'source': source_layer.name,
        'target': target_layer.name,
        'threshold': cutoff
    }
    
    if not hasattr(colocalization_widget, 'rules'):
        colocalization_widget.rules = []
        
    colocalization_widget.rules.append(rule)
    
    lines = [f"{r['source']} -> {r['target']} (<= {r['threshold']} um)" for r in colocalization_widget.rules]
    colocalization_widget.rules_display.value = "\n".join(lines)
    
    napari.current_viewer().status = f"Added rule: {lines[-1]}"

# Add extra UI elements
colocalization_widget.rules_display = widgets.Label(value="")
colocalization_widget.clear_btn = widgets.PushButton(text="Clear Rules")
colocalization_widget.export_btn = widgets.PushButton(text="Run Analysis & Export")
colocalization_widget.filename = widgets.LineEdit(label="Filename", value="coloc_analysis.xlsx")

colocalization_widget.append(widgets.Label(value="<b>Current Rules:</b>"))
colocalization_widget.append(colocalization_widget.rules_display)
colocalization_widget.append(colocalization_widget.clear_btn)
colocalization_widget.append(widgets.Label(value="<b>Export:</b>"))
colocalization_widget.append(colocalization_widget.filename)
colocalization_widget.append(colocalization_widget.export_btn)

@require_active_session("Please start or load a session before clearing rules.")
def _on_clear_rules():
    viewer = napari.current_viewer()

    if hasattr(colocalization_widget, 'rules'):
        colocalization_widget.rules = []
    colocalization_widget.rules_display.value = ""
    napari.current_viewer().status = "Rules cleared."

@require_active_session("Please start or load a session before exporting.")
@error_handler("Colocalization Export Failed")
def _on_coloc_export():
    viewer = napari.current_viewer()

    rules = getattr(colocalization_widget, 'rules', [])
    
    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    if len(points_layers) < 2:
        raise ValueError("Need at least 2 points layers for analysis.")

    with popups.ProgressDialog(viewer.window._qt_window, "Running Colocalization Analysis...") as dialog:
        viewer.status = "Calculating distances..."
        points_data = [{'name': l.name, 'data': l.data, 'scale': l.scale} for l in points_layers]
        df = calculate_distances(points_data)
        
        if df.empty:
            viewer.status = "No distances calculated."
            return

        fname = colocalization_widget.filename.value
        if not fname.endswith(".xlsx"): fname += ".xlsx"
        
        save_path = Path(session.get_data("output_dir", default=Path.home())) / fname
            
        final_path = export_report(
            df, 
            save_path, 
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=session.get_data("output_dir"),
            coloc_rules=rules
        )
        
        popups.show_info_popup(
            viewer.window._qt_window,
            "Export Complete",
            f"Analysis exported successfully.\n\nFile: {final_path.name}\nPath: {final_path}"
        )
        viewer.status = f"Exported: {final_path.name}"

colocalization_widget.clear_btn.clicked.connect(_on_clear_rules)
colocalization_widget.export_btn.clicked.connect(_on_coloc_export)
