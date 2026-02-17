import napari
from pathlib import Path
from magicgui import magicgui, widgets

from ...core import session, analysis  # Delegates math to the core orchestrator
from .. import popups
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Add Colocalization Rule",
    layout="vertical",
    source_layer={"label": "Source Channel (e.g. FITC)"},
    target_layer={"label": "Target Channel (e.g. Cy5)"},
    cutoff={"label": "Cutoff (um)", "min": 0.1, "step": 0.1, "value": 1.0}
)
@require_active_session("Please start or load a session before adding rules.")
def colocalization_widget(
    source_layer: "napari.layers.Points", 
    target_layer: "napari.layers.Points", 
    cutoff: float = 1.0
):
    """
    Unified Analysis Hub: Manages biological rules and master report exports.
    """
    viewer = napari.current_viewer()

    if not source_layer or not target_layer:
        return
        
    rule = {
        'source': source_layer.name,
        'target': target_layer.name,
        'threshold': cutoff
    }
    
    # Initialize rules list if it doesn't exist
    if not hasattr(colocalization_widget, 'rules'):
        colocalization_widget.rules = []
        
    colocalization_widget.rules.append(rule)
    
    # Update the UI display for the user
    lines = [f"{r['source']} -> {r['target']} (<= {r['threshold']} um)" for r in colocalization_widget.rules]
    colocalization_widget.rules_display.value = "\n".join(lines)
    viewer.status = f"Added rule: {lines[-1]}"

# --- Consolidated UI Hub Elements ---
colocalization_widget.rules_display = widgets.Label(value="")
colocalization_widget.clear_btn = widgets.PushButton(text="Clear Rules")
colocalization_widget.filename = widgets.LineEdit(label="Report Name", value="zFISHer_Analysis_Report.xlsx")
colocalization_widget.export_btn = widgets.PushButton(text="Run Analysis & Export Master Report")

colocalization_widget.extend([
    widgets.Label(value="<b>Biological Rules:</b>"),
    colocalization_widget.rules_display,
    colocalization_widget.clear_btn,
    widgets.Label(value="<br><b>Export Settings:</b>"),
    colocalization_widget.filename,
    colocalization_widget.export_btn
])

@require_active_session()
def _on_clear_rules():
    """Resets the rule list."""
    colocalization_widget.rules = []
    colocalization_widget.rules_display.value = ""
    napari.current_viewer().status = "All analysis rules cleared."

@require_active_session("Please start or load a session before exporting.")
@error_handler("Analysis Export Failed")
def _on_coloc_export():
    """
    Triggers the core analysis orchestrator.
    This replaces the old distance_widget and export_widget logic.
    """
    viewer = napari.current_viewer()
    rules = getattr(colocalization_widget, 'rules', [])
    
    # Validation: Ensure we have enough data to compare
    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    if len(points_layers) < 2:
        raise ValueError("At least two puncta layers are required for nearest-neighbor analysis.")

    with popups.ProgressDialog(viewer.window._qt_window, "Generating Master Report...") as dialog:
        # 1. Prepare data for the Core
        points_data = [{'name': l.name, 'data': l.data, 'scale': l.scale} for l in points_layers]
        
        # 2. Get session output path
        out_dir = Path(session.get_data("output_dir")) / constants.REPORTS_DIR
        out_dir.mkdir(exist_ok=True, parents=True)
        
        # 3. Call the Core Analysis Orchestrator
        # This function handles the cKDTree math and Excel formatting.
        final_path = analysis.run_colocalization_analysis(
            layers_data=points_data,
            rules=rules,
            filename=colocalization_widget.filename.value,
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=out_dir
        )
        
        # 4. User Feedback
        popups.show_info_popup(
            viewer.window._qt_window,
            "Export Complete",
            f"Master Report successfully saved to:\n\n{final_path.name}"
        )
        viewer.status = f"Analysis saved to {final_path.name}"

# Bind the consolidated events
colocalization_widget.clear_btn.clicked.connect(_on_clear_rules)
colocalization_widget.export_btn.clicked.connect(_on_coloc_export)