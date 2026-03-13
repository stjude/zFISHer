import logging
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui, widgets
from magicgui.widgets import Container, Label, PushButton, LineEdit

from qtpy.QtWidgets import QFrame

from ...core import session, analysis
from .. import popups
from ..decorators import require_active_session, error_handler
from ... import constants

logger = logging.getLogger(__name__)


def _make_divider():
    """Create a horizontal line divider using a native Qt QFrame."""
    line = QFrame()
    line.setFixedHeight(2)
    line.setStyleSheet("background-color: #7a6b8a; border: none; margin: 8px 0px;")
    return line


# =====================================================================
# Pairwise Colocalization
# =====================================================================

@magicgui(
    call_button="Add Colocalization Rule",
    layout="vertical",
    source_layer={"label": "Source Channel (e.g. FITC)"},
    target_layer={"label": "Target Channel (e.g. Cy5)"},
    cutoff={"label": "Cutoff (um)", "min": 0.1, "step": 0.1, "value": 1.0}
)
@require_active_session("Please start or load a session before adding rules.")
def _rule_builder(
    source_layer: "napari.layers.Points",
    target_layer: "napari.layers.Points",
    cutoff: float = 1.0
):
    """Adds a colocalization rule to the current session."""
    viewer = napari.current_viewer()

    if not source_layer or not target_layer:
        return

    rule = {
        'source': source_layer.name,
        'target': target_layer.name,
        'threshold': cutoff
    }

    rules = session.get_data("colocalization_rules", default=[])
    rules.append(rule)
    session.update_data("colocalization_rules", rules)

    logger.info("Added pairwise rule: %s -> %s (<= %s um)", rule['source'], rule['target'], rule['threshold'])
    _update_rules_display(rules)
    viewer.status = f"Added rule: {rule['source']} -> {rule['target']} (<= {rule['threshold']} um)"


_rules_display = Label(value="No rules defined.")
_clear_btn = PushButton(text="Clear Rules")


def _update_rules_display(rules):
    """Formats and updates the rules display label."""
    if rules:
        lines = [f"{r['source']}  \u2192  {r['target']}  (\u2264 {r['threshold']} \u00b5m)" for r in rules]
        _rules_display.value = "\n".join(lines)
    else:
        _rules_display.value = "No rules defined."


@require_active_session()
def _on_clear_rules():
    """Resets the rule list."""
    logger.info("Cleared all pairwise colocalization rules")
    session.update_data("colocalization_rules", [])
    _update_rules_display([])
    napari.current_viewer().status = "All analysis rules cleared."


_clear_btn.clicked.connect(_on_clear_rules)


# =====================================================================
# Tri-Colocalization
# =====================================================================

@magicgui(
    call_button="Add Tri-Colocalization Rule",
    layout="vertical",
    anchor_layer={"label": "Anchor Channel"},
    channel_a_layer={"label": "Channel A"},
    channel_b_layer={"label": "Channel B"},
    cutoff={"label": "Cutoff (um)", "min": 0.1, "step": 0.1, "value": 1.0}
)
@require_active_session("Please start or load a session before adding rules.")
def _tri_rule_builder(
    anchor_layer: "napari.layers.Points",
    channel_a_layer: "napari.layers.Points",
    channel_b_layer: "napari.layers.Points",
    cutoff: float = 1.0
):
    """Adds a tri-colocalization rule: anchor must be near both Channel A and Channel B."""
    viewer = napari.current_viewer()

    if not anchor_layer or not channel_a_layer or not channel_b_layer:
        return

    rule = {
        'anchor': anchor_layer.name,
        'channel_a': channel_a_layer.name,
        'channel_b': channel_b_layer.name,
        'threshold': cutoff
    }

    tri_rules = session.get_data("tri_colocalization_rules", default=[])
    tri_rules.append(rule)
    session.update_data("tri_colocalization_rules", tri_rules)

    logger.info("Added tri-coloc rule: %s + %s + %s (<= %s um)", rule['anchor'], rule['channel_a'], rule['channel_b'], rule['threshold'])
    _update_tri_rules_display(tri_rules)
    viewer.status = f"Added tri-coloc rule: {rule['anchor']} + {rule['channel_a']} + {rule['channel_b']} (<= {rule['threshold']} um)"


_tri_rules_display = Label(value="No tri-coloc rules defined.")
_tri_clear_btn = PushButton(text="Clear Tri-Coloc Rules")


def _update_tri_rules_display(tri_rules):
    """Formats and updates the tri-colocalization rules display."""
    if tri_rules:
        lines = [
            f"{r['anchor']}  \u2194  {r['channel_a']} + {r['channel_b']}  (\u2264 {r['threshold']} \u00b5m)"
            for r in tri_rules
        ]
        _tri_rules_display.value = "\n".join(lines)
    else:
        _tri_rules_display.value = "No tri-coloc rules defined."


@require_active_session()
def _on_clear_tri_rules():
    """Resets the tri-colocalization rule list."""
    logger.info("Cleared all tri-colocalization rules")
    session.update_data("tri_colocalization_rules", [])
    _update_tri_rules_display([])
    napari.current_viewer().status = "All tri-colocalization rules cleared."


_tri_clear_btn.clicked.connect(_on_clear_tri_rules)


# =====================================================================
# Export
# =====================================================================

_filename = LineEdit(label="Report Name", value="zFISHer_Analysis_Report.xlsx")
_export_btn = PushButton(text="Run Analysis & Export Master Report")


@require_active_session("Please start or load a session before exporting.")
@error_handler("Analysis Export Failed")
def _on_export():
    """Triggers the core analysis orchestrator."""
    viewer = napari.current_viewer()
    rules = session.get_data("colocalization_rules", default=[])
    tri_rules = session.get_data("tri_colocalization_rules", default=[])

    points_layers = [l for l in viewer.layers if isinstance(l, napari.layers.Points)]
    if len(points_layers) < 2:
        raise ValueError("At least two puncta layers are required for nearest-neighbor analysis.")

    with popups.ProgressDialog(viewer.window._qt_window, "Generating Master Report...") as dialog:
        points_data = []
        for l in points_layers:
            d = {'name': l.name, 'data': l.data, 'scale': l.scale, 'translate': l.translate}
            if hasattr(l, 'features') and 'Nucleus_ID' in l.features.columns:
                d['nucleus_ids'] = l.features['Nucleus_ID'].values
            points_data.append(d)

        # Count actual nuclei from the consensus mask layer
        total_nuclei = None
        for l in viewer.layers:
            if isinstance(l, napari.layers.Labels) and constants.CONSENSUS_MASKS_NAME in l.name:
                unique_ids = np.unique(l.data)
                total_nuclei = int((unique_ids > 0).sum())
                break

        out_dir = Path(session.get_data("output_dir")) / constants.REPORTS_DIR
        out_dir.mkdir(exist_ok=True, parents=True)

        final_path = analysis.run_colocalization_analysis(
            layers_data=points_data,
            rules=rules,
            tri_rules=tri_rules,
            filename=_filename.value,
            r1_path=session.get_data("r1_path"),
            r2_path=session.get_data("r2_path"),
            output_dir=out_dir,
            total_nuclei=total_nuclei
        )

        popups.show_info_popup(
            viewer.window._qt_window,
            "Export Complete",
            f"Master Report successfully saved to:\n\n{final_path.name}"
        )
        viewer.status = f"Analysis saved to {final_path.name}"


_export_btn.clicked.connect(_on_export)


# =====================================================================
# Public API
# =====================================================================

def refresh_rules_display():
    """Restores all rules displays from session data (e.g., after loading a session)."""
    rules = session.get_data("colocalization_rules", default=[])
    _update_rules_display(rules)
    tri_rules = session.get_data("tri_colocalization_rules", default=[])
    _update_tri_rules_display(tri_rules)


# =====================================================================
# UI Wrapper
# =====================================================================

def _filter_id_layers(widget):
    """Remove _IDs points layers from all dropdown choices in a magicgui widget."""
    for name, param_widget in widget.__signature__.parameters.items():
        try:
            combo = getattr(widget, name)
            if hasattr(combo, 'choices'):
                combo.choices = [c for c in combo.choices if not c.name.endswith("_IDs")]
        except (AttributeError, TypeError):
            pass


class _ColocalizationContainer(Container):
    """Wrapper that delegates reset_choices and exposes the inner magicgui."""
    def reset_choices(self):
        _rule_builder.reset_choices()
        _filter_id_layers(_rule_builder)
        _tri_rule_builder.reset_choices()
        _filter_id_layers(_tri_rule_builder)

colocalization_widget = _ColocalizationContainer(labels=False)
colocalization_widget._rule_builder = _rule_builder
header = Label(value="Colocalization Analysis")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Define distance rules between puncta channels, then export a report.</i>")
info.native.setObjectName("widgetInfo")

# Build layout using native layout to keep dividers in correct order
_pairwise_header = Label(value="<b>Pairwise Colocalization:</b>")
_active_rules_header = Label(value="<b>Active Rules:</b>")
_tri_header = Label(value="<b>Tri-Colocalization:</b>")
_tri_active_header = Label(value="<b>Active Tri-Coloc Rules:</b>")
_export_header = Label(value="<b>Export:</b>")

_layout = colocalization_widget.native.layout()
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())
# --- Pairwise Colocalization ---
_layout.addWidget(_pairwise_header.native)
_layout.addWidget(_rule_builder.native)
_layout.addWidget(_active_rules_header.native)
_layout.addWidget(_rules_display.native)
_layout.addWidget(_clear_btn.native)
# --- Divider ---
_layout.addWidget(_make_divider())
# --- Tri-Colocalization ---
_layout.addWidget(_tri_header.native)
_layout.addWidget(_tri_rule_builder.native)
_layout.addWidget(_tri_active_header.native)
_layout.addWidget(_tri_rules_display.native)
_layout.addWidget(_tri_clear_btn.native)
# --- Divider ---
_layout.addWidget(_make_divider())
# --- Export ---
_layout.addWidget(_export_header.native)
_layout.addWidget(_filename.native)
_layout.addWidget(_export_btn.native)
