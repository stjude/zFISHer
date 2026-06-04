import napari
from magicgui import magicgui
from magicgui.widgets import Container, Label

from ...core import session, segmentation
from ._shared import make_divider as _make_divider, make_section_header as _make_section_header, make_section_desc as _make_section_desc, make_spacer as _make_spacer
from .. import popups, viewer_helpers
from ..decorators import require_active_session, error_handler
from ... import constants

@magicgui(
    call_button="Match & Merge Nuclei",
    layout="vertical",
    r1_mask_layer={"label": "R1 Mask (Aligned)", "tooltip": "The aligned nuclei mask from Round 1."},
    r2_mask_layer={"label": "R2 Mask (Warped)", "tooltip": "The nuclei mask from Round 2 after registration."},
    method={
        "label": "Overlap Method",
        "widget_type": "RadioButtons", # Changes dropdown to toggle/radio buttons
        "choices": ["Intersection", "Union"], # Intersection is now first
        "orientation": "horizontal", # Places buttons side-by-side
        "tooltip": "Intersection: Keep only overlapping pixels. Union: Keep all pixels from both rounds."
    },
    match_threshold={"label": "Match Threshold, px (0=auto)", "value": 0, "min": 0, "max": 100, "tooltip": "Maximum centroid distance in pixels to match nuclei between rounds. 0 = auto-detect."},
    remove_outliers={"label": "Remove Extranuclear Puncta", "tooltip": "Remove puncta located outside the merged nuclei mask."},
)
@require_active_session("Please start or load a session before matching nuclei.")
@error_handler("Nuclei Matching Failed")
def _nuclei_matching_widget(
    r1_mask_layer: "napari.layers.Labels",
    r2_mask_layer: "napari.layers.Labels",
    method: str = "Intersection",
    match_threshold: int = 0,
    remove_outliers: bool = True,
):
    """
    Matches nuclei between two aligned mask layers and syncs their IDs.
    Refactored to delegate math and saving to core.segmentation.
    """
    viewer = napari.current_viewer()

    # 1. Validation Logic
    if not r1_mask_layer or not r2_mask_layer:
        viewer.status = "Please select both mask layers."
        return
    
    if r1_mask_layer == r2_mask_layer:
        viewer.status = "Error: Same layer selected for both."
        return
        
    # 2. Execution with Progress Feedback
    with popups.ProgressDialog(viewer.window._qt_window, "Matching Nuclei...") as dialog:
        viewer.status = f"Matching nuclei ({method})..."
        
        output_dir = session.get_data("output_dir")

        # 3. Call the Core Orchestrator
        merged_mask, pts1 = segmentation.process_consensus_nuclei(
            mask1=r1_mask_layer.data,
            mask2=r2_mask_layer.data,
            output_dir=output_dir,
            threshold=match_threshold or None,
            method=method,
            progress_callback=lambda p, m: dialog.update_progress(p, m)
        )
        
        # 4. Update the Viewer
        viewer_helpers.add_consensus_nuclei_to_viewer(
            viewer,
            r1_mask_layer,
            merged_mask,
            pts1
        )

        # 5. Reassign nucleus IDs and optionally remove extranuclear puncta
        removed_total = 0
        if merged_mask is not None:
            dialog.update_progress(90, "Reassigning nuclei to puncta...")
            consensus_layer = next(
                (l for l in viewer.layers
                 if isinstance(l, napari.layers.Labels)
                 and constants.CONSENSUS_MASKS_NAME in l.name),
                None
            )
            if consensus_layer is not None:
                result = viewer_helpers.resync_puncta_nucleus_ids(
                    viewer, consensus_layer,
                    remove_extranuclear=remove_outliers, save_csv=True,
                )
                removed_total = result['removed_total']

        # Hide the input mask layers so only the consensus is visible
        r1_mask_layer.visible = False
        r2_mask_layer.visible = False

        outlier_msg = f" Removed {removed_total} extranuclear puncta." if removed_total > 0 else ""
        viewer.status = f"Matched {len(pts1) if pts1 else 0} nuclei using {method}.{outlier_msg}"
        dialog.update_progress(100, "Done.")

# --- UI Helpers ---
from qtpy.QtWidgets import QLabel, QFrame, QSizePolicy
from ..style import COLORS

# --- UI Wrapper ---
class _NucleiMatchingContainer(Container):
    def reset_choices(self):
        _nuclei_matching_widget.reset_choices()

nuclei_matching_widget = _NucleiMatchingContainer(labels=False)
nuclei_matching_widget._nuclei_matching_widget = _nuclei_matching_widget
header = Label(value="Match Nuclei")
header.native.setObjectName("widgetHeader")
info = Label(value="<i>Matches nuclei between rounds.</i>")
info.native.setObjectName("widgetInfo")
info.native.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

_layout = nuclei_matching_widget.native.layout()
_layout.setSpacing(2)
_layout.setContentsMargins(0, 0, 0, 0)
_layout.addWidget(header.native)
_layout.addWidget(info.native)
_layout.addWidget(_make_divider())

# Insert section headers into inner form
_inner = _nuclei_matching_widget.native.layout()
_masks_header = _make_section_header("Mask Layers")
_masks_desc = _make_section_desc("Select the aligned R1 and warped R2 nuclei mask layers.")
_inner.insertWidget(0, _masks_header)
_inner.insertWidget(1, _masks_desc)

_inner.insertWidget(4, _make_spacer())
_inner.insertWidget(5, _make_divider())
_method_header = _make_section_header("Method")
_method_desc = _make_section_desc("Choose how to combine overlapping nuclei between rounds.")
_inner.insertWidget(6, _method_header)
_inner.insertWidget(7, _method_desc)

_inner.insertWidget(_inner.count() - 1, _make_spacer())
_inner.setSpacing(2)
_inner.setContentsMargins(0, 0, 0, 0)

_nuclei_matching_widget.native.setMinimumWidth(0)
from qtpy.QtWidgets import QWidget as _QW
for child in _nuclei_matching_widget.native.findChildren(_QW):
    child.setMinimumWidth(0)

_layout.addWidget(_nuclei_matching_widget.native)
_layout.addStretch(1)