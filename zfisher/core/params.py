"""The settings a person or a batch template chooses, as typed models.

Each user-facing setting is stated once, here: its default, its range, its
choices, its label and its tooltip. The widgets build their controls from
these fields (``magicgui_options``), the batch template is parsed into them,
and the session stores their ``model_dump``. ``model_json_schema()`` is what
tool signatures for other front ends are generated from.

Algorithm internals that nobody sets (RANSAC trial counts, B-spline mesh size,
classical segmentation's blur) stay in ``constants``.

Two sets of defaults exist on purpose, for now: a ``DetectionParams`` field's
default is the value the GUI and the template start from, while
``puncta.DETECTOR_DEFAULTS`` is what detection uses for a setting it is not
given at all (a batch channel without a Puncta row). They differ for
sensitivity, min distance, sigma and the top-hat radius; making them one set
changes the puncta found for such channels, which is a decision for the lab.
"""
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
"""The version of the session and template formats these models describe.
Version 1 is the format zFISHer 2.x writes; a change that older zFISHer could
not read raises it."""


class DetectionMethod(str, Enum):
    LOCAL_MAXIMA = "Local Maxima"
    LOG = "Laplacian of Gaussian"
    DOG = "Difference of Gaussian"
    RADIAL_SYMMETRY = "Radial Symmetry"


class SegMethod(str, Enum):
    CLASSICAL = "Classical"
    CELLPOSE = "Cellpose"

    @classmethod
    def parse(cls, text):
        """The method named by ``text`` in any of its spellings: the template's
        ``Classical``/``Cellpose``, the widget's ``Classical (Fast)``/``Deep
        Learning (Cellpose)``, lower case. Raises ``ValueError`` otherwise."""
        if isinstance(text, cls):
            return text
        key = str(text).strip().lower()
        if "cellpose" in key or "deep learning" in key:
            return cls.CELLPOSE
        if key.startswith("classical"):
            return cls.CLASSICAL
        raise ValueError(f"Unknown segmentation method {text!r}; use Classical or Cellpose.")

    @property
    def display(self):
        """The label the segmentation widget shows."""
        return {SegMethod.CLASSICAL: "Classical (Fast)", SegMethod.CELLPOSE: "Deep Learning (Cellpose)"}[self]


class OverlapMethod(str, Enum):
    INTERSECTION = "Intersection"
    UNION = "Union"


def _ui(label, step=None, **extra):
    meta = {"label": label}
    if step is not None:
        meta["step"] = step
    meta.update(extra)
    return meta


class _Params(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True, use_enum_values=False)


class DetectionParams(_Params):
    """Settings of puncta detection on one channel."""
    method: DetectionMethod = Field(
        DetectionMethod.LOCAL_MAXIMA,
        description="Detection algorithm: Local Maxima (fastest, well-separated puncta), Laplacian of Gaussian "
                    "(blob-aware, crowded fields), Difference of Gaussian (fast blob approximation), Radial "
                    "Symmetry (high-density fields).",
        json_schema_extra=_ui("Algorithm"))
    threshold_rel: float = Field(
        0.05, ge=0.01, le=1.0, alias="sensitivity",
        description="Intensity threshold as a fraction (0-1) of the maximum. Lower values detect dimmer puncta "
                    "but may increase false positives.",
        json_schema_extra=_ui("Sensitivity (0-1)", step=0.01))
    min_distance: int = Field(
        2, ge=1, le=20,
        description="Minimum separation between detected puncta in pixels. Prevents double-counting nearby "
                    "puncta.",
        json_schema_extra=_ui("Min Distance (px)"))
    sigma: float = Field(
        0.0, ge=0.0, le=5.0,
        description="Gaussian sigma for detection in pixels. Match to the approximate radius of your puncta. "
                    "0 = no smoothing.",
        json_schema_extra=_ui("Spot Radius, px (Sigma)", step=0.1))
    z_scale: Optional[float] = Field(
        None, ge=0.01, le=20.0,
        description="Ratio of Z-step to XY pixel size. Corrects for non-cubic voxels. Auto-computed from image "
                    "metadata when available.",
        json_schema_extra=_ui("Z/XY Scale Ratio", step=0.01, gui_value=1.0))
    use_tophat: bool = Field(
        False, description="Apply white top-hat filter to subtract uneven background before detection.",
        json_schema_extra=_ui("Subtract Background (Top-hat)"))
    tophat_radius: int = Field(
        15, ge=1, le=50,
        description="Radius of the top-hat structuring element in pixels. Should be larger than puncta "
                    "diameter.",
        json_schema_extra=_ui("Top-hat Radius (px)"))
    nuclei_only: bool = Field(
        True, description="Keep only puncta inside the nuclei mask. Uncheck to detect puncta everywhere.",
        json_schema_extra=_ui("Nuclei Only (discard outside nuclei)"))

    def detection_kwargs(self):
        """The dict ``puncta.process_puncta_detection`` takes: method as its
        name, ``z_scale`` left out when not set (detection takes it from the
        voxel size)."""
        out = self.model_dump(mode="json")
        if out["z_scale"] is None:
            del out["z_scale"]
        return out


class SegmentationParams(_Params):
    """Settings of nuclei segmentation for one round."""
    method: SegMethod = Field(
        SegMethod.CLASSICAL,
        description="Classical: Fast, watershed-based. Cellpose: Deep learning, slower but more accurate.",
        json_schema_extra=_ui("Method"))
    merge_splits: bool = Field(
        True, description="Merge adjacent labels that one nucleus was split into.",
        json_schema_extra=_ui("Merge Split Nuclei"))


class RegistrationParams(_Params):
    """Settings of the R1-R2 registration."""
    max_ransac_distance: int = Field(
        0, ge=0, le=100,
        description="Maximum distance in micrometres for matching centroid pairs between rounds. 0 = auto-detect.",
        json_schema_extra=_ui("Max RANSAC Distance, µm (0=auto)"))
    apply_warp: bool = Field(
        True, description="Apply elastic warping for better alignment (recommended). Disable for rigid "
                          "rotation/translation only.",
        json_schema_extra=_ui("Apply B-spline Warp"))


class ConsensusParams(_Params):
    """Settings of the consensus nuclei mask."""
    overlap_method: OverlapMethod = Field(
        OverlapMethod.INTERSECTION,
        description="Intersection: Keep only overlapping pixels. Union: Keep all pixels from both rounds.",
        json_schema_extra=_ui("Overlap Method"))
    match_threshold: int = Field(
        0, ge=0, le=100,
        description="Maximum centroid distance in pixels to match nuclei between rounds. 0 = auto-detect.",
        json_schema_extra=_ui("Match Threshold, px (0=auto)"))
    remove_extranuclear_puncta: bool = Field(
        True, description="Remove puncta located outside the merged nuclei mask after matching.",
        json_schema_extra=_ui("Remove Extranuclear Puncta"))


class ColocRule(_Params):
    """A pairwise colocalization rule: target puncta within ``cutoff_um`` of source puncta."""
    type: Literal["pairwise"] = "pairwise"
    source: str
    target: str
    cutoff_um: float = Field(
        1.0, ge=0.1, le=10.0,
        description="Maximum distance (in microns) for two puncta to count as colocalized (close together).",
        json_schema_extra=_ui("Cutoff (um)", step=0.1))


class TriColocRule(_Params):
    """A three-channel rule: anchor puncta within ``cutoff_um`` of both channel A and channel B."""
    type: Literal["tri"] = "tri"
    anchor: str
    channel_a: str
    channel_b: str
    cutoff_um: float = Field(
        1.0, ge=0.1, le=10.0, description="Maximum distance in microns for tri-colocalization.",
        json_schema_extra=_ui("Cutoff (um)", step=0.1))


class QCThresholds(_Params):
    """When ``qc.flag_batch`` flags a dataset."""
    z_threshold: float = Field(3.5, gt=0, description="Robust standard deviations from the batch median "
                                                      "at which a metric is flagged.")
    min_cohort: int = Field(4, ge=2, description="Datasets needed before metrics are compared with the batch.")
    min_inlier_fraction: float = Field(0.5, ge=0, le=1, description="Least share of centroid pairs that "
                                                                    "must agree with the registration.")
    max_saturated_fraction: float = Field(0.01, ge=0, le=1, description="Largest share of saturated voxels "
                                                                        "in an aligned image.")


class DatasetSpec(_Params):
    """One row of the batch template's Datasets sheet."""
    name: str
    r1: Path
    r2: Path
    output_dir: Optional[Path] = None
    r1_nuclear_channel: str = "DAPI"
    r2_nuclear_channel: str = "DAPI"
    segmentation: SegmentationParams = SegmentationParams()
    registration: RegistrationParams = RegistrationParams()
    consensus: ConsensusParams = ConsensusParams()
    distribution_bins: Optional[list[int]] = Field(
        None, description="Lower edges of the Distribution sheet's puncta-count bins, starting at 0; "
                          "None for 0, 1, 2, 3 to 10 and more than 10.")


class PunctaRule(_Params):
    """One row of the Puncta sheet: detection settings for a channel, in every
    dataset (``dataset='ALL'``) or one, in both rounds or one."""
    dataset: str = "ALL"
    channel: str
    round: Literal["Both", "R1", "R2"] = "Both"
    params: DetectionParams = DetectionParams()


class ColocRuleRow(_Params):
    """One row of the Colocalization sheet, for every dataset or one."""
    dataset: str = "ALL"
    rule: Union[ColocRule, TriColocRule] = Field(discriminator="type")


class BatchConfig(_Params):
    """A parsed batch template."""
    datasets: list[DatasetSpec]
    puncta_rules: list[PunctaRule] = []
    coloc_rules: list[ColocRuleRow] = []
    qc: QCThresholds = QCThresholds()
    warnings: list[str] = []


def field_range(model, name):
    """``(ge, le)`` of a field of ``model``, None where unbounded."""
    ge = le = None
    for meta in model.model_fields[name].metadata:
        ge = getattr(meta, "ge", ge)
        le = getattr(meta, "le", le)
    return ge, le


def magicgui_options(model, names=None):
    """magicgui option dicts for the fields of ``model`` (all, or ``names``):
    value, min, max, step, label, tooltip and, for an enum, its choices, all
    read from the field, so a control cannot drift from the model."""
    out = {}
    for name, field in model.model_fields.items():
        if names is not None and name not in names:
            continue
        extra = dict(field.json_schema_extra or {})
        value = extra.pop("gui_value", field.default)
        if isinstance(value, Enum):
            value = value.value
        option = {"value": value, "tooltip": field.description or "", "label": extra.pop("label", name)}
        ge, le = field_range(model, name)
        if ge is not None:
            option["min"] = ge
        if le is not None:
            option["max"] = le
        if isinstance(field.default, Enum):
            option["choices"] = [member.value for member in type(field.default)]
        option.update(extra)
        out[name] = option
    return out


def gui_option(model, name, **override):
    """The magicgui option dict of one field (``magicgui_options``), with
    ``override`` on top for what only the widget decides (widget type,
    orientation, a shorter label)."""
    option = magicgui_options(model, [name])[name]
    option.update(override)
    return option


def default(model, name):
    """The GUI and template default of ``model``'s field ``name``."""
    field = model.model_fields[name]
    value = (field.json_schema_extra or {}).get("gui_value", field.default)
    return value.value if isinstance(value, Enum) else value


def schema():
    """The JSON schema of every model, by name: what tool signatures for other
    front ends are generated from, and what the snapshot test holds fixed."""
    models = (DetectionParams, SegmentationParams, RegistrationParams, ConsensusParams, ColocRule,
              TriColocRule, DatasetSpec, PunctaRule, ColocRuleRow, BatchConfig, QCThresholds)
    return {"schema_version": SCHEMA_VERSION, "models": {m.__name__: m.model_json_schema() for m in models}}
