"""Parsing a batch template into a ``params.BatchConfig``.

The template's sheets (Datasets, Puncta, Colocalization) are parsed into the
typed models of ``core.params``: an empty cell takes the setting's default, a
value outside its range is clamped to it with a warning, the ``ALL`` rows
apply to every dataset unless a dataset has rows of its own, and the CLI and
the Batch Process widget build the same pipeline call from a dataset.

Run:  python -m pytest tests/test_batch_template.py -q
"""
import numpy as np
import pandas as pd
import pytest

from zfisher.core import generate_batch_template as gbt
from zfisher.core import params

from synthetic_rounds import write_rounds


@pytest.fixture(scope="module")
def rounds(tmp_path_factory):
    root = tmp_path_factory.mktemp("rounds")
    write_rounds(root)
    return root


def _workbook(path, datasets, puncta=(), coloc=()):
    sheets = gbt.build_template_sheets()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheets["Instructions"].to_excel(writer, sheet_name="Instructions", index=False)
        pd.DataFrame(list(datasets), columns=sheets["Datasets"].columns).to_excel(writer, "Datasets", index=False)
        pd.DataFrame(list(puncta), columns=sheets["Puncta"].columns).to_excel(writer, "Puncta", index=False)
        pd.DataFrame(list(coloc), columns=sheets["Colocalization"].columns).to_excel(
            writer, "Colocalization", index=False)
    return path


def _dataset(rounds, name, **cells):
    row = {"Name": name, "R1": str(rounds / "R1.ome.tif"), "R2": str(rounds / "R2.ome.tif")}
    row.update(cells)
    return row


def test_empty_cells_take_the_model_defaults(rounds, tmp_path):
    book = _workbook(tmp_path / "b.xlsx", [_dataset(rounds, "ds1")], puncta=[{"Channel": "FITC"}])
    config, error = gbt.parse_batch_config(book)
    assert error is None and isinstance(config, params.BatchConfig)
    (ds,) = config.datasets
    assert ds.segmentation == params.SegmentationParams() and ds.registration == params.RegistrationParams()
    assert ds.consensus == params.ConsensusParams() and ds.r1_nuclear_channel == "DAPI"
    (rule,) = config.puncta_rules
    assert (rule.dataset, rule.round, rule.params) == ("ALL", "Both", params.DetectionParams())
    assert config.warnings == []


def test_values_outside_a_range_are_clamped_with_a_warning(rounds, tmp_path):
    book = _workbook(tmp_path / "b.xlsx", [_dataset(rounds, "ds1", Max_RANSAC_Distance=500, Match_Threshold=-3)],
                     puncta=[{"Channel": "FITC", "Sensitivity": 0.0, "Min_Distance": 99, "Sigma": 9}],
                     coloc=[{"Type": "pairwise", "Source": "FITC", "Target": "FITC", "Cutoff_um": 50}])
    config, error = gbt.parse_batch_config(book)
    assert error is None
    ds, rule, coloc = config.datasets[0], config.puncta_rules[0].params, config.coloc_rules[0].rule
    assert ds.registration.max_ransac_distance == 100 and ds.consensus.match_threshold == 0
    assert (rule.threshold_rel, rule.min_distance, rule.sigma) == (0.01, 20, 5.0)   # the GUI's ranges
    assert coloc.cutoff_um == 10.0
    text = "\n".join(config.warnings)
    assert "'Sensitivity' value 0.0 below minimum 0.01" in text and "Cutoff_um 50.0 outside range 0.1-10" in text


def test_errors_name_the_row(rounds, tmp_path):
    book = _workbook(tmp_path / "b.xlsx", [_dataset(rounds, "ds1", Seg_Method="Watershed"),
                                           _dataset(rounds, "ds1")],
                     coloc=[{"Type": "tri", "Source": "FITC", "Target": "FITC"}])
    config, error = gbt.parse_batch_config(book)
    assert config is None
    assert "Invalid Seg_Method 'Watershed'" in error and "Duplicate name 'ds1'" in error
    assert "Tri-colocalization requires Channel_B" in error


def test_all_rows_apply_unless_a_dataset_has_its_own(rounds, tmp_path):
    book = _workbook(
        tmp_path / "b.xlsx", [_dataset(rounds, "ds1"), _dataset(rounds, "ds2", Seg_Method="Cellpose")],
        puncta=[{"Channel": "FITC", "Sensitivity": 0.2},
                {"Dataset": "ds2", "Channel": "FITC", "Round": "R2", "Algorithm": "Radial Symmetry"}],
        coloc=[{"Type": "pairwise", "Source": "FITC", "Target": "DAPI", "Cutoff_um": 1.5},
               {"Dataset": "ds2", "Type": "tri", "Source": "FITC", "Target": "DAPI", "Channel_B": "FITC",
                "Cutoff_um": 2}])
    config, error = gbt.parse_batch_config(book)
    assert error is None
    assert gbt.resolve_puncta_for_dataset("ds1", config.puncta_rules) == {"FITC": {
        "method": "Local Maxima", "threshold_rel": 0.2, "min_distance": 2, "sigma": 0.0, "use_tophat": False,
        "tophat_radius": 15, "nuclei_only": True, "round": "Both"}}
    ds2 = gbt.resolve_puncta_for_dataset("ds2", config.puncta_rules)["FITC"]
    assert ds2["method"] == "Radial Symmetry" and ds2["round"] == "R2" and ds2["threshold_rel"] == 0.05
    assert gbt.resolve_coloc_for_dataset("ds1", config.coloc_rules) == (
        [{"source": "FITC", "target": "DAPI", "threshold": 1.5}], [])
    assert gbt.resolve_coloc_for_dataset("ds2", config.coloc_rules) == (
        [], [{"source": "FITC", "target": "DAPI", "channel_b": "FITC", "threshold": 2.0}])
    kwargs = gbt.pipeline_kwargs(config.datasets[1])
    assert kwargs["seg_method"] == "Cellpose" and kwargs["overlap_method"] == "Intersection"
    assert kwargs["r1_path"].name == "R1.ome.tif"


def test_channel_names_are_checked_against_the_files(rounds, tmp_path):
    book = _workbook(tmp_path / "b.xlsx", [_dataset(rounds, "ds1", R2_Nuclear_Channel="Hoechst")],
                     puncta=[{"Channel": "Cy5"}],
                     coloc=[{"Type": "tri", "Source": "FITC", "Target": "Cy3", "Channel_B": "DAPI"}])
    config, _ = gbt.parse_batch_config(book)
    text = "\n".join(gbt.validate_channels(config))
    assert "R2 nuclear channel 'Hoechst' not found" in text and "Puncta channel 'Cy5' not found" in text
    assert "Colocalization Target 'Cy3' not found" in text and "Channel_B" not in text
    assert np.all([w.startswith("[ds1]") for w in gbt.validate_channels(config)])


def test_the_cli_runs_a_batch_from_a_workbook(rounds, tmp_path):
    import logging
    from zfisher import cli
    from zfisher.core import session
    factory = logging.getLogRecordFactory()
    logging.setLogRecordFactory(logging.LogRecord)        # see synthetic_rounds.run_batch
    try:
        book = _workbook(tmp_path / "b.xlsx", [_dataset(rounds, "ds1")],
                         puncta=[{"Channel": "FITC", "Sensitivity": 0.05, "Min_Distance": 2}],
                         coloc=[{"Type": "pairwise", "Source": "FITC", "Target": "FITC", "Cutoff_um": 1.0}])
        assert cli.main(["batch", str(book), "-o", str(tmp_path / "out")]) == 0
    finally:
        logging.setLogRecordFactory(factory)
        session.clear_session()
    reports = tmp_path / "out" / "ds1" / "reports"
    assert list(reports.glob("*.xlsx")) and list(reports.glob("Aligned_R1_-_FITC_puncta.csv"))
