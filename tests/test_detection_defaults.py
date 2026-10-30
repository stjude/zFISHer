"""What is recorded for a detection is what the detection used.

A setting missing from the parameters (a channel with no row on the batch
template's Puncta sheet, the CLI's ``run``) was detected with the detectors'
own fallback, for sigma 1.0, while the report's Parameters sheet and the
provenance record filled in the GUI's starting values, sigma 0.0. Detection
now fills in its fallbacks first, and records those.

Run:  python -m pytest tests/test_detection_defaults.py -q
"""
import numpy as np

from zfisher.core import provenance, puncta, stages
from zfisher import constants

from test_stages import CHANNEL, finished, reopened  # noqa: F401  (fixtures)


def _volume():
    rng = np.random.default_rng(3)
    image = rng.normal(100, 5, (6, 40, 40))
    for z, y, x in [(2, 10, 10), (3, 25, 30), (4, 30, 8)]:
        image[z, y - 1:y + 2, x - 1:x + 2] += 400
    return image


def test_missing_settings_detect_as_the_filled_in_ones():
    image = _volume()
    partial = {"method": "Local Maxima", "threshold_rel": 0.3}
    filled = puncta.fill_detection_defaults(dict(partial), voxels=(0.3, 0.1, 0.1))
    a = puncta.process_puncta_detection(image, voxels=(0.3, 0.1, 0.1), params=dict(partial))
    b = puncta.process_puncta_detection(image, voxels=(0.3, 0.1, 0.1), params=filled)
    np.testing.assert_array_equal(a, b)
    assert filled["sigma"] == 1.0 and filled["min_distance"] == 3 and filled["tophat_radius"] == 10


def test_the_parameters_given_are_completed_in_place():
    params = {"method": "Local Maxima"}
    puncta.process_puncta_detection(_volume(), voxels=(0.3, 0.1, 0.1), params=params)
    assert set(puncta.DETECTOR_DEFAULTS) | {"z_scale"} <= set(params)
    assert params["z_scale"] == 0.3 / 0.1


def test_the_settings_row_states_the_fallbacks_used():
    row = puncta.detection_settings_row({"method": "Local Maxima"}, 5, "raw")
    assert (row["sensitivity"], row["min_distance"], row["sigma"], row["tophat"], row["tophat_radius"]) == \
        (0.1, 3, 1.0, False, 10)
    assert "z_scale" not in row                                   # not given, not invented
    given = puncta.detection_settings_row({"sigma": 0.0, "threshold_rel": 0.05, "z_scale": 3.0}, 5, "raw")
    assert given["sigma"] == 0.0 and given["sensitivity"] == 0.05 and given["z_scale"] == 3.0


def test_a_batch_channel_without_a_puncta_row_records_sigma_one(reopened):
    """The synthetic batch has no Puncta sheet, so its channels take the
    pipeline's built-in settings, which leave sigma out."""
    for round_id in ("R1", "R2"):
        csv = constants.puncta_csv_path(reopened.reports, stages.raw_puncta_name(round_id, CHANNEL))
        recorded = provenance.latest(csv)["detection_parameters"]
        assert recorded["sigma"] == 1.0 and recorded["tophat_radius"] == 10
        assert puncta.detection_settings()[f"{round_id} - {CHANNEL}"]["sigma"] == 1.0
