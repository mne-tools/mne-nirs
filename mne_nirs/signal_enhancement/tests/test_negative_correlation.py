# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs
import numpy as np


def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()

    raw_intensity.annotations.crop(raw_intensity.annotations.onset[0],
                                   raw_intensity.annotations.onset[-1])

    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ['A' if x == "1.0" else x for x in new_des]
    new_des = ['B' if x == "2.0" else x for x in new_des]
    new_des = ['C' if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(raw_intensity.annotations.onset,
                            raw_intensity.annotations.duration, new_des)
    raw_intensity.set_annotations(annot)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks)
    raw_intensity.pick(picks[dists > 0.01])

    assert 'fnirs_raw' in raw_intensity
    assert len(np.unique(raw_intensity.annotations.description)) == 4

    return raw_intensity


def test_cui():
    raw_intensity = _load_dataset()
    raw_intensity = raw_intensity.pick(picks=range(2))  # Keep the test fast
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_anti = mne_nirs.signal_enhancement.enhance_negative_correlation(
        raw_haemo)
    assert np.abs(np.corrcoef(raw_haemo._data[0],
                              raw_haemo._data[1])[0, 1]) < 1

    np.testing.assert_almost_equal(np.corrcoef(raw_anti._data[0],
                                   raw_anti._data[1])[0, 1],
                                   -1)

