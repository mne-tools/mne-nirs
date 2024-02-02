# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne
import mne_nirs
import numpy as np
from mne_nirs.experimental_design import make_first_level_design_matrix, \
    longest_inter_annotation_interval, drift_high_pass
from mne_nirs.simulation import simulate_nirs_raw


def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()

    raw_intensity.crop(0, raw_intensity.annotations.onset[-1])

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

    assert 'fnirs_cw_amplitude' in raw_intensity
    assert len(np.unique(raw_intensity.annotations.description)) == 4

    return raw_intensity


def test_create_boxcar():
    raw_intensity = _load_dataset()
    raw_intensity = raw_intensity.pick(picks=[0])  # Keep the test fast
    bc = mne_nirs.experimental_design.create_boxcar(raw_intensity)

    assert bc.shape[0] == raw_intensity._data.shape[1]
    assert bc.shape[1] == len(np.unique(raw_intensity.annotations.description))

    assert np.max(bc) == 1
    assert np.min(bc) == 0

    # The value of the boxcar should be 1 when a trigger fires
    assert bc[int(raw_intensity.annotations.onset[0] *
                  raw_intensity.info['sfreq']), :][0] == 1

    # Only one condition was ever present at a time in this data
    # So boxcar should never overlap across channels
    assert np.max(np.mean(bc, axis=1)) * bc.shape[1] == 1


def test_create_design():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    design_matrix = make_first_level_design_matrix(raw_intensity,
                                                   drift_order=1,
                                                   drift_model='polynomial')

    assert design_matrix.shape[0] == raw_intensity._data.shape[1]
    # Number of columns is number of conditions plus the drift plus constant
    assert design_matrix.shape[1] ==\
        len(np.unique(raw_intensity.annotations.description)) + 2


def test_cropped_raw():
    # Ensure timing is correct for cropped signals
    raw = simulate_nirs_raw(sfreq=1., amplitude=1., sig_dur=300., stim_dur=1.,
                            isi_min=20., isi_max=40.)

    onsets = raw.annotations.onset
    onsets_after_crop = [onsets[idx] for idx in np.where(onsets > 100)]

    raw.crop(tmin=100)
    design_matrix = make_first_level_design_matrix(raw, drift_order=0,
                                                   drift_model='polynomial')

    # 100 corrects for the crop time above
    # 4 is peak time after onset
    new_idx = np.round(onsets_after_crop[0][0]) - 100 + 4
    assert design_matrix["A"][new_idx] > 0.09


def test_high_pass_helpers():
    # Test the helpers give reasonable values
    raw = simulate_nirs_raw(sfreq=1., amplitude=1., sig_dur=300., stim_dur=1.,
                            isi_min=20., isi_max=38.)
    lisi, names = longest_inter_annotation_interval(raw)
    lisi = lisi[0]
    assert lisi >= 20
    assert lisi <= 40
    assert drift_high_pass(raw) >= 1 / (40 * 2)
    assert drift_high_pass(raw) <= 1 / (20 * 2)
