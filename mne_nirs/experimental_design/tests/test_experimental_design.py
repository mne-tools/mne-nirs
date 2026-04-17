# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne
import numpy as np

import mne_nirs
from mne_nirs.experimental_design import (
    drift_high_pass,
    longest_inter_annotation_interval,
    make_first_level_design_matrix,
)
from mne_nirs.simulation import simulate_nirs_raw

# for the comparison of vif we need these two libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor

def make_first_level_design_matrix_w_statsmodels_vif(
    raw,
    stim_dur=1.0,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=(0,),
    add_regs=None,
    add_reg_names=None,
    min_onset=-24,
    oversampling=50,
):
    """same test as make_first_level_design_matrix but ran with statsmodels"""
    from nilearn.glm.first_level import make_first_level_design_matrix
    from nilearn.glm import regression
    from pandas import DataFrame

    frame_times = raw.times
    # Create events for nilearn
    conditions = raw.annotations.description
    onsets = raw.annotations.onset - raw.first_time
    duration = stim_dur * np.ones(len(conditions))
    events = DataFrame(
        {"trial_type": conditions, "onset": onsets, "duration": duration}
    )

    dm = make_first_level_design_matrix(
        frame_times,
        events,
        drift_model=drift_model,
        drift_order=drift_order,
        hrf_model=hrf_model,
        min_onset=min_onset,
        high_pass=high_pass,
        add_regs=add_regs,
        oversampling=oversampling,
        add_reg_names=add_reg_names,
        fir_delays=fir_delays,
    )

    dm_no_const = dm.drop(columns=["constant"], errors="ignore")
    predictor_names = list(dm_no_const.columns)

    # VIF 1–4 shows low to moderate correlation between predictors
    # VIF > 4 is often though to indicate high multicollinearity,
    # which may hint that some varaiables may need to be dropped, combined etc

    vif = [
        variance_inflation_factor(dm_no_const.values, i)
        for i in range(dm_no_const.shape[1])
    ]


    return dm, dict(zip(predictor_names, vif))



def _load_dataset():
    """Load data and tidy it a bit"""
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, "Participant-1")
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()

    raw_intensity.crop(0, raw_intensity.annotations.onset[-1])

    new_des = [des for des in raw_intensity.annotations.description]
    new_des = ["A" if x == "1.0" else x for x in new_des]
    new_des = ["B" if x == "2.0" else x for x in new_des]
    new_des = ["C" if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(
        raw_intensity.annotations.onset, raw_intensity.annotations.duration, new_des
    )
    raw_intensity.set_annotations(annot)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks
    )
    raw_intensity.pick(picks[dists > 0.01])

    assert "fnirs_cw_amplitude" in raw_intensity
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
    assert (
        bc[int(raw_intensity.annotations.onset[0] * raw_intensity.info["sfreq"]), :][0]
        == 1
    )

    # Only one condition was ever present at a time in this data
    # So boxcar should never overlap across channels
    assert np.max(np.mean(bc, axis=1)) * bc.shape[1] == 1


def test_create_design():
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    design_matrix, vif = make_first_level_design_matrix(
        raw_intensity, drift_order=1, drift_model="polynomial"
    )

    assert design_matrix.shape[0] == raw_intensity._data.shape[1]
    # Number of columns is number of conditions plus the drift plus constant
    assert (
        design_matrix.shape[1]
        == len(np.unique(raw_intensity.annotations.description)) + 2
    )


def test_cropped_raw():
    # Ensure timing is correct for cropped signals
    raw = simulate_nirs_raw(
        sfreq=1.0,
        amplitude=1.0,
        sig_dur=300.0,
        stim_dur=1.0,
        isi_min=20.0,
        isi_max=40.0,
    )

    onsets = raw.annotations.onset
    onsets_after_crop = [onsets[idx] for idx in np.where(onsets > 100)]

    raw.crop(tmin=100)
    design_matrix, vif = make_first_level_design_matrix(
        raw, drift_order=0, drift_model="polynomial"
    )

    # 100 corrects for the crop time above
    # 4 is peak time after onset
    new_idx = np.round(onsets_after_crop[0][0]) - 100 + 4
    assert design_matrix["A"][new_idx] > 0.09


def test_high_pass_helpers():
    # Test the helpers give reasonable values
    raw = simulate_nirs_raw(
        sfreq=1.0,
        amplitude=1.0,
        sig_dur=300.0,
        stim_dur=1.0,
        isi_min=20.0,
        isi_max=38.0,
    )
    lisi, names = longest_inter_annotation_interval(raw)
    lisi = lisi[0]
    assert lisi >= 20
    assert lisi <= 40
    assert drift_high_pass(raw) >= 1 / (40 * 2)
    assert drift_high_pass(raw) <= 1 / (20 * 2)

def test_statsmodels_vif_equality():
    # Ensure our custom code for vif calculation matches statsmodels
    raw_intensity = _load_dataset()
    raw_intensity.crop(450, 600)  # Keep the test fast
    design_matrix, vif = make_first_level_design_matrix(
        raw_intensity, drift_order=1, drift_model="polynomial", vif_export=True
    )

    design_matrix_statsm, vif_statsm = make_first_level_design_matrix_w_statsmodels_vif(
        raw_intensity, drift_order=1, drift_model="polynomial"
    )

    # expect near identical results but not exact since ourrs is using glm from nii.learn
    # wheras statsmodel has their own implmentation before extracting the vif values
    # note vif will come with a level of uncertainity +/- 0.05 of what is reported
    for key in vif:
            assert abs(vif[key] - vif_statsm[key]) < 0.05


test_statsmodels_vif_equality()