# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs


def test_create_boxcar():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()
    raw_intensity = raw_intensity.pick(picks=[0])
    mne_nirs.experimental_design.create_boxcar(raw_intensity)


def test_create_design():
    fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
    fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
    raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                         verbose=True).load_data()
    raw_intensity.crop(0, 200)
    new_des = [des for des in raw_intensity.annotations.description];
    new_des = ['A' if x == "1.0" else x for x in new_des]
    new_des = ['B' if x == "2.0" else x for x in new_des]
    new_des = ['C' if x == "3.0" else x for x in new_des]
    annot = mne.Annotations(raw_intensity.annotations.onset,
                            raw_intensity.annotations.duration, new_des)
    raw_intensity.set_annotations(annot)

    design_matrix = mne_nirs.experimental_design.create_first_level_design_matrix(raw_intensity)

    picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)
    dists = mne.preprocessing.nirs.source_detector_distances(
        raw_intensity.info, picks=picks)
    raw_intensity.pick(picks[dists > 0.01])
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    labels, glm_estimates = mne_nirs.experimental_design.run_GLM(raw_haemo, design_matrix)
    mne_nirs.experimental_design.plot_GLM_topo(raw_haemo, labels, glm_estimates, design_matrix)