"""
.. _tut-fnirs-hrf:

Haeomodynamic response function analysis
========================================

This document is a work in progress.
It is a first attempt to add GLM analysis to MNE processing of NIRS data.

This is basically a wrapper over the excellent Nilearn stats.
https://github.com/nilearn/nilearn/tree/master/nilearn/stats .

Currently the analysis is only being run on the first third of the measurement
to meet github actions memory constraints.
This means the results are noisier than the MNE fnirs tutorial.

This document quite poorly written, read with caution.


.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import matplotlib.pyplot as plt
import mne
import mne_nirs

from mne_nirs.experimental_design import create_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.visualisation import plot_GLM_topo

from nilearn.reporting import plot_design_matrix


###############################################################################
# Import raw NIRS data
# --------------------
#
# Import the motor tapping data also used in MNE tutorial.
# Crop to meet github memory constraints.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir,
                                     verbose=True).load_data()
raw_intensity.crop(tmax=800)


###############################################################################
# Clean up annotations before analysis
# ------------------------------------
#
# Here I update the annotation names and remove annotations that indicated
# the experiment began and finished.

new_des = [des for des in raw_intensity.annotations.description]
new_des = ['Control' if x == "1.0" else x for x in new_des]
new_des = ['Tapping/Left' if x == "2.0" else x for x in new_des]
new_des = ['Tapping/Right' if x == "3.0" else x for x in new_des]
annot = mne.Annotations(raw_intensity.annotations.onset,
                        raw_intensity.annotations.duration, new_des)
raw_intensity.set_annotations(annot)
raw_intensity.annotations.crop(35, 2967)


###############################################################################
# Preprocess NIRS data
# --------------------
#
# Convert the raw data to haemoglobin concentration and bandpass filter.


picks = mne.pick_types(raw_intensity.info, meg=False, fnirs=True)

dists = mne.preprocessing.nirs.source_detector_distances(
    raw_intensity.info, picks=picks)
raw_intensity.pick(picks[dists > 0.01])
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
raw_haemo = raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2,
                             l_trans_bandwidth=0.02)


###############################################################################
# View experiment events
# ----------------------
#
# First we view the experiment using MNEs plot events.

events, _ = mne.events_from_annotations(raw_haemo)
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}
mne.viz.plot_events(events, event_id=event_dict,
                    sfreq=raw_haemo.info['sfreq'])

###############################################################################
#
# Next we view the same information but displayed as a block design.

s = mne_nirs.experimental_design.create_boxcar(raw_haemo)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
plt.plot(raw_haemo.times, s, axes=axes)
plt.legend(["Control", "Left", "Right"], loc="upper right")
plt.xlabel("Time (s)")


###############################################################################
# Create design matrix
# --------------------
#
# This analysis specifies the experiment using a design matrix which
# is created and plotted below.
# In this example we use the standard SPM haemodynamic response function and
# include a first order polynomial drift.

design_matrix = create_first_level_design_matrix(raw_intensity,
                                                 hrf_model='spm', stim_dur=5.0,
                                                 drift_order=1,
                                                 drift_model='polynomial')


###############################################################################
#
# And we display a summary of the design matrix
# using standard Nilearn reporting functions.

plot_design_matrix(design_matrix)


###############################################################################
#
# And we can also look at a single experimental condition.

s = mne_nirs.experimental_design.create_boxcar(raw_intensity)
plt.plot(raw_intensity.times, s[:, 1])
plt.plot(design_matrix['Tapping/Left'])
plt.xlim(180, 300)
plt.legend(["Stimulus", "Expected HRF"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")


###############################################################################
# Fit GLM to estimate response
# ----------------------------
#
# We run a GLM fit for the data and experiment matrix.
# First we analyse just the first two channels which correspond HbO and HbR
# of a single source detector pair.

labels, glm_estimates = run_GLM(raw_haemo.copy().pick(picks=range(2)),
                                design_matrix)

###############################################################################
#
# We then display the results. Note that the control condition sits
# around zero.
# And that the HbO is positive and larger than the HbR, this is to be expected.
# Further, we note that for this channel the response to tapping on the
# right hand is larger than the left. And the values are similar to what
# is seen in the epoching tutorial.

plt.scatter(design_matrix.columns[:3],
            glm_estimates[labels[0]].theta[:3] * 1e6)
plt.scatter(design_matrix.columns[:3],
            glm_estimates[labels[1]].theta[:3] * 1e6)
plt.xlabel("Experiment Condition")
plt.ylabel("Haemoglobin (μM)")
plt.legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])
plt.hlines([0.0], 0, 2)
plt.show()


###############################################################################
# View GLM resufvlts for all sensors
# --------------------------------
#
# Lastly we run the GLM analysis on all sensors and plot the result on a
# toppmap.
# We see the same result as in the MNE tutorial that activation is largest
# contralateral to the tapping side. Also note that HbR tends to be the
# negative sof HbO as expected.

labels, glm_estimates = run_GLM(raw_haemo, design_matrix)
plot_GLM_topo(raw_haemo, labels, glm_estimates, design_matrix,
              requested_conditions=['Tapping/Left', 'Tapping/Right'])
