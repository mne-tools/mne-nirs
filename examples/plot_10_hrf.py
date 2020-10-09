"""
.. _tut-fnirs-hrf:

GLM Analysis (Measured Data)
============================

In this example we analyse data from a real multichannel fNIRS
experiment (see :ref:`tut-fnirs-hrf-sim` for a simplified simulated
analysis). The experiment consists of three conditions
1) tapping with the left hand,
2) tapping with the right hand,
3) a control condition where the participant does nothing.
We use a GLM analysis to examine the neural activity associated with
the different tapping conditions.
An alternative epoching style analysis on the same data can be
viewed in the
`MNE documentation <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_.

This GLM analysis is a wrapper over the excellent
`Nilearn stats <https://github.com/nilearn/nilearn/tree/master/nilearn/stats>`_.

.. warning::
      This is a work in progress. Comments are appreciated. To provide feedback please create a github issue.

.. contents:: Page contents
   :local:
   :depth: 2

"""


# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM, glm_region_of_interest
from mne_nirs.visualisation import plot_glm_topo
from mne_nirs.channels import (get_long_channels, get_short_channels,
                               picks_pair_to_idx)

from nilearn.plotting import plot_design_matrix
from mne_nirs.utils._io import glm_to_tidy, _tidy_long_to_wide


###############################################################################
# Import raw NIRS data
# --------------------
#
# First we import the motor tapping data, these data are also
# described and used in the
# `MNE fNIRS tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_.
#
# After reading the data we resample down to 1Hz
# to meet github memory constraints.
#
# .. collapse:: Data description (click to expand)
#    :class: success
#
#    Optodes were placed over the motor cortex using the standard NIRX motor
#    montage, but with 8 short channels added (see their web page for details).
#    To view the sensor locations run
#    `raw_intensity.plot_sensors()`.
#    A sound was presented to indicate which hand the participant should tap.
#    Participants tapped their thumb to their fingers for 5s.
#    Conditions were presented in a random order with a randomised inter
#    stimulus interval.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
raw_intensity.resample(1.0)


###############################################################################
# Clean up annotations before analysis
# ------------------------------------
#
# Next we update the annotations by assigning names to each trigger ID.
# Then we crop the recording to the section containing our
# experimental conditions.

original_annotations = raw_intensity.annotations
new_des = [des for des in raw_intensity.annotations.description]
new_des = ['Control' if x == "1.0" else x for x in new_des]
new_des = ['Tapping/Left' if x == "2.0" else x for x in new_des]
new_des = ['Tapping/Right' if x == "3.0" else x for x in new_des]
keepers = [n == 'Control' or
           n == "Tapping/Left" or
           n == "Tapping/Right" for n in new_des]
idxs = np.array(np.where(keepers)[0])
annot = mne.Annotations(original_annotations.onset[idxs],
                        original_annotations.duration[idxs] * 5., 
                        np.array([new_des[idx] for idx in np.where(keepers)[0]]))
raw_intensity.set_annotations(annot)


###############################################################################
# Preprocess NIRS data
# --------------------
# Next we convert the raw data to haemoglobin concentration.

raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)


###############################################################################
#
# .. sidebar:: Relevant literature
#
#    Tachtsidis, Ilias, and Felix Scholkmann. "False positives and false
#    negatives in functional near-infrared spectroscopy: issues, challenges,
#    and the way forward." Neurophotonics 3.3 (2016): 031405.
#
# We then split the data in to
# short channels which predominantly contain systemic responses and
# long channels which have both neural and systemic contributions.

short_chs = get_short_channels(raw_haemo)
raw_haemo = get_long_channels(raw_haemo)


###############################################################################
# View experiment events
# ----------------------
#
# Next we examine the timing and order of events in this experiment.
# There are several options for how to view event information.
# The first option is to use MNE's plot events command.
# Here each dot represents when an event started.
# We observe that the order of conditions was randomised and the time between
# events is also randomised.

events, _ = mne.events_from_annotations(raw_haemo, verbose=False)
event_dict = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}
mne.viz.plot_events(events, event_id=event_dict,
                    sfreq=raw_haemo.info['sfreq'])


###############################################################################
#
# The previous plot did not illustrate the duration that an event lasted for.
# Alternatively, we can view the experiment using a boxcar plot, where the
# line is raised for the duration of the stimulus/condition.

s = mne_nirs.experimental_design.create_boxcar(raw_haemo)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
plt.plot(raw_haemo.times, s, axes=axes)
plt.legend(["Control", "Left", "Right"], loc="upper right")
plt.xlabel("Time (s)");


###############################################################################
# Create design matrix
# --------------------
#
# .. sidebar:: Relevant literature
#
#    For further discussion on design matrices see
#    the Nilearn examples. Specifically the 
#    `first level model <https://5712-1235740-gh.circle-artifacts.com/0/doc/_build/html/auto_examples/plot_first_level_model_details.html>`_
#    and 
#    `design matrix examples <https://5712-1235740-gh.circle-artifacts.com/0/doc/_build/html/auto_examples/04_glm_first_level_models/plot_design_matrix.html>`_.
#
# Next we create a model to fit our data to.
# The model consists of various components to model different things we assume
# contribute to the measured signal.
# We model the expected neural response for each experimental condition
# using the SPM haemodynamic response
# function (HRF) combined with the known stimulus event times and durations
# (as described above).
# We also include a third order polynomial drift and constant to model
# slow fluctuations in the data and a constant DC shift.

design_matrix = make_first_level_design_matrix(raw_haemo,
                                               hrf_model='spm', stim_dur=5.0,
                                               drift_order=3,
                                               drift_model='polynomial')


###############################################################################
#
# We also add the mean of the short channels to the design matrix.
# In theory these channels contain only systemic components, so including
# them in the design matrix allows us to estimate the neural component
# related to each experimental condition
# uncontaminated by systemic effects.

design_matrix["ShortHbO"] = np.mean(short_chs.copy().pick(
                                    picks="hbo").get_data(), axis=0)

design_matrix["ShortHbR"] = np.mean(short_chs.copy().pick(
                                    picks="hbr").get_data(), axis=0)


###############################################################################
#
# And we display a summary of the design matrix
# using standard Nilearn reporting functions.
# The first three columns represent the SPM HRF convolved with our stimulus
# event information.
# The next columns illustrate the drift and constant components.
# The last columns illustrate the short channel signals.

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


###############################################################################
# Examine expected response
# -------------------------
#
# The matrices above can be a bit abstract as they encompase multiple 
# conditons and regressors.
# Instead we can examine a single condition.
# Here we observe the boxcar function for a single condition,
# this illustrates when the stimulus was active.
# We also view the expected neural response using the HRF specified above,
# we observe that each time a stimulus is presented there is an expected
# brain response that lags the stimulus onset and consists of a large positive
# component followed by an undershoot.

s = mne_nirs.experimental_design.create_boxcar(raw_intensity)
plt.plot(raw_intensity.times, s[:, 1])
plt.plot(design_matrix['Tapping/Left'])
plt.xlim(180, 300)
plt.legend(["Stimulus", "Expected Response"])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")


###############################################################################
#
# Fit GLM to subset of data and estimate response for each experimental condition
# -------------------------------------------------------------------------------
#
# .. sidebar:: Relevant literature
#
#    Huppert TJ. Commentary on the statistical properties of noise and its
#    implication on general linear models in functional near-infrared
#    spectroscopy. Neurophotonics. 2016;3(1)
#
# We run a GLM fit for the data and experiment matrix.
# First we analyse just the first two channels which correspond to HbO and HbR
# of a single source detector pair.

data_subset = raw_haemo.copy().pick(picks=range(2))
glm_est = run_GLM(data_subset, design_matrix)


###############################################################################
#
# We then display the results. Note that the control condition sits
# around zero
# and that the HbO is positive and larger than the HbR, this is to be expected.
# Further, we note that for this channel the response to tapping on the
# right hand is larger than the left. And the values are similar to what
# is seen in the epoching tutorial.

plt.scatter(design_matrix.columns[:3], glm_est['S1_D1 hbo'].theta[:3] * 1e6)
plt.scatter(design_matrix.columns[:3], glm_est['S1_D1 hbr'].theta[:3] * 1e6)
plt.xlabel("Experiment Condition")
plt.ylabel("Haemoglobin (μM)")
plt.legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])
plt.hlines([0.0], 0, 2)


###############################################################################
# Fit GLM to all data and view topographic distribution
# -----------------------------------------------------
#
# Lastly we can run the GLM analysis on all sensors and plot the result on a
# topomap.
# We see the same result as in the MNE tutorial,
# that activation is largest
# contralateral to the tapping side. Also note that HbR tends to be the
# negative of HbO as expected.

glm_est = run_GLM(raw_haemo, design_matrix)
plot_glm_topo(raw_haemo, glm_est, design_matrix,
              requested_conditions=['Tapping/Left',
                                    'Tapping/Right'])


###############################################################################
# Analyse regions of interest
# ---------------------------
#
# Or alternatively we can summarise the responses across regions of interest
# for each condition. And you can plot it with your favorite software.

left = [[1, 1], [1, 2], [1, 3], [2, 1], [2, 3],
        [2, 4], [3, 2], [3, 3], [4, 3], [4, 4]]
right = [[5, 5], [5, 6], [5, 7], [6, 5], [6, 7],
         [6, 8], [7, 6], [7, 7], [8, 7], [8, 8]]

groups = dict(Left_ROI=picks_pair_to_idx(raw_haemo, left),
              Right_ROI=picks_pair_to_idx(raw_haemo, right))

df = pd.DataFrame()
for idx, col in enumerate(design_matrix.columns[:3]):
    df = df.append(glm_region_of_interest(glm_est, groups, idx, col))


###############################################################################
#
# Compute contrasts
# -----------------
#
# We can also define a contrast as described in
# `Nilearn docs <https://5874-1235740-gh.circle-artifacts.com/0/doc/_build/html/auto_examples/04_glm_first_level_models/plot_localizer_surface_analysis.html>`_
# and plot it.
# Here we contrast the response to tapping on the left hand with the response
# from tapping on the right hand.

contrast_matrix = np.eye(design_matrix.shape[1])
basic_conts = dict([(column, contrast_matrix[i])
                   for i, column in enumerate(design_matrix.columns)])
contrast_LvR = basic_conts['Tapping/Left'] - basic_conts['Tapping/Right']
contrast = mne_nirs.statistics.compute_contrast(glm_est, contrast_LvR)
mne_nirs.visualisation.plot_glm_contrast_topo(raw_haemo, contrast)


###############################################################################
# Export Results
# ---------------
#
# .. sidebar:: Relevant literature
#
#    Wickham, Hadley. "Tidy data." Journal of Statistical Software 59.10 (2014): 1-23.
#
# Here we export the data in a tidy pandas data frame.
# We export the GLM results for every channel and condition.
# Data is exported in long format by default.
# However, a helper function is also provided to convert the long data to wide format.
# The long to wide conversion also adds some additonal derived data, such as
# if a significant response (p<0.05) was observed, which sensor and detector is
# in the channel, which chroma, etc.

df = glm_to_tidy(raw_haemo, glm_est, design_matrix)


###############################################################################
# Determine true and false positive rates
# ---------------------------------------
#
# We can query the exported data frames to determine the true and false
# positive rates. Note: optodes cover a greater region than just the
# motor cortex, so we dont expect 100% of channels to detect responses to
# the tapping, but we do expect 5% or less for the false positive rate.

(df
 .query('condition in ["Control", "Tapping/Left", "Tapping/Right"]')
 .groupby(['condition', 'Chroma'])
 .agg(['mean'])
 .drop(['df', 'mse', 'p_value', 't'], 1)
 )
