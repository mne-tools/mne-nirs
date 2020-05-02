"""
.. _tut-fnirs-hrf-sim:

GLM Analysis (Simulated Data)
=============================

In this example we simulate a block design NIRS experiment and analyse
the simulated signal. We investigate the effect additive noise has
on response amplitude estimates, and the effect of measurement length.

.. warning::
      This is a work in progress. Suggestions of improvements are
      appreciated. I am finalising the code, then will fix the text.

.. contents:: Page contents
   :local:
   :depth: 2

"""

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne_nirs
import matplotlib.pylab as plt
import numpy as np
from mne_nirs.experimental_design import create_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from nilearn.reporting import plot_design_matrix
import seaborn as sns
np.random.seed(0)

###############################################################################
# Simulate noise free NIRS data
# -----------------------------
#
# First we simulate some noise free data. We simulate 5 minutes of data with a
# block design. The inter stimulus interval of the stimuli is uniformly
# selected between 15 and 45 seconds.
# The amplitude of the simulated signal is 4 uMol and the sample rate is 3 Hz.

sfreq = 3.
amp = 4.

raw = mne_nirs.simulation.simulate_block_design(
    sfreq=sfreq, sig_dur=60 * 5, amplitude=amp, isi_min=15., isi_max=45.)
raw.plot(duration=600, show_scrollbars=False)

###############################################################################
# Create design matrix
# ------------------------------------
#
# Next we create a design matrix based on the annotation times in the simulated
# data. We use the nilearn plotting function to visualise the design matrix.
# For more details on this procedure see :ref:`tut-fnirs-hrf`.

design_matrix = create_first_level_design_matrix(raw, stim_dur=5.0,
                                                 drift_order=1,
                                                 drift_model='polynomial')
fig = plot_design_matrix(design_matrix)

###############################################################################
# Estimate response on clean data
# -------------------------------
#
# Now we can run the GLM analysis on the clean data.
# The design matrix had three columns, so we get an estimate for our simulated
# event, the first order drift, and the constant.
# We see that the estimate of the first component is 4e-6, or 4 uM.
# Which is the amplitude we used in the simulation.
# Consequently the error is zero.

labels, glm_estimates = run_GLM(raw, design_matrix)

print(glm_estimates[labels[0]].theta)






###############################################################################
# Simulate noisy NIRS data
# ------------------------
#
# Real data has noise. Here we add white noise with a standard deviation
# of 3 uM, this value is small compared to real data, but suffices for
# this demo.
# We then use the GLM approach to estimate the response size and report
# the estimate error.

raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * 3.
raw.plot(duration=600, show_scrollbars=False)
labels, glm_estimates = run_GLM(raw, design_matrix)
error = glm_estimates[labels[0]].theta[0] - amp * 1.e-6
print(error)


###############################################################################
# How does estimate error vary with added noise?
# ----------------------------------------------
#
# Now we can vary the amount of noise added and observe how this affects
# the amplitude estimate.
# Here we observe that as the noise is increased the estimate error increases.

noise_std = []
error = []

for std in np.arange(1, 10):
    for repeat in range(5):
        raw = mne_nirs.simulation.simulate_block_design(
            sfreq=sfreq, sig_dur=60 * 10, amplitude=amp)
        raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * std

        design_matrix = create_first_level_design_matrix(
            raw, stim_dur=5.0, drift_order=1, drift_model='polynomial')

        labels, glm_estimates = run_GLM(raw, design_matrix)

        noise_std.append(np.std(raw._data))
        error_abs = glm_estimates[labels[0]].theta[0] - amp * 1.e-6
        error_percentage = error_abs / (amp * 1.e-6)
        error.append(error_percentage * 100)

sns.scatterplot(noise_std, error)
plt.xlabel("Std of signal + noise")
plt.ylabel("Estimate error (%)")
plt.ylim(-30, 30)
plt.hlines(np.mean(error), 0.1e-5, 1e-5, linestyles='dashed')
plt.vlines(3.e-6, -100, 100, linestyles='dashed')


###############################################################################
# How does estimate error vary with signal length?
# ------------------------------------------------
#
# Finally we can vary the length of the simulated signal and observe how
# this affects the estimate error.
# We observe that increasing the signal length decreases the estimate error.

noise_std = []
error = []
signal_length = []

std = 3
for repeat in range(20):
    for slen in [3., 11., 20.]:

        raw = mne_nirs.simulation.simulate_block_design(
            stim_dur=5., sfreq=sfreq, sig_dur=60 * slen, amplitude=amp)
        raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * std

        design_matrix = create_first_level_design_matrix(
            raw, stim_dur=5.0, drift_order=1, drift_model='polynomial')

        labels, glm_estimates = run_GLM(raw, design_matrix)

        error_abs = glm_estimates[labels[0]].theta[0] - amp * 1.e-6
        error_percentage = error_abs / (amp * 1.e-6)
        error.append(error_percentage * 100)
        signal_length.append(slen)

sns.scatterplot(signal_length, error, alpha=0.3)
plt.xlabel("Length of measurement (min)")
plt.ylabel("Estimate error (%)")
plt.ylim(-30, 30)
plt.hlines(np.mean(error), 0, 1e-5, linestyles='dashed')
