"""
.. _tut-fnirs-hrf-sim:

GLM Analysis (Simulated Data)
=============================

Simulate a signal and then run analysis on it.

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
# Some text.

fs = 3.
amp = 4.

raw = mne_nirs.simulation.simulate_nirs_data(fs=fs, signal_length_s=60 * 5,
                                             amplitude=amp)

raw.plot(duration=600)

###############################################################################
# Create design matrix
# ------------------------------------
#
# Some text.

design_matrix = create_first_level_design_matrix(raw, stim_dur=5.0,
                                                 drift_order=1,
                                                 drift_model='polynomial')

fig = plot_design_matrix(design_matrix)

###############################################################################
# Estimate response on clean data
# -------------------------------
#
# Some text.

labels, glm_estimates = run_GLM(raw, design_matrix)

print(glm_estimates[labels[0]].theta)

error = glm_estimates[labels[0]].theta[0] - amp * 1.e-6

print(error)


###############################################################################
# Simulate noisy NIRS data
# ------------------------
#
# Some text.

raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * 3.
raw.plot(duration=600)


###############################################################################
# How does estimate error vary with added noise?
# ----------------------------------------------
#
# Some text.

noise_std = []
error = []

for std in np.arange(1, 10):
    for repeat in range(5):
        raw = mne_nirs.simulation.simulate_nirs_data(
            fs=fs, signal_length_s=60 * 10, amplitude=amp)
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
# Some text.

noise_std = []
error = []
signal_length = []

std = 3
for repeat in range(20):
    for slen in [3., 11., 20.]:

        raw = mne_nirs.simulation.simulate_nirs_data(
            stim_dur=5., fs=fs, signal_length_s=60 * slen, amplitude=amp)
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
