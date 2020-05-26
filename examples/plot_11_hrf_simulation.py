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

import mne
import mne_nirs
import matplotlib.pylab as plt
import numpy as np
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from nilearn.reporting import plot_design_matrix
import seaborn as sns
np.random.seed(1)

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

raw = mne_nirs.simulation.simulate_nirs_raw(
    sfreq=sfreq, sig_dur=60 * 5, amplitude=amp, isi_min=15., isi_max=45.)
raw.plot(duration=600, show_scrollbars=False)

###############################################################################
# Create design matrix
# ------------------------------------
#
# Next we create a design matrix based on the annotation times in the simulated
# data. We use the nilearn plotting function to visualise the design matrix.
# For more details on this procedure see :ref:`tut-fnirs-hrf`.

design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0,
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
# We see that the estimate of the first component is 4e-6 (4 uM),
# which was the amplitude we used in the simulation.
# We also see that the mean square error of the model fit is close to zero.

labels, glm_est = run_GLM(raw, design_matrix)

print("Estimate:", glm_est[labels[0]].theta[0],
      "  MSE:", glm_est[labels[0]].MSE)


###############################################################################
# Simulate noisy NIRS data (white)
# --------------------------------
#
# Real data has noise. Here we add white noise with a standard deviation
# of 3 uM, this noise is not realistic, 
# but suffices for this demo.
# We plot the noisy data and the GLM fitted model.
# We report the model estimate and mean square error of the fit.

# First take a copy of noise free data for comparison
raw_noise_free = raw.copy()

raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * 3.
labels, glm_est = run_GLM(raw, design_matrix)

plt.plot(raw.times, raw_noise_free.get_data().T)
plt.plot(raw.times, raw.get_data().T, alpha=0.3)
plt.plot(raw.times, glm_est[labels[0]].theta[0] * design_matrix["A"].values)
plt.xlabel("Time (s)")
plt.legend(["Clean Data", "Noisy Data", "GLM Estimate"])

print("Estimate:", glm_est[labels[0]].theta[0],
      "  MSE:", glm_est[labels[0]].MSE)


###############################################################################
# Simulate noisy NIRS data (colored)
# ----------------------------------
#
# Here we add colored noise.

raw = raw_noise_free.copy()
cov = mne.Covariance(np.ones(1)*1e-11, raw.ch_names,
                     raw.info['bads'], raw.info['projs'], nfree=0)
raw = mne.simulation.add_noise(raw, cov,
                               iir_filter=[1. , -0.58853134, -0.29575669,
                                           -0.52246482,  0.38735476,
                                           0.02428681])
labels, glm_est = run_GLM(raw, design_matrix)

plt.plot(raw.times, raw_noise_free.get_data().T)
plt.plot(raw.times, raw.get_data().T, alpha=0.3)
plt.plot(raw.times, glm_est[labels[0]].theta[0] * design_matrix["A"].values)
plt.xlabel("Time (s)")
plt.legend(["Clean Data", "Noisy Data", "GLM Estimate"])

print("Estimate:", glm_est[labels[0]].theta[0],
      "  MSE:", glm_est[labels[0]].MSE)


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
        raw = mne_nirs.simulation.simulate_nirs_raw(
            sfreq=sfreq, sig_dur=60 * 10, amplitude=amp)
        raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * std

        design_matrix = make_first_level_design_matrix(
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

        raw = mne_nirs.simulation.simulate_nirs_raw(
            stim_dur=5., sfreq=sfreq, sig_dur=60 * slen, amplitude=amp)
        raw._data += np.random.randn(raw._data.shape[1]) * 1.e-6 * std

        design_matrix = make_first_level_design_matrix(
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
