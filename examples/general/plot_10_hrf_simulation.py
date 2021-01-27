"""
.. _tut-fnirs-hrf-sim:

GLM Analysis (Simulated)
=============================

In this example we simulate a block design fNIRS experiment and analyse
the simulated signal. We investigate the effect additive noise and
measurement length has on response amplitude estimates.

Test change is visible

.. contents:: Page contents
   :local:
   :depth: 2

"""
# sphinx_gallery_thumbnail_number = 3

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import mne
import mne_nirs
import matplotlib.pylab as plt
import numpy as np
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from nilearn.plotting import plot_design_matrix
np.random.seed(1)


###############################################################################
# Simulate noise free NIRS data
# -----------------------------
#
# First we simulate some noise free data. We simulate 5 minutes of data with a
# block design. The inter stimulus interval of the stimuli is uniformly
# selected between 15 and 45 seconds.
# The amplitude of the simulated signal is 4 uMol and the sample rate is 3 Hz.
# The simulated signal is plotted below.

sfreq = 3.
amp = 4.

raw = mne_nirs.simulation.simulate_nirs_raw(
    sfreq=sfreq, sig_dur=60 * 5, amplitude=amp, isi_min=15., isi_max=45.)
raw.plot(duration=300, show_scrollbars=False)


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
fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


###############################################################################
# Estimate response on clean data
# -------------------------------
#
# Here we run the GLM analysis on the clean data.
# The design matrix had three columns, so we get an estimate for our simulated
# event, the first order drift, and the constant.
# We see that the estimate of the first component is 4e-6 (4 uM),
# which was the amplitude we used in the simulation.
# We also see that the mean square error of the model fit is close to zero.

glm_est = run_GLM(raw, design_matrix)

print("Estimate:", glm_est['Simulated'].theta[0],
      "  MSE:", glm_est['Simulated'].MSE,
      "  Error (uM):", 1e6*(glm_est['Simulated'].theta[0] - amp*1e-6))


###############################################################################
# Simulate noisy NIRS data (white)
# --------------------------------
#
# Real data has noise. Here we add white noise, this noise is not realistic
# but serves as a reference point for evaluating the estimation process.
# We run the GLM analysis exactly as in the previous section
# and plot the noisy data and the GLM fitted model.
# We print the response estimate and see that is close, but not exactly correct,
# we observe the mean square error is similar to the added noise.
# Note that the clean data plot is so similar to the GLM estimate that it is hard to see unless zoomed in. 

# First take a copy of noise free data for comparison
raw_noise_free = raw.copy()

raw._data += np.random.normal(0, np.sqrt(1e-11), raw._data.shape)
glm_est = run_GLM(raw, design_matrix)

plt.plot(raw.times, raw_noise_free.get_data().T * 1e6)
plt.plot(raw.times, raw.get_data().T * 1e6, alpha=0.3)
plt.plot(raw.times, glm_est['Simulated'].theta[0] * design_matrix["A"].values * 1e6)
plt.xlabel("Time (s)")
plt.ylabel("Haemoglobin (uM)")
plt.legend(["Clean Data", "Noisy Data", "GLM Estimate"])

print("Estimate:", glm_est['Simulated'].theta[0],
      "  MSE:", glm_est['Simulated'].MSE,
      "  Error (uM):", 1e6*(glm_est['Simulated'].theta[0] - amp*1e-6))


###############################################################################
# Simulate noisy NIRS data (colored)
# ----------------------------------
#
# Here we add colored noise which better matches what is seen with real data.
# Again, the same GLM procedure is run.
# The estimate is reported below, and even though the signal was difficult to
# observe in the raw data, the GLM analysis has extracted an accurate estimate.
# However, the error is greater for the colored than white noise.

raw = raw_noise_free.copy()
cov = mne.Covariance(np.ones(1) * 1e-11, raw.ch_names,
                     raw.info['bads'], raw.info['projs'], nfree=0)
raw = mne.simulation.add_noise(raw, cov,
                               iir_filter=[1., -0.58853134, -0.29575669,
                                           -0.52246482, 0.38735476,
                                           0.02428681])
design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0,
                                               drift_order=1,
                                               drift_model='polynomial')
glm_est = run_GLM(raw, design_matrix)

plt.plot(raw.times, raw_noise_free.get_data().T * 1e6)
plt.plot(raw.times, raw.get_data().T * 1e6, alpha=0.3)
plt.plot(raw.times, glm_est['Simulated'].theta[0] * design_matrix["A"].values * 1e6)
plt.xlabel("Time (s)")
plt.ylabel("Haemoglobin (uM)")
plt.legend(["Clean Data", "Noisy Data", "GLM Estimate"])

print("Estimate:", glm_est['Simulated'].theta[0],
      "  MSE:", glm_est['Simulated'].MSE,
      "  Error (uM):", 1e6*(glm_est['Simulated'].theta[0] - amp*1e-6))


###############################################################################
# How does increasing the measurement length affect estimation accuracy?
# ----------------------------------------------------------------------
#
# The easiest way to reduce error in your response estimate is to collect more
# data. Here we simulated increasing the recording time to 30 minutes.
# We run the same analysis and observe that the error is reduced from
# approximately 0.6 uM for 5 minutes of data to 0.25 uM for 30 minutes of data.

raw = mne_nirs.simulation.simulate_nirs_raw(
    sfreq=sfreq, sig_dur=60 * 30, amplitude=amp, isi_min=15., isi_max=45.)
cov = mne.Covariance(np.ones(1) * 1e-11, raw.ch_names,
                     raw.info['bads'], raw.info['projs'], nfree=0)
raw = mne.simulation.add_noise(raw, cov,
                               iir_filter=[1., -0.58853134, -0.29575669,
                                           -0.52246482, 0.38735476,
                                           0.02428681])

design_matrix = make_first_level_design_matrix(raw, stim_dur=5.0,
                                               drift_order=1,
                                               drift_model='polynomial')
glm_est = run_GLM(raw, design_matrix)

plt.plot(raw.times, raw.get_data().T * 1e6, alpha=0.3)
plt.plot(raw.times, glm_est['Simulated'].theta[0] * design_matrix["A"].values * 1e6)
plt.xlabel("Time (s)")
plt.ylabel("Haemoglobin (uM)")
plt.legend(["Noisy Data", "GLM Estimate"])

print("Estimate:", glm_est['Simulated'].theta[0],
      "  MSE:", glm_est['Simulated'].MSE,
      "  Error (uM):", 1e6*(glm_est['Simulated'].theta[0] - amp*1e-6))


###############################################################################
# Conclusion?
# -----------
#
# In this example we have generated a noise free signal containing simulated
# haemodynamic responses. We were then able to accurately estimate the amplitude
# of the simulated signal. We then added noise and illustrated that the
# estimate provided by the GLM was correct, but contained some error. We
# observed that as the measurement time was increased, the estimated
# error decreased.
