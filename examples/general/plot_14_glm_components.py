"""
.. _tut-fnirs-glm-components:

GLM and Design Matrix Elements
==============================

This tutorial describes the various design choices available when analysing
your fNIRS data using a GLM approach.

.. sidebar:: Nilearn

   If you use MNE-NIRS to conduct a GLM analysis please cite Nilearn.
   This package relies heavily on Nilearn for the underlying computation.
   Without their great work this would not be possible.
   See here for how to accurately cite Nilearn:
   http://nilearn.github.io/authors.html#citing

The MNE-NIRS GLM analysis framework is entirely based on the Nilearn package.
Their excellent software forms the basis of the analysis described in this tutorial.
As such, you may also wish to read their documentation to familiarise yourself with
different concepts used here.
Specifically this tutorial is heavily based on the following Nilearn examples,
but placed within an fNIRS context.

* http://nilearn.github.io/auto_examples/04_glm_first_level/plot_first_level_details.html
* https://nilearn.github.io/auto_examples/04_glm_first_level/plot_hrf.html

Accordingly we will access nilearn functions directly in this tutorial to illustrate
various choices available in your analysis.
However, this is just to illustrate various points. In reality (see all other tutorials),
MNE-NIRS will wrap all required Nilearn functions so you don't need to access them directly.


.. contents:: Page contents
   :local:
   :depth: 2

"""
# sphinx_gallery_thumbnail_number = 1

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


# Import common libraries
import os
import numpy as np
import mne

# Import MNE-NIRS processing
from mne_nirs.experimental_design import make_first_level_design_matrix, \
    longest_ISI, drift_high_pass

# Import Nilearn
from nilearn.glm import first_level
from nilearn.plotting import plot_design_matrix

# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl


# %%
# Haemodynamic Response Function
# ---------------------------------------------------------------------
#
# Various Haemodynamic Response Functions (HRFs) are provided for use
# when analysing your data. A summary of these functions in the context
# of fMRI is provided in the Nilearn tutorial
# https://nilearn.github.io/auto_examples/04_glm_first_level/plot_hrf.html.
# This example heavily borrows from that example but expands the description
# within an fNIRS context.
#
# To illustrate underlying concepts we will use Nilearn functions here directly,
# but for analysing actual data you should use the MNE-NIRS
# :func:`mne_nirs.experimental_design.make_first_level_design_matrix`
# wrapper.


# %%
# HRF Model Selection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Two standard HRF models are provided. The SPM and Glover models.
# These differ in their response dynamics.
# Both are plotted on top of each other below for comparison.

time_length = 30

glover_timecourse = first_level.glover_hrf(1, oversampling=50, time_length=time_length)
spm_timecourse = first_level.spm_hrf(1, oversampling=50, time_length=time_length)

sample_times = np.linspace(0, time_length, num=len(glover_timecourse))

plt.plot(sample_times, glover_timecourse, label="Glover")
plt.plot(sample_times, spm_timecourse, label="SPM")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (AU)")
plt.legend()


# %%
# Regressor Computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These functions alone are not used directly in the GLM analysis.
# Instead they are used as the basis to compute a regressor which is
# utilised in the GLM fit.
# This is done by convolving the HRF model with a boxcar function that
# distills information known
# about the experimental design. Specifically the stimulus onset times
# are used to indicate when a response begins, and a duration is used
# to specify the time over which the model should be convolved.
#
# Modifying the duration changes the regressor shape. Below we demonstrate
# how this varies for several duration values with the Glover HRF.

# Generate an event of 1 second duration that occurs at time zero.
onset, amplitude, duration = 0.0, 1.0, 1.0
hrf_model = "glover"


def generate_stim(onset, amplitude, duration, hrf_model, maxtime=30):

    # Generate signal with specified duration and onset
    frame_times = np.linspace(0, maxtime, 601)
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    stim = np.zeros_like(frame_times)
    stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude

    signal, name = first_level.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id="main", oversampling=16
    )

    return frame_times, stim, signal


def plot_regressor(onset, amplitude, duration, hrf_model):
    frame_times, stim, signal = generate_stim(
        onset, amplitude, duration, hrf_model)
    plt.fill(frame_times, stim, "k", alpha=0.5, label="stimulus")
    plt.plot(frame_times, signal.T[0], label="Regressor")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (AU)")
    plt.legend(loc=1)
    plt.title(hrf_model)

    return 1


plot_regressor(onset, amplitude, duration, hrf_model)


# %%
#
# If the duration is increased we see the resulting regressor
# is modified, and the transformation is not a simple scaling.

duration = 15
plot_regressor(onset, amplitude, duration, hrf_model)


# %%
#
# We can plot multiple durations together to see how the
# resulting regressor varies as a function of this parameter.

cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=40)

for n in [1, 3, 5, 10, 15, 20, 25, 30, 35]:
    frame_times, stim, signal = generate_stim(
        onset, amplitude, n, hrf_model, maxtime=50)
    plt.plot(frame_times, signal.T[0], label="Regressor", c=cmap(norm(n)))

plt.xlabel("Time (s)")
plt.ylabel("Amplitude (AU)")
plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))


# %%
# Inclusion in Design matrix
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# As mentioned above, we don't directly compute these regressors for
# each condition. Instead the function :func:`mne_nirs.experimental_design.make_first_level_design_matrix`
# conveniently does this for us.
#
# As an example we will import a measurement and generate a
# design matrix for it. We will specify that we wish to use a Glover
# HRF convolved with a 3 second duration.
# This is similar to the first level GLM example, see there for more details.
#
# First we import the example data, crop to just the firs few minutes,
# and give names to the annotations.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data().crop(tmax=300)
# raw_intensity.resample(0.7)
raw_intensity.annotations.rename({'1.0': 'Control',
                                  '2.0': 'Tapping/Left',
                                  '3.0': 'Tapping/Right'})
raw_intensity.annotations.delete(raw_intensity.annotations.description == '15.0')
raw_intensity.annotations.set_durations(5)


# %%
#
# Next we compute the design matrix and plot it.
# This representation of the regressor is transposed, as in time goes down the vertical
# axis and is specified in scan number (fMRI hangover) or sample.
# There is no colorbar for this plot, as specified in Nilearn.
#
# We can see that when each event occurs the model value increases before returning to baseline.
# this is the same information as was shown in the time courses above, except displayed differently
# with color representing amplitude.

design_matrix = make_first_level_design_matrix(raw_intensity,
                                               # Ignore drift model for now, see section below
                                               drift_model='polynomial',
                                               drift_order=0,
                                               # Here we specify the HRF and duration
                                               hrf_model='glover',
                                               stim_dur=3.0)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


# %%
#
# As before we can explore the effect of modifying the duration,
# the resulting regressor for each annotation is elongated.

design_matrix = make_first_level_design_matrix(raw_intensity,
                                               # Ignore drift model for now, see section below
                                               drift_model='polynomial',
                                               drift_order=0,
                                               # Here we specify the HRF and duration
                                               hrf_model='glover',
                                               stim_dur=13.0)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


# %%
# Drift Regressors
# ---------------------------------------------------------------------
#
# Aspects of the measured signal may change in over time in an manner
# unrelated to the neural response we wish to measure.
# For example, the measurement room may warm up and result in a steady
# increase in the signal over the measurement duration.
# These signal changes that are unrelated to our feature of interest are
# termed drifts, and can be included in the design matrix and the GLM
# fitting as drift regressors.
#
# In the examples above a single drift regressor was used to model a constant
# offset in the data. This is also termed a zero order polynomial regressor.
# Two types of regressors are provided for in MNE-NIRS thanks to Nilearn.
# Polynomial  and cosine drift regressors.


# %%
# Polynomial Drift Regressors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In the example above a polynomial drift regressor is included.
# In this case we can specify the order of the polynomils to be included.
# A zero order polynomial will fit a constant, a first order will fit an
# increasing function, and so on.
# As an example we demonstrate how to include up to a fifth order polynomial.
# You can observe that with increasing polynomial order,
# higher frequency components will be removed from the signal.

design_matrix = make_first_level_design_matrix(raw_intensity,
                                               drift_model='polynomial',
                                               drift_order=5)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


# %%
# Cosine Drift Regressors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An alternative way to specify drift regressors is via the cosine drift model.
# This may be more intuitive as you can specify regressors up to a certain cutoff
# frequency. Effectively regressing out frequency components below a limit,
# which may be interpreted as a high pass filter.
# In the example below we demonstrate how to regress our signals up to 0.01 Hz.
# We observe that the function has included 6 drift regressors in the design matrix.

design_matrix = make_first_level_design_matrix(raw_intensity,
                                               drift_model='cosine',
                                               high_pass=0.01)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


# %%
#
# As described above, including additional regressor components will remove
# higher frequency components. So we can increase the high pass cut off and
# this should add more regressors.

design_matrix = make_first_level_design_matrix(raw_intensity,
                                               drift_model='cosine',
                                               high_pass=0.03)

fig, ax1 = plt.subplots(figsize=(10, 6), nrows=1, ncols=1)
fig = plot_design_matrix(design_matrix, ax=ax1)


# %%
# Selecting Drift Regressors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The aim of the drift regressors is to remove signal components unrelated
# to the expected neural response. As the expected response can be computed
# based on annotation timing and expected brain responses
# (see :ref:`frequency commentary <tut-fnirs-freq>`)
# the high pass cut off can be set on first principles.
#
# The Nilearn documentation states that
# "The cutoff period (1/high_pass) should be set as the longest period between two trials of the same condition multiplied by 2.
# For instance, if the longest period is 32s, the high_pass frequency shall be 1/64 Hz ~ 0.016 Hz."
# `(reference) <http://nilearn.github.io/auto_examples/04_glm_first_level/plot_first_level_details.html#changing-the-drift-model>`__.
#
# To assist in selecting a high pass value a few convenience functions are included in MNE-NIRS.
# First we can query what the longest ISI is per annotation, but first we must be sure
# to remove annotations we arent interested in (in this experiment the trigger
# 15 is not of interest).

raw_original = mne.io.read_raw_nirx(fnirs_raw_dir)
raw_original.annotations.delete(raw_original.annotations.description == '15.0')

isis = longest_ISI(raw_original)
print(isis)


# %%
#
# We see that the longest period between two trials is 435 seconds. Which multiplied
# by two is 870 seconds. So aa high pass value of 1/870 or 0.001 Hz is appropriate.
# We can also use the function
# :func:`mne_nirs.experimental_design.make_first_level_design_matrix`
# to suggest the high pass value. Note however, that you should not blindly follow
# this functions suggestion, as each experiment is different. Instead use this as
# a sanity check on your own calculations.

print(drift_high_pass(raw_original))


# %%
#
# For example, if all conditions were evoking the same response it may make more
# sense to include them as a single condition when computing the ISI.
# This would be achieved by renaming the triggers.

raw_original.annotations.rename({'2.0': 'Tapping',
                                 '3.0': 'Tapping'})
raw_original.annotations.delete(raw_original.annotations.description == '1.0')
isis = longest_ISI(raw_original)
print(isis)
print(drift_high_pass(raw_original))
