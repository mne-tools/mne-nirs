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
import numpy as np
import pandas as pd

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# Import MNE-NIRS processing
from mne_nirs.statistics import run_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.channels import get_short_channels, get_long_channels

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids

# Import Nilearn
from nilearn.glm import first_level

# Import Plotting Library
import matplotlib.pyplot as plt


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
plt.show()


# %%
# Regressor Computation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These functions alone are not used directly in the GLM analysis.
# Instead they are used as the basis to compute a regressor which is
# utilised in the GLM fit.
# This is done by convolving the HRF model with a information known
# about the experimental design. Specifically the stimulus onset times
# are used to indicate when a response begins, and a duration is used
# to specify the time over which the model should be convolved.
#
# Modifying the duration changes the regressor shape. Below we demonstrate
# how this varies for several duration values with the Glover HRF.

# Generate an event of 1 second duration that occurs at time zero.
onset, amplitude, duration = 0., 1., 1.
hrf_model = 'glover'

def generate_stim(onset, amplitude, duration, hrf_model):

    # Generate signal with specified duration and onset
    frame_times = np.linspace(0, 30, 601)
    exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
    stim = np.zeros_like(frame_times)
    stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude

    signal, name = first_level.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16)

    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    plt.plot(frame_times, signal.T[0], label="Regressor")
    plt.xlabel('Time (s)')
    plt.ylabel("Amplitude (AU)")
    plt.legend(loc=1)
    plt.title(hrf_model)

    return 1


generate_stim(onset, amplitude, duration, hrf_model)
plt.show()


# %%
#
# If the duration is increased we see the resulting regressor
# is modified.

duration = 5

generate_stim(onset, amplitude, duration, hrf_model)
plt.show()

# %%
#
# And even further

duration = 15

generate_stim(onset, amplitude, duration, hrf_model)
plt.show()