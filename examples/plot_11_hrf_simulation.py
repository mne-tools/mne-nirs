"""
.. _tut-fnirs-hrf-sim:

Simulated HRF Analysis
========================================

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

import mne
import mne_nirs
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
import numpy as np
import os
from mne_nirs.experimental_design import create_first_level_design_matrix
from mne_nirs.statistics import run_GLM
from mne_nirs.visualisation import plot_GLM_topo
from nilearn.reporting import plot_design_matrix


###############################################################################
# Simulate raw NIRS data
# ----------------------
#
# Some text.

fs = 3.
amplitude = 4.

raw = mne_nirs.simulation.simulate_nirs_data(fs=fs, signal_length_s=60*5,
                                             amplitude=amplitude)

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
# Run GLM estimate
# ----------------
#
# Some text.

labels, glm_estimates = run_GLM(raw, design_matrix)

print(glm_estimates[labels[0]].theta)


###############################################################################
# Calculate error
# ---------------
#
# Some text.

error = glm_estimates[labels[0]].theta[0] - amplitude * 1.e-6

print(error)
