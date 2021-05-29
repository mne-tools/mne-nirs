"""
.. _tut-fnirs-group-wave:

Group Level Waveform
====================

This is an example of a group level waveform based
functional near-infrared spectroscopy (fNIRS)
analysis in MNE-NIRS.

.. sidebar:: Relevant literature

   Gorgolewski, Krzysztof J., et al.
   "The brain imaging data structure, a format for organizing and describing
   outputs of neuroimaging experiments." Scientific data 3.1 (2016): 1-9.

   Santosa, H., Zhai, X., Fishburn, F., & Huppert, T. (2018).
   The NIRS brain AnalyzIR toolbox. Algorithms, 11(5), 73.

Individual level analysis of this data is described in the
:ref:`MNE-NIRS fNIRS waveform tutorial <tut-fnirs-processing>`
and the
:ref:`MNE-NIRS fNIRS GLM tutorial <tut-fnirs-hrf>`
So this example will skim over the individual level details
and focus on the group level aspect of analysis.
Here we describe how to process multiple measurements
and summarise  group level effects both as summary statistics and visually.

The data used in this example is available
`at this location <https://github.com/rob-luke/BIDS-NIRS-Tapping>`_.
It is a finger tapping example and is briefly described below.
The dataset contains 5 participants.
The example dataset is in
`BIDS <https://bids.neuroimaging.io>`_
format and therefore already contains
information about triggers, condition names, etc.
The BIDS specification for NIRS data is still under development,
as such you must use the development branch of MNE-BIDS as listed in the
requirements_doc.txt file to run this example.

.. collapse:: |chevron-circle-down| Data description (click to expand)
   :class: success

   Optodes were placed over the motor cortex using the standard NIRX motor
   montage, but with 8 short channels added (see their web page for details).
   To view the sensor locations run
   `raw_intensity.plot_sensors()`.
   A sound was presented to indicate which hand the participant should tap.
   Participants tapped their thumb to their fingers for 5s.
   Conditions were presented in a random order with a randomised inter
   stimulus interval.

.. contents:: Page contents
   :local:
   :depth: 2
"""
# sphinx_gallery_thumbnail_number = 2

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


# Import common libraries
import numpy as np
import pandas as pd
from itertools import compress
from collections import defaultdict

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.viz import plot_compare_evokeds

# Import MNE-NIRS processing
from mne import Epochs, events_from_annotations
from mne_nirs.statistics import run_GLM
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import glm_region_of_interest, statsmodels_to_results
from mne_nirs.statistics import compute_contrast
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.channels import picks_pair_to_idx
from mne_nirs.utils._io import glm_to_tidy
from mne_nirs.visualisation import plot_glm_group_topo
from mne_nirs.datasets import fnirs_motor_group
from mne.preprocessing.nirs import (
    beer_lambert_law,
    temporal_derivative_distribution_repair,
    optical_density,
    scalp_coupling_index,
)
from mne_nirs.signal_enhancement import (
    enhance_negative_correlation,
    short_channel_regression,
)

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl
from lets_plot import *

LetsPlot.setup_html()


###############################################################################
# Define individual analysis
# --------------------------
#
# .. sidebar:: Individual analysis procedures
#
#    Waveform individual analysis:
#    :ref:`MNE docs <mne:tut-fnirs-processing>`
#
#    GLM individual analysis:
#    :ref:`MNE-NIRS docs <tut-fnirs-hrf>`
#
# First we define the analysis that will be applied to each file.
# This is a waveform analysis as described in the
# :ref:`individual waveform tutorial <tut-fnirs-processing>`,
# so this example will skim over the individual level details.
#
# Here we also resample to a 0.3 Hz sample rate just to speed up the example
# and use less memory, resampling to 0.6 Hz is a better choice for full
# analyses.


def individual_analysis(bids_path):

    # Read data with annotations in BIDS format
    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)

    # Convert signal to optical density and determine bad channels
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od, h_freq=1.35, h_trans_bandwidth=0.1)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.8))

    # Interpolate bad channels based on neighbouring data and downample
    raw_od.interpolate_bads()
    raw_od.resample(0.8)

    # Apply signal cleaning techniques
    raw_od = temporal_derivative_distribution_repair(raw_od)
    raw_od = short_channel_regression(raw_od)

    # Convert to haemoglobin and filter
    raw_haemo = beer_lambert_law(raw_od)
    raw_haemo = raw_haemo.filter(0.05, 0.3,
                                 h_trans_bandwidth=0.1, l_trans_bandwidth=0.01,
                                 verbose=False)

    # Apply further data cleaning techniques and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)
    raw_haemo = get_long_channels(raw_haemo, min_dist=0.01, max_dist=0.05)
    events, event_dict = events_from_annotations(raw_haemo)
    epochs = Epochs(raw_haemo, events, event_id=event_dict,
        tmin=-5, tmax=15,
        reject=dict(hbo=800e-6), reject_by_annotation=True,
        proj=True, baseline=(None, 0), detrend=None,
        preload=True, verbose=False)

    return raw_haemo, epochs


###############################################################################
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
# for the region of interest, channel level, and contrast results.

all_evokeds = defaultdict(list)

for sub in range(1, 6):  # Loop from first to fifth subject

    # Create path to file based on experiment info
    bids_path = BIDSPath(subject="%02d" % sub, task="tapping", datatype="nirs",
                         root=fnirs_motor_group.data_path(), suffix="nirs",
                         extension=".snirf")

    # Analyse data and return both ROI and channel results
    raw_haemo, epochs = individual_analysis(bids_path)

    for cidx, condition in enumerate(epochs.event_id):
        all_evokeds[condition].append(epochs[condition].average())

all_evokeds

###############################################################################
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
# for the region of interest, channel level, and contrast results.

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(17, 5))
lims = dict(hbo=[-2, 3], hbr=[-2, 3])
ci = 0.95
for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for idx, evoked in enumerate(all_evokeds):
        plot_compare_evokeds({evoked: all_evokeds[evoked]}, combine='mean',
                             picks=pick, axes=axes[idx], show=False,
                             colors=[color], legend=False, ylim=lims, ci=0.95)
        axes[idx].set_title('{}'.format(evoked))
