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
import pandas as pd
from itertools import compress
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

# Import MNE processing
from mne.viz import plot_compare_evokeds
from mne import Epochs, events_from_annotations

# Import MNE-NIRS processing
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.channels import get_long_channels
from mne_nirs.channels import picks_pair_to_idx
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
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    raw_od.interpolate_bads()

    # Downsample and apply signal cleaning techniques
    raw_od.resample(0.8)
    raw_od = temporal_derivative_distribution_repair(raw_od)
    raw_od = short_channel_regression(raw_od)

    # Convert to haemoglobin and filter
    raw_haemo = beer_lambert_law(raw_od)
    raw_haemo = raw_haemo.filter(0.02, 0.3,
                                 h_trans_bandwidth=0.1, l_trans_bandwidth=0.01,
                                 verbose=False)

    # Apply further data cleaning techniques and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)
    raw_haemo = get_long_channels(raw_haemo, min_dist=0.01, max_dist=0.05)

    # Extract events but ignore those with
    # the word ends (i.e. drop ExperimentEnds events)
    events, event_dict = events_from_annotations(raw_haemo, verbose=False,
                                                 regexp='^(?![Ends]).*$')
    epochs = Epochs(raw_haemo, events, event_id=event_dict,
        tmin=-5, tmax=20,
        reject=dict(hbo=200e-6), reject_by_annotation=True,
        proj=True, baseline=(None, 0), detrend=0,
        preload=True, verbose=False)

    return raw_haemo, epochs


###############################################################################
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. For each individual the function returns the raw data and an
# epoch structure. The epoch structure is then averaged to obtain an evoked
# response per participant. Each condition is then added to list per condition.

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


###############################################################################
# The end result is a dictionary indexed per condition.
# With each item in the dictionary being a list of evoked responses.
# See below that for each condition we have obtained an MNE evoked type
# that is generated from the average of 30 trials and epoched from -5 to
# 20 seconds.

pprint(all_evokeds)

###############################################################################
# View average waveform
# ---------------------
#
# Next we generate a grand average epoch waveform per condition.
# This is generated using all long fNIRS channels.

# Specify the figure size and limits per chromophore.
fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds), figsize=(17, 5))
lims = dict(hbo=[-5, 12], hbr=[-5, 12])


for (pick, color) in zip(['hbo', 'hbr'], ['r', 'b']):
    for idx, evoked in enumerate(all_evokeds):
        plot_compare_evokeds({evoked: all_evokeds[evoked]}, combine='mean',
                             picks=pick, axes=axes[idx], show=False,
                             colors=[color], legend=False, ylim=lims, ci=0.95)
        axes[idx].set_title('{}'.format(evoked))
axes[0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])


###############################################################################
# Generate regions of interest
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that

# Specify channel pairs for each ROI
left = [[4, 3], [1, 3], [3, 3], [1, 2], [2, 3], [1, 1]]
right = [[8, 7], [5, 7], [7, 7], [5, 6], [6, 7], [5, 5]]

# Then generate the correct indices for each pair
groups = dict(
    Left_Hemisphere=picks_pair_to_idx(raw_haemo, left, on_missing='ignore'),
    Right_Hemisphere=picks_pair_to_idx(raw_haemo, right, on_missing='ignore'))


###############################################################################
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
# for the region of interest, channel level, and contrast results.

df = pd.DataFrame(columns=['ID', 'ROI', 'Chroma', 'Condition', 'Value'])

for idx, evoked in enumerate(all_evokeds):
    for subject_data in all_evokeds[evoked]:
        for roi in groups:
            for chroma in ["hbo", "hbr"]:
                sub_id = subject_data.info["subject_info"]['first_name']
                data = deepcopy(subject_data).pick(picks=groups[roi]).pick(chroma)
                mean_value = data.crop(tmin=5.0, tmax=7.0).data.mean() * 1.0e6

                df = df.append({'ID': sub_id, 'ROI': roi, 'Chroma': chroma,
                                'Condition': evoked, 'Value': mean_value},
                               ignore_index=True)

# You can export the dataframe for analyis in your favorite stats program
df.to_csv("stats-export.csv")

# Print out the first entries in the dataframe
df.head()


###############################################################################
# View individual results
# -----------------------
#
# In this example question we ask: is the hbo response to tapping with the

ggplot(df.query("Chroma == 'hbo'"),
       aes(x='Condition', y='Value', color='ID', shape='ROI')) \
    + geom_hline(y_intercept=0, linetype="dashed", size=1) \
    + geom_point(size=5) \
    + scale_shape_manual(values=[16, 17]) \
    + ggsize(800, 300)


###############################################################################
# Research question 1: Comparison of conditions
# ---------------------------------------------------------------------------------------------------
#
# In this example question we ask: is the hbo response to tapping with the
# right hand larger than the response when not tapping in the left ROI?
# For this token example we subset the dataframe then apply the mixed
# effect model.

input_data = df.query("Condition in ['Control', 'Tapping/Right']")
input_data = input_data.query("Chroma in ['hbo']")
input_data = input_data.query("ROI in ['Left_Hemisphere']")

roi_model = smf.mixedlm("Value ~ Condition", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()

###############################################################################
# And the model indicates that for the oxyhaemoglobin data in the left
# region of interest, that the tapping condition with the right hand evokes
# a larger response than the control.


###############################################################################
# Research question 2: Are responses larger on the contralateral side to tapping?
# -------------------------------------------------------------------------------
#
# In this example question we ask: is the hbo response to tapping with the
# right hand larger than the response when not tapping in the left ROI?
# For this token example we subset the dataframe then apply the mixed
# effect model.

# Encode the ROIs as ipsi- or contralateral to the hand that is tapping.
df["Hemishphere"] = "Unknown"
df.loc[(df["Condition"] == "Tapping/Right") & (df["ROI"] == "Right_Hemisphere"), "Hemishphere"] = "Ipsilateral"
df.loc[(df["Condition"] == "Tapping/Right") & (df["ROI"] == "Left_Hemisphere"), "Hemishphere"] = "Contralateral"
df.loc[(df["Condition"] == "Tapping/Left") & (df["ROI"] == "Left_Hemisphere"), "Hemishphere"] = "Ipsilateral"
df.loc[(df["Condition"] == "Tapping/Left") & (df["ROI"] == "Right_Hemisphere"), "Hemishphere"] = "Contralateral"

# Subset the data for example model
input_data = df.query("Condition in ['Tapping/Right', 'Tapping/Left']")
input_data = input_data.query("Chroma in ['hbo']")

roi_model = smf.mixedlm("Value ~ Hemishphere", input_data,
                        groups=input_data["ID"]).fit()
roi_model.summary()

###############################################################################
# And the model indicates that for the oxyhaemoglobin data that larger
# responses are evoked on the contralateral side to the hand that is tapping
# compared to the ipsilateral side.