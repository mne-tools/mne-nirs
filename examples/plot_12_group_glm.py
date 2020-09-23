"""
.. _tut-fnirs-group:

Group Level GLM
===============

.. sidebar:: Relevant literature

   Gorgolewski, Krzysztof J., et al.
   "The brain imaging data structure, a format for organizing and describing
   outputs of neuroimaging experiments." Scientific data 3.1 (2016): 1-9.

   Santosa, H., Zhai, X., Fishburn, F., & Huppert, T. (2018).
   The NIRS brain AnalyzIR toolbox. Algorithms, 11(5), 73.

This is an example of a group level GLM based fNIRS analysis in MNE.

Individual level example analysis of this data is described in the
`MNE fNIRS waveform tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_
and the
`MNE-NIRS fNIRS GLM tutorial <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_.
So this example will skim over the individual level details
and focus on the group level aspect of analysis.
Instead here we describe how to process multiple measurements,
and summarise the group level effects as summary statistics and visually.

The data used in this example is available `at this location <https://github.com/rob-luke/BIDS-NIRS-Tapping>`_.
It is an finger tapping example and is briefly described below.
The dataset contains 5 participants.
The example dataset is in
`BIDS <https://bids.neuroimaging.io>`_
format and therefore already contains
information about triggers, condition names, etc.

Individual level example analysis of this data is described in the
`MNE fNIRS waveform tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_
and the
`MNE-NIRS fNIRS GLM tutorial <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_.
So this example will skim over the individual level details
and focus on the group level aspect of analysis.

.. collapse:: Data description (click to expand)
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

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


# Import common libraries
import numpy as np
import pandas as pd

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# Import MNE-NIRS processing
from mne_nirs.statistics import run_GLM
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import glm_region_of_interest, statsmodels_to_results, compute_contrast
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.channels import picks_pair_to_idx as pair_to_idx
from mne_nirs.utils._io import glm_to_tidy, _tidy_long_to_wide
from mne_nirs.visualisation import plot_glm_group_topo

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids

# Import StatsModels
<<<<<<< HEAD
=======
from statsmodels.formula.api import rlm
>>>>>>> 2b1db77... Simplify robust models
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
#    `MNE docs <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_.
#
#    GLM individual analysis:
#    `MNE-NIRS docs <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_.
#
# First we define the analysis that will be applied to each file.
# This is a GLM analysis as described in the
<<<<<<< HEAD
# `MNE-NIRS fNIRS GLM tutorial <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_,
# so this example will skim over the individual level details.
#
# The analysis extracts a response estimate for each channel,
# each region of interest, and computes a contrast between left and right
# finger tapping.
# We return the raw object, and data frames for the computed results.
# Information about channels, triggers and their meanings are stored in the
# BIDS structure, so are automatically obtained when importing the data.
=======
# `MNE-NIRS fNIRS GLM tutorial <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_
# so this example will skim over the individual level details.
#
# The analysis extracts a response estimate for each region of interest and
# each condition and returns the results as a dataframe with the participant
# ID.
>>>>>>> 2b1db77... Simplify robust models

def individual_analysis(bids_path, ID):

    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)

    # Convert signal to haemoglobin and resample
    raw_od = optical_density(raw_intensity)
    raw_haemo = beer_lambert_law(raw_od)
    raw_haemo.resample(0.6)

    # Cut out just the short channels for creating a GLM repressor
    sht_chans = get_short_channels(raw_haemo)

    raw_haemo = get_long_channels(raw_haemo)

    # Create a design matrix
    design_matrix = make_first_level_design_matrix(raw_haemo, stim_dur=5.0)

    # Append short channels mean to design matrix
    design_matrix["ShortHbO"] = np.mean(sht_chans.copy().pick(picks="hbo").get_data(), axis=0)
    design_matrix["ShortHbR"] = np.mean(sht_chans.copy().pick(picks="hbr").get_data(), axis=0)


    # Run GLM
    glm_est = run_GLM(raw_haemo, design_matrix)

    # Define channels in each region of interest
    # List the channel pairs manually
    left = [[4, 3], [1, 3], [3, 3], [1, 2], [2, 3], [1, 1]]
    right = [[6, 7], [5, 7], [7, 7], [5, 6], [6, 7], [5, 5]]

    # Then generate the correct indices for each pair
    groups = dict(
        Left_Hemisphere=pair_to_idx(raw_haemo, left, on_missing='ignore'),
        Right_Hemisphere=pair_to_idx(raw_haemo, right, on_missing='ignore'))

    # Extract channel metrics
    cha = glm_to_tidy(raw_haemo, glm_est, design_matrix)
    cha = _tidy_long_to_wide(cha)
    cha["ID"] = ID  # Add the participant ID to the dataframe
<<<<<<< HEAD

=======
>>>>>>> 2b1db77... Simplify robust models

    # Compute region of interest results from channel data
    roi = pd.DataFrame()
    for idx, col in enumerate(design_matrix.columns):
        roi = roi.append(glm_region_of_interest(glm_est, groups, idx, col))
    roi["ID"] = ID  # Add the participant ID to the dataframe
<<<<<<< HEAD

    # Contrast left vs right tapping
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['Tapping/Left'] - basic_conts['Tapping/Right']
    contrast = compute_contrast(glm_est, contrast_LvR)
    con = _tidy_long_to_wide(glm_to_tidy(raw_haemo, contrast, design_matrix))
    con["ID"] = ID  # Add the participant ID to the dataframe

    # Convert to uM for nicer plotting below.
    cha["theta"] = [t * 1.e6 for t in cha["theta"]]
    roi["theta"] = [t * 1.e6 for t in roi["theta"]]
    con["effect"] = [t * 1.e6 for t in con["effect"]]
=======
>>>>>>> 2b1db77... Simplify robust models

    return raw_haemo, roi, cha, con


###############################################################################
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
<<<<<<< HEAD
# for both the region of interest, channel level, and contrast results.
=======
# for both the region of interest and channel level results.
>>>>>>> 2b1db77... Simplify robust models

df_roi = pd.DataFrame()  # To store region of interest results
df_cha = pd.DataFrame()  # To store channel level results
df_con = pd.DataFrame()  # To store channel level contrast results

for sub in range(1, 6):  # Loop from first to fifth subject
    ID = '%02d' % sub  # Tidy the subject name

    # Create path to file based on experiment info
    bids_path = BIDSPath(subject=ID, task="tapping", root='BIDS-NIRS-Tapping',
                         datatype="nirs", suffix="nirs", extension=".snirf")

    # Analyse data and return both ROI and channel results
    raw_haemo, roi, channel, con = individual_analysis(bids_path, ID)

    # Append individual results to all participants
    df_roi = df_roi.append(roi)
    df_cha = df_cha.append(channel)
    df_con = df_con.append(con)


###############################################################################
# Visualise Individual results
# ----------------------------
#
# First we visualise the results from each individual to ensure the
# data values look reasonable.
# Here we see that we have data from five participants, we plot just the HbO
# values and observe they are in the expect range.
# We can already see that the control condition is always near zero,
# and that the responses look to be contralateral to the tapping hand.

grp_results = df_roi.query("Condition in ['Control', 'Tapping/Left', 'Tapping/Right']")
grp_results = grp_results.query("Chroma in ['hbo']")

ggplot(grp_results, aes(x='Condition', y='theta', color='ROI', shape='ROI')) \
    + geom_hline(y_intercept=0, linetype="dashed", size=1) \
    + geom_point(size=5) \
    + facet_grid('ID') \
    + ggsize(800, 300)


###############################################################################
# Compute group level results
# ---------------------------
#
# .. sidebar:: Relevant literature
#
#    For an introduction to mixed effects analysis see:
#    Winter, Bodo. "A very basic tutorial for performing linear mixed effects
#    analyses." arXiv preprint arXiv:1308.5499 (2013).
#
#    For a summary of linear mixed models in python
#    and the relation to lmer see:
#    `statsmodels docs <https://www.statsmodels.org/stable/mixed_linear.html>`_.
#
#    For a summary of these models in the context of fNIRS see section 3.5 of:
#    Santosa, H., Zhai, X., Fishburn, F., & Huppert, T. (2018).
#    The NIRS brain AnalyzIR toolbox. Algorithms, 11(5), 73.
#
# Next we use a linear mixed effects model to understand the
# relation between conditions and our response estimate (theta).
# Combinations of 3 fixed effects will be evaluated, ROI (left vs right),
# condition (control, tapping/left, tapping/right), and chromophore (HbO, HbR).
# With a random effect of subject.
# Alternatively, you could export the dataframe `df_roi.to_csv()` and
# analyse in your favorite stats program.
#
# Here we examine the effect of ROI, condition and chroma,
# controlling for participant. Alternatively, we could use a robust linear
# model by using the code
# `roi_model = rlm('theta ~ -1 + ROI:Condition:Chroma', grp_results).fit()`.
#
# We do not explore the modeling procedure in depth here as topics
# such model selection and examining residuals are beyond the scope of
# this example (see relevant literature).

grp_results = df_roi.query("Condition in ['Control','Tapping/Left', 'Tapping/Right']")

roi_model = smf.mixedlm("theta ~ -1 + ROI:Condition:Chroma",
                        grp_results, groups=grp_results["ID"]).fit(method='nm')

roi_model.summary()


###############################################################################
# Visualise group results
# -----------------------
#
# Now we can summarise the output of the second level model.
# This figure shows that control condition has small respones that
# are not significant for both HbO and HbR in both hemispheres.
# Whereas clear significant responses are show for the two tapping conditions.
# We also observe the it is the contralateral hemisphere that has the
# larger response for each tapping condition.
# Filled symbols represent HbO, unfilled symbols represent HbR.

df = statsmodels_to_results(roi_model)

ggplot(df.query("Chroma == 'hbo'"),
       aes(x='Condition', y='coef', color='sig', shape='ROI')) \
    + geom_hline(y_intercept=0, linetype="dashed", size=1) \
    + geom_point(size=5) \
    + scale_shape_manual(values=[16, 17]) \
    + ggsize(800, 300) \
    + geom_point(data=df.query("Chroma == 'hbr'")
                 .query("ROI == 'Left_Hemisphere'"), size=5, shape=1) \
    + geom_point(data=df.query("Chroma == 'hbr'")
                 .query("ROI == 'Right_Hemisphere'"), size=5, shape=2)


###############################################################################
# Group topographic visualisation
# -------------------------------
#
# We can also view the topographic representation of the data
# (rather than the ROI summary above).
# Here we just plot the oxyhaemoglobin for the two tapping conditions.
# First we compute the mixed effects model, but for each channel rather
# than region of interest. Then we pass these results to the topo function.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         gridspec_kw=dict(width_ratios=[1, 1]))

# Cut down the dataframe just to the conditions we are interested in
ch_summary = df_cha.query("condition in ['Tapping/Left', 'Tapping/Right']")
ch_summary = ch_summary.query("Chroma in ['hbo']")

# Run group level model and convert to dataframe
ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:condition",
                       ch_summary, groups=ch_summary["ID"]).fit(method='nm')
ch_model_df = statsmodels_to_results(ch_model)

# Plot the two conditions
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    ch_model_df.query("condition in ['Tapping/Left']"),
                    colorbar=False, axes=axes[0, 0],
                    vmin=0, vmax=20, cmap=mpl.cm.Oranges)
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    ch_model_df.query("condition in ['Tapping/Right']"),
                    colorbar=True, axes=axes[0, 1],
                    vmin=0, vmax=20, cmap=mpl.cm.Oranges)

# Cut down the dataframe just to the conditions we are interested in
ch_summary = df_cha.query("condition in ['Tapping/Left', 'Tapping/Right']")
ch_summary = ch_summary.query("Chroma in ['hbr']")

# Run group level model and convert to dataframe
ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:condition",
                       ch_summary, groups=ch_summary["ID"]).fit(method='nm')
ch_model_df = statsmodels_to_results(ch_model)

# Plot the two conditions
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbr"),
                    ch_model_df.query("condition in ['Tapping/Left']"),
                    colorbar=False, axes=axes[1, 0],
                    vmin=-10, vmax=0, cmap=mpl.cm.Blues_r)
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbr"),
                    ch_model_df.query("condition in ['Tapping/Right']"),
                    colorbar=True, axes=axes[1, 1],
                    vmin=-10, vmax=0, cmap=mpl.cm.Blues_r)



###############################################################################
# Contrasts
# ---------
#
# Finally we can view the contrast results in a topographic representation.
# We can also mark the channels the vary significantly between conditions
# with a black mark.

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
con_summary = df_con.query("Chroma in ['hbo']")

# Run group level model and convert to dataframe
con_model = smf.mixedlm("effect ~ -1 + ch_name:Chroma",
                        con_summary, groups=con_summary["ID"]).fit(method='nm')
con_model_df = statsmodels_to_results(con_model)

plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    con_model_df, colorbar=True, axes=axes)

# Mark significantly varying channels (uncomment to run)
# raw_haemo.copy().pick(picks="hbo").pick(picks=list(
#     con_model_df.query("sig == True")["ch_name"])).plot_sensors(axes=axes)
