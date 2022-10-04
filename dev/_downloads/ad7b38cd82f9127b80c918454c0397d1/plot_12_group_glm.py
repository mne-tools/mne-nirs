"""
.. _tut-fnirs-group:

Group Level GLM Analysis
========================

This is an example of a group level GLM based
functional near-infrared spectroscopy (fNIRS)
analysis in MNE-NIRS.

.. sidebar:: Relevant literature

   Luke, Robert, et al.
   "Analysis methods for measuring passive auditory fNIRS responses generated
   by a block-design paradigm." Neurophotonics 8.2 (2021):
   `025008 <https://www.spiedigitallibrary.org/journals/neurophotonics/volume-8/issue-2/025008/Analysis-methods-for-measuring-passive-auditory-fNIRS-responses-generated-by/10.1117/1.NPh.8.2.025008.short>`_.

   Santosa, H., Zhai, X., Fishburn, F., & Huppert, T. (2018).
   The NIRS brain AnalyzIR toolbox. Algorithms, 11(5), 73.

   Gorgolewski, Krzysztof J., et al.
   "The brain imaging data structure, a format for organizing and describing
   outputs of neuroimaging experiments." Scientific data 3.1 (2016): 1-9.

Individual level analysis of this data is described in the
:ref:`MNE fNIRS waveform tutorial <mne:tut-fnirs-processing>`
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

.. note::

   This tutorial uses data in the BIDS format.
   The BIDS specification for NIRS data is still under development. See:
   `fNIRS BIDS proposal <https://github.com/bids-standard/bids-specification/pull/802>`_.
   As such, to run this tutorial you must use the MNE-BIDS 0.10 or later.

   MNE-Python allows you to process fNIRS data that is not in BIDS format too.
   Simply modify the ``read_raw_`` function to match your data type.
   See :ref:`data importing tutorial <tut-importing-fnirs-data>` to learn how
   to use your data with MNE-Python.

.. note::

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

# Import MNE processing
from mne.preprocessing.nirs import optical_density, beer_lambert_law

# Import MNE-NIRS processing
from mne_nirs.statistics import run_glm
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.channels import get_short_channels, get_long_channels
from mne_nirs.channels import picks_pair_to_idx
from mne_nirs.visualisation import plot_glm_group_topo
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.visualisation import plot_glm_surface_projection
from mne_nirs.io.fold import fold_channel_specificity

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


# %%
# Set up directories
# ------------------
# .. sidebar:: Requires MNE-BIDS fNIRS branch
#
#    This section of code requires the MNE-BIDS fNIRS branch.
#    See instructions at the top of the page on how to install.
#    Alternatively, if your data is not in BIDS format,
#    skip to the next section.
#
# First we will define where the raw data is stored. We will analyse a
# BIDS dataset. This ensures we have all the metadata we require
# without manually specifying the trigger names etc.
# We first define where the root directory of our dataset is.
# In this example we use the example dataset ``fnirs_motor_group``.

root = fnirs_motor_group.data_path()
print(root)


# %%
# And as we are using MNE-BIDS we can create a BIDSPath object.
# This class helps to handle all the path wrangling.
# We inform the software that we are analysing nirs data that is saved in
# the snirf format.

dataset = BIDSPath(root=root, task="tapping",
                   datatype="nirs", suffix="nirs", extension=".snirf")

print(dataset.directory)

# %%
# For example we can automatically query the subjects, tasks, and sessions.

subjects = get_entity_vals(root, 'subject')
print(subjects)


# %%
# Define individual analysis
# --------------------------
#
# .. sidebar:: Individual analysis procedures
#
#    :ref:`Waveform individual analysis <tut-fnirs-processing>`
#
#    :ref:`GLM individual analysis <tut-fnirs-hrf>`
#
# First we define the analysis that will be applied to each file.
# This is a GLM analysis as described in the
# :ref:`individual GLM tutorial <tut-fnirs-hrf>`,
# so this example will skim over the individual level details.
#
# The analysis extracts a response estimate for each channel,
# each region of interest, and computes a contrast between left and right
# finger tapping.
# We return the raw object and data frames for the computed results.
# Information about channels, triggers and their meanings are stored in the
# BIDS structure and are automatically obtained when importing the data.
#
# Here we also resample to a 0.3 Hz sample rate just to speed up the example
# and use less memory, resampling to 0.6 Hz is a better choice for full
# analyses.
#
# .. note::
#
#    The nilearn library does not allow backslash characters in the condition
#    name. So we must replace the backslash with an underscore to ensure the
#    GLM computation is successful. Hopefully future versions of MNE-NIRS will
#    automatically handle these characters, see https://github.com/mne-tools/mne-nirs/issues/420
#    for more information. In the meantime use the following code to replace the
#    illegal characters.


def individual_analysis(bids_path, ID):

    raw_intensity = read_raw_bids(bids_path=bids_path, verbose=False)
    # Delete annotation labeled 15, as these just signify the start and end of experiment.
    raw_intensity.annotations.delete(raw_intensity.annotations.description == '15.0')
    # sanitize event names
    raw_intensity.annotations.description[:] = [
        d.replace('/', '_') for d in raw_intensity.annotations.description]

    # Convert signal to haemoglobin and resample
    raw_od = optical_density(raw_intensity)
    raw_haemo = beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo.resample(0.3)

    # Cut out just the short channels for creating a GLM repressor
    sht_chans = get_short_channels(raw_haemo)
    raw_haemo = get_long_channels(raw_haemo)

    # Create a design matrix
    design_matrix = make_first_level_design_matrix(raw_haemo, stim_dur=5.0)

    # Append short channels mean to design matrix
    design_matrix["ShortHbO"] = np.mean(sht_chans.copy().pick(picks="hbo").get_data(), axis=0)
    design_matrix["ShortHbR"] = np.mean(sht_chans.copy().pick(picks="hbr").get_data(), axis=0)

    # Run GLM
    glm_est = run_glm(raw_haemo, design_matrix)

    # Define channels in each region of interest
    # List the channel pairs manually
    left = [[4, 3], [1, 3], [3, 3], [1, 2], [2, 3], [1, 1]]
    right = [[8, 7], [5, 7], [7, 7], [5, 6], [6, 7], [5, 5]]
    # Then generate the correct indices for each pair
    groups = dict(
        Left_Hemisphere=picks_pair_to_idx(raw_haemo, left, on_missing='ignore'),
        Right_Hemisphere=picks_pair_to_idx(raw_haemo, right, on_missing='ignore'))

    # Extract channel metrics
    cha = glm_est.to_dataframe()

    # Compute region of interest results from channel data
    roi = glm_est.to_dataframe_region_of_interest(groups,
                                                  design_matrix.columns,
                                                  demographic_info=True)

    # Define left vs right tapping contrast
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i])
                        for i, column in enumerate(design_matrix.columns)])
    contrast_LvR = basic_conts['Tapping_Left'] - basic_conts['Tapping_Right']

    # Compute defined contrast
    contrast = glm_est.compute_contrast(contrast_LvR)
    con = contrast.to_dataframe()

    # Add the participant ID to the dataframes
    roi["ID"] = cha["ID"] = con["ID"] = ID

    # Convert to uM for nicer plotting below.
    cha["theta"] = [t * 1.e6 for t in cha["theta"]]
    roi["theta"] = [t * 1.e6 for t in roi["theta"]]
    con["effect"] = [t * 1.e6 for t in con["effect"]]

    return raw_haemo, roi, cha, con


# %%
# Run analysis on all participants
# --------------------------------
#
# Next we loop through the five measurements and run the individual analysis
# on each. We append the individual results in to a large dataframe that
# will contain the results from all measurements. We create a group dataframe
# for the region of interest, channel level, and contrast results.

df_roi = pd.DataFrame()  # To store region of interest results
df_cha = pd.DataFrame()  # To store channel level results
df_con = pd.DataFrame()  # To store channel level contrast results

for sub in subjects:  # Loop from first to fifth subject

    # Create path to file based on experiment info
    bids_path = dataset.update(subject=sub)

    # Analyse data and return both ROI and channel results
    raw_haemo, roi, channel, con = individual_analysis(bids_path, sub)

    # Append individual results to all participants
    df_roi = pd.concat([df_roi, roi], ignore_index=True)
    df_cha = pd.concat([df_cha, channel], ignore_index=True)
    df_con = pd.concat([df_con, con], ignore_index=True)


# %%
# Visualise Individual results
# ----------------------------
#
# First we visualise the results from each individual to ensure the
# data values look reasonable.
# Here we see that we have data from five participants, we plot just the HbO
# values and observe they are in the expect range.
# We can already see that the control condition is always near zero,
# and that the responses look to be contralateral to the tapping hand.

grp_results = df_roi.query("Condition in ['Control', 'Tapping_Left', 'Tapping_Right']")
grp_results = grp_results.query("Chroma in ['hbo']")

sns.catplot(x="Condition", y="theta", col="ID", hue="ROI", data=grp_results, col_wrap=5, ci=None, palette="muted", height=4, s=10)


# %%
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
#    :ref:`statsmodels docs <statsmodels:mixedlmmod>`
#
#    For a summary of these models in the context of fNIRS see section 3.5 of:
#    Santosa, H., Zhai, X., Fishburn, F., & Huppert, T. (2018).
#    The NIRS brain AnalyzIR toolbox. Algorithms, 11(5), 73.
#
# Next we use a linear mixed effects model to examine the
# relation between conditions and our response estimate (theta).
# Combinations of 3 fixed effects will be evaluated, ROI (left vs right),
# condition (control, tapping/left, tapping/right), and chromophore (HbO, HbR).
# With a random effect of subject.
# Alternatively, you could export the group dataframe (`df_roi.to_csv()`) and
# analyse in your favorite stats program.
#
# We do not explore the modeling procedure in depth here as topics
# such model selection and examining residuals are beyond the scope of
# this example (see relevant literature).
# Alternatively, we could use a robust linear
# model by using the code
# `roi_model = rlm('theta ~ -1 + ROI:Condition:Chroma', grp_results).fit()`.

grp_results = df_roi.query("Condition in ['Control','Tapping_Left', 'Tapping_Right']")

roi_model = smf.mixedlm("theta ~ -1 + ROI:Condition:Chroma",
                        grp_results, groups=grp_results["ID"]).fit(method='nm')
roi_model.summary()


# %%
# Second level analysis with covariates
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# .. sidebar:: Relevant literature
#
#    For a detailed discussion about covariates in fNIRS analysis see
#    the seminar by Dr. Jessica Gemignani
#    (`youtube <https://www.youtube.com/watch?feature=emb_logo&v=3E28sT1JI14>`_).
#
# It is simple to extend these models to include covariates.
# This dataset is small, so including additional factors may not be
# appropriate. However, for instructional purpose, we will include a
# covariate of gender. There are 3 females and 2 males in this dataset.
# Also, for instructional purpose, we modify the model
# above to only explore the difference between the two tapping conditions in
# the hbo signal in the right hemisphere.
#
# From the model result we observe that hbo responses in the right hemisphere
# are smaller when the right hand was used (as expected for these
# contralaterally dominant responses) and there is no significant
# effect of gender.

grp_results = df_roi.query("Condition in ['Tapping_Left', 'Tapping_Right']")
grp_results = grp_results.query("Chroma in ['hbo']")
grp_results = grp_results.query("ROI in ['Right_Hemisphere']")

roi_model = smf.mixedlm("theta ~ Condition + Sex",
                        grp_results, groups=grp_results["ID"]).fit(method='nm')
roi_model.summary()

# %%
# Visualise group results
# -----------------------
#
# Now we can summarise the output of the second level model.
# This figure shows that the control condition has small responses that
# are not significantly different to zero for both HbO
# and HbR in both hemispheres.
# Whereas clear significant responses are show for the two tapping conditions.
# We also observe the the tapping response is
# larger in the contralateral hemisphere.
# Filled symbols represent HbO, unfilled symbols represent HbR.

# Regenerate the results from the original group model above
grp_results = df_roi.query("Condition in ['Control','Tapping_Left', 'Tapping_Right']")
roi_model = smf.mixedlm("theta ~ -1 + ROI:Condition:Chroma",
                        grp_results, groups=grp_results["ID"]).fit(method='nm')

df = statsmodels_to_results(roi_model)

sns.catplot(x="Condition", y="Coef.", hue="ROI", data=df.query("Chroma == 'hbo'"), ci=None, palette="muted", height=4, s=10)


# %%
# Group topographic visualisation
# -------------------------------
#
# We can also view the topographic representation of the data
# (rather than the ROI summary above).
# Here we just plot the oxyhaemoglobin for the two tapping conditions.
# First we compute the mixed effects model for each channel (rather
# than region of interest as above).
# Then we pass these results to the topomap function.

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         gridspec_kw=dict(width_ratios=[1, 1]))

# Cut down the dataframe just to the conditions we are interested in
ch_summary = df_cha.query("Condition in ['Tapping_Left', 'Tapping_Right']")
ch_summary = ch_summary.query("Chroma in ['hbo']")

# Run group level model and convert to dataframe
ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition",
                       ch_summary, groups=ch_summary["ID"]).fit(method='nm')
ch_model_df = statsmodels_to_results(ch_model)

# Plot the two conditions
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    ch_model_df.query("Condition in ['Tapping_Left']"),
                    colorbar=False, axes=axes[0, 0],
                    vlim=(0, 20), cmap=mpl.cm.Oranges)

plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    ch_model_df.query("Condition in ['Tapping_Right']"),
                    colorbar=True, axes=axes[0, 1],
                    vlim=(0, 20), cmap=mpl.cm.Oranges)

# Cut down the dataframe just to the conditions we are interested in
ch_summary = df_cha.query("Condition in ['Tapping_Left', 'Tapping_Right']")
ch_summary = ch_summary.query("Chroma in ['hbr']")

# Run group level model and convert to dataframe
ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition",
                       ch_summary, groups=ch_summary["ID"]).fit(method='nm')
ch_model_df = statsmodels_to_results(ch_model)

# Plot the two conditions
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbr"),
                    ch_model_df.query("Condition in ['Tapping_Left']"),
                    colorbar=False, axes=axes[1, 0],
                    vlim=(-10, 0), cmap=mpl.cm.Blues_r)
plot_glm_group_topo(raw_haemo.copy().pick(picks="hbr"),
                    ch_model_df.query("Condition in ['Tapping_Right']"),
                    colorbar=True, axes=axes[1, 1],
                    vlim=(-10, 0), cmap=mpl.cm.Blues_r)


# %%
# Contrasts
# ---------
#
# Finally we can examine the difference between the left and right hand
# tapping conditions by viewing the contrast results
# in a topographic representation.

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
con_summary = df_con.query("Chroma in ['hbo']")

# Run group level model and convert to dataframe
con_model = smf.mixedlm("effect ~ -1 + ch_name:Chroma",
                        con_summary, groups=con_summary["ID"]).fit(method='nm')
con_model_df = statsmodels_to_results(con_model,
                                      order=raw_haemo.copy().pick(
                                          picks="hbo").ch_names)

plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo"),
                    con_model_df, colorbar=True, axes=axes)


# %%
#
# Or we can view only the left hemisphere for the contrast.
# And set all channels that dont have a significant response to zero.
#

plot_glm_group_topo(raw_haemo.copy().pick(picks="hbo").pick(picks=range(10)),
                    con_model_df, colorbar=True, threshold=True)


# %%
# Cortical Surface Projections
# ----------------------------
#
# The topographic plots above can sometimes be difficult to interpret with
# respect to the underlying cortical locations. It is also possible to present
# the data by projecting the channel level GLM values to the nearest cortical
# surface. This can make it easier to understand the spatial aspects of your
# data. Note however, that this is not a complete forward model with photon
# migration simulations.
# In the figure below we project the group results from the two conditions
# to the cortical surface, and also present the contrast results in the same
# fashion.
# As in the topo plots above you can see that the activity is predominately
# contralateral to the side of finger tapping.


# Generate brain figure from data
clim = dict(kind='value', pos_lims=(0, 8, 11))
brain = plot_glm_surface_projection(raw_haemo.copy().pick("hbo"),
                                    con_model_df, clim=clim, view='dorsal',
                                    colorbar=True, size=(800, 700))
brain.add_text(0.05, 0.95, "Left-Right", 'title', font_size=16, color='k')

# Run model code as above
clim = dict(kind='value', pos_lims=(0, 11.5, 17))
for idx, cond in enumerate(['Tapping_Left', 'Tapping_Right']):

    # Run same model as explained in the sections above
    ch_summary = df_cha.query("Condition in [@cond]")
    ch_summary = ch_summary.query("Chroma in ['hbo']")
    ch_model = smf.mixedlm("theta ~ -1 + ch_name", ch_summary,
                           groups=ch_summary["ID"]).fit(method='nm')
    model_df = statsmodels_to_results(ch_model, order=raw_haemo.copy().pick("hbo").ch_names)

    # Generate brain figure from data
    brain = plot_glm_surface_projection(raw_haemo.copy().pick("hbo"),
                                        model_df, clim=clim, view='dorsal',
                                        colorbar=True, size=(800, 700))
    brain.add_text(0.05, 0.95, cond, 'title', font_size=16, color='k')


# %%
# Table of channel level results
# ------------------------------
#
# Sometimes a reviewer wants a long table of results per channel.
# This can be generated from the statistics dataframe.

ch_summary = df_cha.query("Condition in ['Tapping_Left', 'Tapping_Right']")
ch_summary = ch_summary.query("Chroma in ['hbo']")

# Run group level model and convert to dataframe
ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition",
                       ch_summary, groups=ch_summary["ID"]).fit(method='nm')

# Here we can use the order argument to ensure the channel name order
ch_model_df = statsmodels_to_results(ch_model,
                                     order=raw_haemo.copy().pick(
                                         picks="hbo").ch_names)
# And make the table prettier
ch_model_df.reset_index(drop=True, inplace=True)
ch_model_df = ch_model_df.set_index(['ch_name', 'Condition'])
ch_model_df


# %%
# .. _tut-fnirs-group-relating:
#
# Relating Responses to Brain Landmarks
# -------------------------------------
#
# .. sidebar:: fOLD Toolbox
#
#    You should use the fOLD toolbox to pick your optode locations
#    when designing your experiment.
#    The tool is very intuitive and easy to use.
#    Be sure to cite the authors if you use their tool or data:
#
#    Morais, Guilherme Augusto Zimeo, Joana Bisol Balardin, and João Ricardo Sato. "fNIRS optodes’ location decider (fOLD): a toolbox for probe arrangement guided by brain regions-of-interest." Scientific reports 8.1 (2018): 1-11.
#
# It can be useful to understand what brain structures
# the measured response may have resulted from. Here we illustrate
# how to report the brain structures/landmarks that the source
# detector pair with the largest response was sensitive to.
#
# First we determine the channel with the largest response.
#
# Next, we query the fOLD dataset to determine the
# brain landmarks that this channel is most sensitive to.
# MNE-NIRS does not distribute the fOLD toolbox or the data
# that they provide. See the Notes section of
# :func:`mne_nirs.io.fold_channel_specificity` for more information.

largest_response_channel = ch_model_df.loc[ch_model_df['Coef.'].idxmax()]
largest_response_channel


# %%
#
# Next we use information from the fOLD toolbox to report the
# channel specificity to different brain regions.
# For licensing reasons, these files are not distributed with MNE-NIRS.
# To set up your system to use the fOLD functions, see the Notes section of
# :func:`mne_nirs.io.fold_channel_specificity`.

raw_channel = raw_haemo.copy().pick(largest_response_channel.name[0])
fold_channel_specificity(raw_channel)[0]


# %%
#
# We observe that the channel with the largest response to tapping
# had the greatest specificity to the Precentral Gyrus, which is
# the site of the primary motor cortex. This is consistent
# with the expectation for a finger tapping task.


# %%
# Conclusion
# ----------
#
# This example has demonstrated how to perform a group level analysis
# using a GLM approach.
# We observed the responses were evoked primarily contralateral to the
# hand of tapping and most likely originate from the primary motor cortex.
