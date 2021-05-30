"""
.. _tut-fnirs-hrf:

GLM FIR Analysis
================

In this example we analyse data from a real multichannel
functional near-infrared spectroscopy (fNIRS)
experiment (see :ref:`tut-fnirs-hrf-sim` for a simplified simulated
analysis). The experiment consists of three conditions
1) tapping with the left hand,
2) tapping with the right hand,
3) a control condition where the participant does nothing.

In this tutorial the morphology of an fNIRS response is obtained using an
Finite Impulse Response (FIR) GLM analysis.
An alternative epoching style analysis on the same data can be
viewed in the
:ref:`waveform analysis example <tut-fnirs-processing>`.
See
`Luke et al (2021) <https://www.spiedigitallibrary.org/journals/neurophotonics/volume-8/issue-2/025008/Analysis-methods-for-measuring-passive-auditory-fNIRS-responses-generated-by/10.1117/1.NPh.8.2.025008.short>`_
for a comparison of the epoching and GLM approaches.

This tutorial only examines the control and tapping with right hand conditions
to simplify explanation and minimise computation time. Extending the code to
also analyse the left tapping condition is left as an exercise for the reader.

This GLM analysis is a wrapper over the excellent
`Nilearn GLM <http://nilearn.github.io/modules/reference.html#module-nilearn.glm>`_.

.. note::

   This is an advanced tutorial and requires knowledge of pandas and numpy.

   The sample rate used in this example is set to 0.5 Hz. This is to ensure
   the code can run on the continuous integration servers. You may wish to
   increase the sample rate by adjusting `resample` below for your
   own analysis.

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
from pprint import pprint

import mne
import mne_nirs

# Import MNE-NIRS processing
from mne_nirs.statistics import run_GLM
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import glm_region_of_interest, statsmodels_to_results
from mne_nirs.datasets import fnirs_motor_group

# Import MNE-BIDS processing
from mne_bids import BIDSPath, read_raw_bids

# Import StatsModels
import statsmodels.formula.api as smf

# Import Plotting Library
import matplotlib.pyplot as plt


###############################################################################
# Define FIR analysis
# ---------------------------------------------------------------------
#

def analysis(fname, ID):

    raw_intensity = read_raw_bids(bids_path=fname, verbose=False)

    # Convert signal to haemoglobin and just keep hbo
    raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od)
    raw_haemo.resample(0.5, npad="auto")

    # Cut out just the short channels for creating a GLM regressor
    short_chans = mne_nirs.channels.get_short_channels(raw_haemo)
    raw_haemo = mne_nirs.channels.get_long_channels(raw_haemo)

    # Create a design matrix
    design_matrix = make_first_level_design_matrix(raw_haemo,
                                                   hrf_model='fir',
                                                   stim_dur=1.0,
                                                   fir_delays=range(10),
                                                   drift_model='cosine',
                                                   high_pass=0.01,
                                                   oversampling=1)
    # Add short channels as regressor in GLM
    for chan in range(len(short_chans.ch_names)):
        design_matrix[f"short_{chan}"] = short_chans.get_data(chan).T

    # Run GLM
    glm_est = run_GLM(raw_haemo, design_matrix)

    # Create a single ROI that includes all channels for example
    rois = dict(AllChannels=range(len(raw_haemo.ch_names)))
    # Compute output metrics by ROI
    df_ind = pd.DataFrame()
    for idx, col in enumerate(design_matrix.columns):
        df_ind = df_ind.append(glm_region_of_interest(glm_est, rois, idx, col))

    df_ind["ID"] = ID
    df_ind["theta"] = [t * 1.e6 for t in df_ind["theta"]]

    return df_ind, raw_haemo, design_matrix


###############################################################################
# Run analysis
# ---------------------------------------------------------------------
#

df = pd.DataFrame()

for sub in range(1, 6):  # Loop from first to fifth subject
    ID = '%02d' % sub  # Tidy the subject name

    # Create path to file based on experiment info
    bids_path = BIDSPath(subject=ID, task="tapping",
                         root=fnirs_motor_group.data_path(),
                         datatype="nirs", suffix="nirs", extension=".snirf")

    df_individual, raw, dm = analysis(bids_path, ID)

    df = df.append(df_individual)


###############################################################################
# Tidy the dataframe
# ---------------------------------------------------------------------
#

# Create extra columns to easily index different conditions
df["isControl"] = ["Control" in n for n in df["Condition"]]
df["isTapping"] = ["Tapping/Right" in n for n in df["Condition"]]
df["isDelay"] = ["delay" in n for n in df["Condition"]]
df = df.query("isDelay in [True]")

# To simplify this example we will only look at the right tapping condition
# so we now remove the left tapping conditions from the design matrix and
# GLM results
dm_cols_not_left = np.where(["Left" not in c for c in dm.columns])[0]
dm = dm[[dm.columns[i] for i in dm_cols_not_left]]

# Create a new column that stores the condition in a tidy fashion
df.loc[df["isControl"] == True, "TidyCond"] = "Control"
df.loc[df["isTapping"] == True, "TidyCond"] = "Tapping"
df = df.query("TidyCond in ['Control', 'Tapping']")


# Finally, extract the FIR delay in to its own column in data frame
df.loc[:, "delay"] = [n.split('_')[2] for n in df.Condition]

df


###############################################################################
# Run group level model
# ---------------------------------------------------------------------
#


rlm_model = smf.mixedlm('theta ~ -1 + delay:TidyCond:Chroma', df,
                        groups=df["ID"]).fit()

# Create a dataframe from LME model for plotting below
df_sum = statsmodels_to_results(rlm_model)
df_sum["delay"] = [int(n) for n in df_sum["delay"]]
df_sum = df_sum.sort_values('delay')

rlm_model.summary()


###############################################################################
# Summarise group level findings
# ---------------------------------------------------------------------
#

# Print the result for the oxyhaemoglobin data in the tapping condition
df_sum.query("TidyCond in ['Tapping']").query("Chroma in ['hbo']")


###############################################################################
# Plot response from single condition
# ---------------------------------------------------------------------
#

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))

cond = "Tapping"

# Extract the design matrix columns that correspond to the condition
dm_cond_idxs = np.where([cond in n for n in dm.columns])[0]
dm_cond = dm[[dm.columns[i] for i in dm_cond_idxs]]

# Extract the
df_hbo = df_sum.query("TidyCond in [@cond]").query("Chroma in ['hbo']")
vals = [float(v) for v in df_hbo["Coef."]]
dm_cond_scaled = dm_cond * vals

# correct for  timing of first event
index_values = dm_cond_scaled.index - np.ceil(raw.annotations.onset[1])

axes[0].plot(index_values, dm_cond_scaled)
axes[1].plot(index_values, np.sum(dm_cond_scaled, axis=1))

# Extract the
df_hbo = df_sum.query("TidyCond in [@cond]").query("Chroma in ['hbr']")
vals = [float(v) for v in df_hbo["Coef."]]
dm_cond_scaled = dm_cond * vals

axes[1].plot(index_values, np.sum(dm_cond_scaled, axis=1))

axes[0].set_xlim(-5, 35)
axes[1].set_xlim(-5, 35)
axes[0].set_title("FIR Components (Tapping/Right)")
axes[1].set_title("Evoked Response (Tapping/Right)")
axes[0].set_ylabel("Oyxhaemoglobin (ΔμMol)")
axes[1].set_ylabel("Haemoglobin (ΔμMol)")
axes[1].legend(["Oyxhaemoglobin", "Deoyxhaemoglobin"])
axes[0].set_xlabel("Time (s)")
axes[1].set_xlabel("Time (s)")

plt.show()