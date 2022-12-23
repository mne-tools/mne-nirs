"""
.. _ex-fnirs-datasets:

================
Example Datasets
================

To get you up and running with MNE-NIRS and fNIRS data analysis we provide
some example datasets.
Each dataset is published and if you use the data at any stage of
your experiment (including piloting and exploration) then please cite the
authors of the data.

A brief explanation of each dataset is provided below, and a demonstration
of how to download the data and address it with MNE.
Each dataset is provided in BIDs format, as such we can use the MNE-BIDS
package to quickly report the number of trials for each condition in the
dataset.

"""
# sphinx_gallery_thumbnail_number = 1

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)


import mne_nirs
import mne_bids.stats


# %%
# *******************
# Finger Tapping Data
# *******************
#
# This data is from the publication
# :footcite:p:`Luke_fNIRS_Finger_Tapping_2021`.
# This data provides an example of the canonical finger tapping experiment.
# Data is provided from five participants. The experiment was a block design
# with three conditions. The first condition consisted of the participants
# tapping their thumb to fingers on the left hand, the second condition was the
# same but with the right hand. The final condition was a control, with no
# motor task.

datapath = mne_nirs.datasets.fnirs_motor_group.data_path()
mne_bids.stats.count_events(datapath)


# %%
# *************************
# Auditory Speech and Noise
# *************************
#
# This data is from the publication
# :footcite:p:`luke2021analysis`.
# In this experiment listeners were presented with auditory stimuli via insertphones.
# Data is provided from seventeen participants. The experiment was a block
# design with three conditions. The first condition consisted of speech,
# the second condition consisted of low frequency noise.
# The final condition was a control, no audio was presented to the listeners.

datapath = mne_nirs.datasets.block_speech_noise.data_path()
mne_bids.stats.count_events(datapath)


# %%
# **********************
# Audio or Visual Speech
# **********************
#
# This data is from the publication
# :footcite:p:`shader2021use`.
# In this experiment participants were presented a continuously running story
# in blocked segments. Each segment was presented as either audio only
# or visual only.
# In addition to the audio visual stimuli, a control condition was also
# presented randomly throughout the experiment.

datapath = mne_nirs.datasets.audio_or_visual_speech.data_path()
mne_bids.stats.count_events(datapath)


# %%
# Bibliography
# -----------------------------------------------
#
# .. footbibliography::
