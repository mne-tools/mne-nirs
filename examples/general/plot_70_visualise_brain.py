"""
.. _tut-fnirs-vis-brain:

Utilising Anatomical Information
================================

This example demonstrates how you can utilise anatomical and sensor position
information in your analysis pipeline. This information can be used to
verify measurement/analysis and also improve analysis accuracy
:footcite:`novi2020integration`.

This example demonstrates how to plot your data on a 3D brain
and overlay the sensor locations and regions of interest.

This tutorial glosses over the processing details, see the
:ref:`GLM tutorial <tut-fnirs-hrf>` for details on the preprocessing.

.. note::

   This tutorial does not render correctly when built on the cloud infrastructure.
   As such, the images on this web page are currently just black squares.
   However, this should work perfectly well when run locally.
   Hopefully this issue with the cloud server will be fixed soon.

.. contents:: Page contents
   :local:
   :depth: 2


"""
# sphinx_gallery_thumbnail_number = 9

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import numpy as np

import mne

from mne_bids import BIDSPath, read_raw_bids
import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.io.fold import fold_landmark_specificity
from mne_nirs.visualisation import plot_nirs_source_detector


# %%
# Download example data
# -------------------------------
#
# Download required data for this tutorial.

# %%
# Download example fNIRS data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download the ``audio_or_visual_speech`` dataset and load the first measurement.

root = mne_nirs.datasets.audio_or_visual_speech.data_path()
dataset = BIDSPath(root=root, suffix="nirs", extension=".snirf",
                   task="AudioVisualBroadVsRestricted", datatype="nirs",
                   subject="01", session="01")
raw = read_raw_bids(dataset)


# %%
# Download annotation information
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download the HCP-MMP parcellation.

# Download anatomical locations
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)


# %%
# Verify placement of sensors
# ---------------------------
#
# The first thing we can do is plot the location of the optodes and channels
# over an average brain surface to verify the data, specifically the 3D coordinates,
# have been loaded correctly. The sources are represented as red dots,
# the detectors are represented as black dots, the whit lines represent source-detector
# pairs, and the orange dots represent channel locations.
# In this example we can see channels over the left inferior frontal gyrus,
# auditory cortex, planum temporale, and occipital lobe.

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])
brain.show_view(azimuth=180, elevation=80, distance=450)


# %%
# Plot sensor channels and anatomical region of interest
# ------------------------------------------------------
#
# Once the data has been loaded we can highlight anatomical regions of interest
# to ensure that the sensors are appropriately placed to measure from
# the relevant brain structures.
# In this example we highlight the primary auditory cortex in blue,
# and we can see that a number of channels are placed over this structure.

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])

aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
brain.add_label(aud_label, borders=False, color='blue')
brain.show_view(azimuth=180, elevation=80, distance=450)


# %%
# Plot channels sensitive to anatomical region of interest
# --------------------------------------------------------
#
# Rather than simply eye balling the sensor and ROIs of interest, we can
# quantify the specificity of each channel to the anatomical region of interest
# and select channels that are sufficiently sensitive for further analysis.
# In this example we highlight the left inferior frontal gyrus (IFG) and
# use data from the fOLD toolbox :footcite:`morais2018fnirs`.
# To see more details about how to use the fOLD data see
# `this tutorial <https://mne.tools/mne-nirs/main/auto_examples/general/plot_12_group_glm.html#id13>`_.

# Specify the location of the fOLD files
fold_files = [os.path.join(os.path.expanduser("~"), "mne_data", "fOLD", "fOLD-public-master", "Supplementary", "10-10.xls"),
              os.path.join(os.path.expanduser("~"), "mne_data", "fOLD", "fOLD-public-master", "Supplementary", "10-5.xls")]

# Return specificity of each channel to the Left IFG
specificity = fold_landmark_specificity(raw, 'L IFG (p. Triangularis)', fold_files)

# Retain only channels with specificity to left IFG of greater than 50%
raw_IFG = raw.copy().pick(picks=np.where(specificity > 50)[0])

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw_IFG.info, trans='fsaverage', fnirs=['channels', 'pairs'])

ifg_label = [label for label in labels_combined if label.name == 'Inferior Frontal Cortex-lh'][0]
brain.add_label(ifg_label, borders=False, color='green')

brain.show_view(azimuth=140, elevation=95, distance=360)


# %%
#
# Alternatively, we can retain all channels and visualise the specificity of each channel the ROI
# by encoding the specificty in the color of the line between each source and detector.
# In this example we see that several channels have substantial specificity to
# the region of interest.
#
# Note: this function currently doesnt support the new MNE brain API, so does
# not allow the same behaviour as above (adding sensors, highlighting ROIs etc).
# It should be updated in the near future.

fig = plot_nirs_source_detector(specificity, raw.info, surfaces='brain',
                                subject='fsaverage', subjects_dir=subjects_dir, trans='fsaverage')
mne.viz.set_3d_view(fig, azimuth=140, elevation=95)


# %%
# Preprocess fNIRS data
# ---------------------
#
# We can also use the 3D information to project the results on to the cortical surface.
# First, we process the fNIRS data. This is a duplication of the GLM tutorial
# analysis. The details will not be described here, instead view the
# :ref:`fNIRS GLM tutorial <tut-fnirs-hrf>`
#
# After reading the data we resample down to 1Hz
# to meet github memory constraints.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_raw_dir).load_data()
raw_intensity.resample(0.7)
raw_intensity.annotations.rename({'1.0': 'Control',
                                  '2.0': 'Tapping/Left',
                                  '3.0': 'Tapping/Right'})
raw_intensity.annotations.delete(raw_intensity.annotations.description == '15.0')
raw_intensity.annotations.set_durations(5)
raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)
short_chs = get_short_channels(raw_haemo)
raw_haemo = get_long_channels(raw_haemo)
design_matrix = make_first_level_design_matrix(raw_haemo, drift_model='cosine',
                                               high_pass=0.005, hrf_model='spm', stim_dur=5.0)
design_matrix["ShortHbO"] = np.mean(short_chs.copy().pick(picks="hbo").get_data(), axis=0)
design_matrix["ShortHbR"] = np.mean(short_chs.copy().pick(picks="hbr").get_data(), axis=0)
glm_est = run_glm(raw_haemo, design_matrix)


# %%
# Plot surface projection of GLM results
# --------------------------------------
#
# Finally, we can project the GLM results from each channel to the nearest cortical surface
# and overlay the sensor positions and two different regions of interest.
# In this example we highlight the motor cortex and auditory association cortex.

# Plot the projection and sensor locations
brain = glm_est.copy().surface_projection(condition="Tapping/Right", view="dorsal", chroma="hbo")
brain.add_sensors(glm_est.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])

# mark the premotor cortex in green
aud_label = [label for label in labels_combined if label.name == 'Premotor Cortex-lh'][0]
brain.add_label(aud_label, borders=True, color='green')

# mark the auditory association cortex in blue
aud_label = [label for label in labels_combined if label.name == 'Auditory Association Cortex-lh'][0]
brain.add_label(aud_label, borders=True, color='blue')

brain.show_view(azimuth=160, elevation=60, distance=400)


# %%
# Bibliography
# -----------------------------------------------
#
# .. footbibliography::
