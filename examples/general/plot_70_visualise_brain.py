"""
.. _tut-fnirs-vis-brain:

Brain Plotting
==============

This example demonstrates how to plot your data on a 3D brain
and overlay the sensor locations and regions of interest.

This tutorial glosses over the processing details, see the other examples
for more details on preprocessing.

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
import matplotlib.pyplot as plt

import mne

from mne_bids import BIDSPath, read_raw_bids, get_entity_vals

import mne_nirs

from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_long_channels, get_short_channels
from mne_nirs.io.fold import fold_landmark_specificity


from PIL import Image
import matplotlib.pyplot as plt


# %%
# Download annotation information
# -------------------------------
#
# Download the HCP-MMP parcellation.

# Download anatomical locations
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, accept=True)
labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)
labels_combined = mne.read_labels_from_annot('fsaverage', 'HCPMMP1_combined', 'lh', subjects_dir=subjects_dir)


# %%
# Create function to grab a screenshot of the 3D image
# ----------------------------------------------------
#
# This function creates a figure from the 3D visualisation.
# It is only required as the tutorial is being run on a cloud server.
# On your local machine the tutorial should open a
# nice 3D viewer automatically.
# You should not need to use this function locally.

def cloud_3d_screenshot(brain, azimuth=160, elevation=60, distance=400):
    brain.show_view(azimuth=azimuth, elevation=elevation, distance=distance)
    fig, axes = plt.subplots(figsize=(16, 16))
    b = np.asarray(Image.fromarray(brain.screenshot()))
    axes.imshow(b)
    axes.axis('off')
    return fig


# %%
# Plot brain with all sensors and channels
# ----------------------------------------
#
# Overlay the sensor information for a specific measurement
# on the image of a brain.

# Load data
root = mne_nirs.datasets.audio_or_visual_speech.data_path()
dataset = BIDSPath(root=root, suffix="nirs", extension=".snirf",
                   task="AudioVisualBroadVsRestricted", datatype="nirs",
                   subject="01", session="01")
raw = read_raw_bids(dataset)

# Plot brain and overlay sensors
brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])
cloud_3d_screenshot(brain, azimuth=180, elevation=80, distance=450)


# %%
# Plot sensor channels and anatomical region of interest
# ------------------------------------------------------
#
# Overlay the sensor information for a specific measurement
# on the image of a brain and mark a specific region of interest.

# Plot brain and overlay sensors
brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])

# mark the primary auditory cortex in blue
aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
brain.add_label(aud_label, borders=False, color='blue')

cloud_3d_screenshot(brain, azimuth=180, elevation=80, distance=450)\


# %%
# Plot channels sensitive to anatomical region of interest
# --------------------------------------------------------
#
# Overlay the sensor information for a specific measurement
# on the image of a brain and mark a specific region of interest.
#
# This example highlights the left inferior frontal gyrus in green
# and displays the channels with greater than 50% specificity to this structure.

fold_files = [os.path.join(os.path.expanduser("~"), "mne_data", "fOLD", "fOLD-public-master", "Supplementary", "10-10.xls"),
              os.path.join(os.path.expanduser("~"), "mne_data", "fOLD", "fOLD-public-master", "Supplementary", "10-5.xls")]
specificity = fold_landmark_specificity(raw, 'L IFG (p. Triangularis)', fold_files)

raw_IFG = raw.copy().pick(picks=np.where(specificity > 50)[0])

# Plot brain and overlay sensors
brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, background='w', cortex='0.5')
brain.add_sensors(raw_IFG.info, trans='fsaverage', fnirs=['channels', 'pairs'])

# mark the primary auditory cortex in blue
ifg_label = [label for label in labels_combined if label.name == 'Inferior Frontal Cortex-lh'][0]
brain.add_label(ifg_label, borders=False, color='green')

cloud_3d_screenshot(brain, azimuth=140, elevation=95, distance=360)


# %%
# Preprocess fNIRS data
# ---------------------
#
# First we process the fNIRS data. This is a duplication of the GLM tutorial
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
# First we process the fNIRS data. This is a duplication of the GLM tutorial
# analysis. The details will not be described here, instead view the
# :ref:`fNIRS GLM tutorial <tut-fnirs-hrf>`
#
# After reading the data we resample down to 1Hz
# to meet github memory constraints.

# Plot the projection and sensor locations
brain = glm_est.copy().surface_projection(condition="Tapping/Right", view="dorsal", chroma="hbo")
brain.add_sensors(glm_est.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])

# mark the premotor cortex in green
aud_label = [label for label in labels_combined if label.name == 'Premotor Cortex-lh'][0]
brain.add_label(aud_label, borders=True, color='green')

# mark the auditory association cortex in blue
aud_label = [label for label in labels_combined if label.name == 'Auditory Association Cortex-lh'][0]
brain.add_label(aud_label, borders=True, color='blue')

cloud_3d_screenshot(brain)
