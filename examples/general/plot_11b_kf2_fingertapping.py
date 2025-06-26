"""
.. _tut-fnirs-kf2-ft:

Finger Tapping Analysis with Kernel Flow2 Data
================================================

This example follows closely the :ref:`tut-fnirs-hrf`
`mne-nirs` finger tapping example with some minor variations,
using fNIRS data from the Kernel Flow 2 (KF2) system.

KF2 is a time-domain (TD)-fNIRS system, and so some aspects of the
``snirf`` file i/o are different to other mne-nirs examples which all
use continuous-wave data. This example thus serves as a (minimal) demo
and test of ``mne-nirs`` for TD data, and also as an example of fNIRS
brain activity measurements from a high-density (~5000 channel)
whole-head montage.

The dataset was collected at Kernel on September 15, 2023, using the
Kernel finger tapping task. As with the main ``mne-nirs`` finger tapping
example, the following demonstrates an ‘Evoked’ (trial-avearaging) and
GLM-based analysis of an experiment consisting of three conditions:

1. Left-handed finger tapping
2. Right-handed finger tapping
3. A control condition where the participant does nothing.

There are also some small tweaks to the MNE plotting tools to accomodate
high channel density.

"""
# sphinx_gallery_thumbnail_number = 10

# Authors: Julien DuBois     <https://github.com/julien-dubois-k>
#          John D Griffiths  <john.griffiths@utoronto.ca>
#          Eric Larson       <https://larsoner.com>
#
# License: BSD (3-clause)

# Importage
import os

import gdown
import h5py
import matplotlib
import matplotlib.pyplot as plt
import mne
import mne.channels
import mne.io.snirf
import nilearn.plotting
import numpy as np
import pandas as pd

import mne_nirs.experimental_design
import mne_nirs.statistics

# %%
# Import raw NIRS data
# --------------------
#
# First we import the motor tapping data, These data are similar to those
# described and used in the  ``MNE fNIRS tutorial <mne:tut-fnirs-processing>``
#
# After reading the data we resample down to 1Hz to meet github memory constraints.

# first download the data
snirf_id = "1VYtux3p3hcM41FPjmnAwDgc4MM8DBxkg"
snirf_file = "Hb_Moments.snirf"
if not os.path.isfile(snirf_file):
    gdown.download(id=snirf_id, output=snirf_file)

# now load into an MNE object
raw = mne.io.snirf.read_raw_snirf(snirf_file)
raw.load_data()
# raw._data *= 1e-6


# %%
# Get more info from the snirf file
# ------------------------------------
#
# Unfortunately, a lot of useful information that is in the SNIRF file is not
# yet read by the MNE SNIRF reader. For example, the actual source and detector
# names (which reflect the modules they belong to).
#
# Fortunately, it's quite easy to find what you need in the SNIRF hdf archive from the
# `SNIRF specification <https://github.com/fNIRS/snirf/blob/master/snirf_specification.md>`_

probe_keys = [
    ("detectorLabels", str),
    ("sourceLabels", str),
    ("sourcePos3D", float),
    ("detectorPos3D", float),
]
with h5py.File(snirf_file, "r") as file:
    probe_data = {
        key: np.array(file["nirs"]["probe"][key]).astype(dtype)
        for key, dtype in probe_keys
    }
print([*probe_data])


# %%
#
# We also need data about the events.

raw.annotations.to_data_frame()


# %%
#
# Unfortunately MNE didn't load the block types so we don't know whether
# a block is LEFT or RIGHT tapping. Fear not! the SNIRF file has it all,
# albeit in a convoluted format. Let's reconstruct the information here:

with h5py.File(snirf_file, "r") as file:
    ctr = 1
    while (stim := f"stim{ctr}") in file["nirs"]:
        print(stim, np.array(file["nirs"][stim]["name"]))
        ctr += 1


# %%
#
# Looks like "stim1" has the StartBlock event information, let's dig in:

with h5py.File(snirf_file, "r") as file:
    df_start_block = pd.DataFrame(
        data=np.array(file["nirs"]["stim1"]["data"]),
        columns=[col.decode("UTF-8") for col in file["nirs"]["stim1"]["dataLabels"]],
    )
df_start_block


# %%
#
# Ok, `BlockType.Left` and `BlockType.Right` look useful.
# Alright, now we can make events from the MNE annotations and sort
# them into two types, left and right tapping blocks.

# %%
# Define the events
# ------------------------------------
#
events, _ = mne.events_from_annotations(raw, {"StartBlock": 1})
event_id = {"Tapping/Left": 1, "Tapping/Right": 2}
events[df_start_block["BlockType.Left"] == 1.0, 2] = event_id["Tapping/Left"]
events[df_start_block["BlockType.Right"] == 1.0, 2] = event_id["Tapping/Right"]
events

# %%
#
# Plot the events
mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info["sfreq"])


# %%
#
# Convert useful events back to annotations...
event_desc = {v: k for k, v in event_id.items()}
annotations_from_events = mne.annotations_from_events(
    events, raw.info["sfreq"], event_desc=event_desc
)


# %%
#
# Set these annotations on the raw data
raw.set_annotations(annotations_from_events)


# %%
# Epoch the data
# ------------------------------------
#
# The SNIRF Hb Moments file contains data that has been
# preprocessed quite extensively and is almost "ready for consumption".
# Details of the preprocessing are outlined on the
# `Kernel Docs <https://docs.kernel.com/docs/data-export-pipelines>`_ .
# All that remains to be done is some filtering to focus on the
# "neural" band. We typically use a moving-average filter for
# detrending and a FIR filter for low-pass filtering. With MNE, we
# can use the available bandpass FIR filter to achieve similar effects.

raw_filt = raw.copy().filter(0.01, 0.1, h_trans_bandwidth=0.01, l_trans_bandwidth=0.01)

# %%
#
# A little bit of MNE syntax to epoch the data with respect to the events we just
# extracted:

tmin, tmax = -5, 45
epochs = mne.Epochs(
    raw_filt,
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    proj=True,
    baseline=(None, 0),
    preload=True,
    detrend=None,
    verbose=True,
)


# %%
# Plot the evoked respones
# ------------------------------------
#

# %%
#
# Let's look again at the info loaded by MNE for each channel (source-detector pair)
epochs.info["chs"][0]


# %%
#
# Extract the indices of the sources and detectors from the "channel names"
# and also the source and detector positions so we can access the source
# detector distance for each channel.
idx_sources = np.array(
    [int(ch.split("_")[0][1:]) - 1 for ch in epochs.info["ch_names"]]
)
idx_detectors = np.array(
    [int(ch.split("_")[1].split(" ")[0][1:]) - 1 for ch in epochs.info["ch_names"]]
)
source_positions = np.array(probe_data["sourcePos3D"])[idx_sources]
detector_positions = np.array(probe_data["detectorPos3D"])[idx_detectors]
sds = np.sqrt(np.sum((source_positions - detector_positions) ** 2, axis=1))

# %%
#
# Make `evoked` objects for the evoked response to LEFT and RIGHT tapping,
# and for the contrast left < right, for channels with a source-detector
# distance between 15-30mm
idx_channels = np.flatnonzero((sds > 15) & (sds < 30))
left_evoked = epochs["Tapping/Left"].average(picks=idx_channels)
right_evoked = epochs["Tapping/Right"].average(picks=idx_channels)
left_right_evoked = left_evoked.copy()
left_right_evoked._data = left_evoked._data - right_evoked._data

# %%
#
# Now plot the evoked data
chromophore = "hbo"
times = [0, 10, 20, 30, 40]
vlim = (-2, 2)

plot_topo_kwargs = dict(
    ch_type=chromophore,
    sensors=False,
    image_interp="linear",
    vlim=vlim,
    extrapolate="local",
    contours=0,
    colorbar=False,
    show=False,
)

fig, ax = plt.subplots(
    figsize=(12, 8), nrows=3, ncols=len(times), sharex=True, sharey=True
)

for idx_time, time in enumerate(times):
    _ = left_evoked.plot_topomap([time], axes=ax[0][idx_time], **plot_topo_kwargs)
    _ = right_evoked.plot_topomap([time], axes=ax[1][idx_time], **plot_topo_kwargs)
    _ = left_right_evoked.plot_topomap([time], axes=ax[2][idx_time], **plot_topo_kwargs)
    if idx_time == 0:
        ax[0][0].set_ylabel("LEFT")
        ax[1][0].set_ylabel("RIGHT")
        ax[2][0].set_ylabel("LEFT - RIGHT")
fig.suptitle(chromophore)


# %%
#
# Despite the absence of thresholding, we can discern:
#
# - LEFT tapping (first row): a nice hotspot in the right motor cortex at 10s
# - RIGHT tapping (second row): a nice hotspot in the left motor cortex at 10s
# - LEFT-RIGHT tapping (last row): hotspot in the right motor cortex, and
#   negative counterpart in the left motor cortex, at 10s
#

# %%
#
# Now let's look at the time courses:
idx_sources = np.array(
    [int(ch.split("_")[0][1:]) - 1 for ch in left_evoked.info["ch_names"]]
)
is_selected_hbo = np.array([ch.endswith("hbo") for ch in left_evoked.info["ch_names"]])


# %%
#
# MODULE 21 is in the left motor cortex, MODULE 20 in the right motor cortex
is_left_motor = is_selected_hbo & (
    idx_sources == np.flatnonzero(np.array(probe_data["sourceLabels"]) == "M021S01")[0]
)
is_right_motor = is_selected_hbo & (
    idx_sources == np.flatnonzero(np.array(probe_data["sourceLabels"]) == "M020S01")[0]
)

# %%
#
# average all channels coming from source 20 or 21 formed with detectors
# between 15-30mm from the source
right_evoked_combined = mne.channels.combine_channels(
    right_evoked,
    {
        "left_motor": np.flatnonzero(is_left_motor),
        "right_motor": np.flatnonzero(is_right_motor),
    },
)
left_evoked_combined = mne.channels.combine_channels(
    left_evoked,
    {
        "left_motor": np.flatnonzero(is_left_motor),
        "right_motor": np.flatnonzero(is_right_motor),
    },
)

# %%
#
# now plot these
fig, axes = plt.subplots(figsize=(10, 5), ncols=2, sharey=True)
mne.viz.plot_compare_evokeds(
    dict(
        left=left_evoked_combined.copy().pick_channels(["left_motor"]),
        right=right_evoked_combined.copy().pick_channels(["left_motor"]),
    ),
    legend="upper left",
    axes=axes[0],
    show=False,
)
axes[0].set_title("Left motor cortex\n\n")
mne.viz.plot_compare_evokeds(
    dict(
        left=left_evoked_combined.copy().pick_channels(["right_motor"]),
        right=right_evoked_combined.copy().pick_channels(["right_motor"]),
    ),
    legend=False,
    axes=axes[1],
    show=False,
)
axes[1].set_title("Right motor cortex\n\n")
plt.tight_layout()


# %%
# GLM Analysis
# ------------------------------------
#
# GLM analysis in MNE-NIRS is powered under the hood by `Nilearn` functionality.
#
# Here we mostly followed the :ref:`tut-fnirs-hrf` tutorial
#

# %%
#
# First show the boxcar design looks
s = mne_nirs.experimental_design.create_boxcar(
    raw, stim_dur=(stim_dur := df_start_block["Duration"].mean())
)
fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)
ax.plot(raw.times, s)
ax.legend(["Left", "Right"], loc="upper right")
ax.set_xlabel("Time (s)")


# %%
#
# Now make a design matrix, including drift regressors
design_matrix = mne_nirs.experimental_design.make_first_level_design_matrix(
    raw,
    drift_model="cosine",
    high_pass=0.01,  # Must be specified per experiment
    hrf_model="glover",
    stim_dur=stim_dur,
)

# %%
#
# Next, plot the design matrix
fig, axes = plt.subplots(figsize=(10, 6), constrained_layout=True)
nilearn.plotting.plot_design_matrix(design_matrix, axes=axes)


# %%
#
# Estimate the GLM model

# (clear channel names because mne_nirs plot_topo doesn't have the
# option to hide sensor names, and we have a LOT)
mne.channels.rename_channels(
    raw.info, {ch: "" for ch in raw.info["ch_names"]}, allow_duplicates=True
)
glm_est = mne_nirs.statistics.run_glm(raw, design_matrix, noise_model="auto")

# %%
#
# Now plot the GLM results of the comparison between tapping and
# control task on the scalp


# A hack to remove channel markers from a topoplot
def remove_markers_from_topo(topo):
    for marker in topo.findobj(match=matplotlib.collections.PathCollection):
        marker.set_visible(False)
    for marker in topo.findobj(match=matplotlib.spines.Spine):
        marker.set_visible(False)
    for marker in topo.findobj(match=matplotlib.text.Text):
        if marker.get_text() == "-":
            marker.set_visible(False)


fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True, ncols=2, nrows=2)
topo = glm_est.plot_topo(conditions=["Tapping/Left", "Tapping/Right"], axes=ax)
remove_markers_from_topo(topo)


# %%
#
# compute simple contrasts: LEFT, RIGHT, and LEFT - RIGHT
contrast_matrix = np.eye(2)
basic_conts = dict(
    [
        (column, contrast_matrix[i])
        for i, column in enumerate(design_matrix.columns)
        if i < 2
    ]
)
contrast_LvR = basic_conts["Tapping/Left"] - basic_conts["Tapping/Right"]
contrast = glm_est.compute_contrast(contrast_LvR)
topo = contrast.plot_topo()
remove_markers_from_topo(topo)
