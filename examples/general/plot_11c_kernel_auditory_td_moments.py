"""
.. _tut-fnirs-kernel-audio-td:

Auditory GLM Analysis with TD-fNIRS Statistical Moments
========================================================

This example demonstrates how to analyze time-domain (TD) functional
near-infrared spectroscopy (fNIRS) data using the statistical moments
representation. We run a GLM on all three moments — intensity, mean time
of flight, and variance — and compare the Story < Noise contrast across
them.

The `dataset <https://openneuro.org/datasets/ds006545>`_ was collected with
the Kernel Flow 2 system, a high-density TD-fNIRS device with ~3500 channels.
The task is an auditory paradigm with alternating blocks of **Story** listening,
**Noise**, and **Silence**.

TD-fNIRS systems measure the full distribution of photon travel times
(the DTOF). From this distribution, three statistical moments are extracted
per channel and stored in the SNIRF file:

- **Moment 0 (intensity):** total photon count — analogous to CW amplitude
- **Moment 1 (mean time of flight):** average arrival time in picoseconds —
  sensitive to deeper (cortical) absorption changes
- **Moment 2 (variance):** temporal spread of the distribution

See :footcite:`Dubois2024` for details on this dataset and the
reliability of TD-fNIRS brain metrics.

.. footbibliography::

    Dubois2024
        Dubois, J., et al. (2024). Reliability of brain metrics derived from
        a Time-Domain Functional Near-Infrared Spectroscopy System.
        Scientific Reports, 14(1), 17500.
        https://doi.org/10.1038/s41598-024-68555-9

"""
# sphinx_gallery_thumbnail_number = 4

# Authors: Julien Dubois     <https://github.com/jcrdubois>
#
# License: BSD (3-clause)

import numpy as np
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from mne import events_from_annotations as get_events_from_annotations
from mne.io.snirf import read_raw_snirf
from mne.viz import plot_events, plot_topomap
from nilearn.plotting import plot_design_matrix

from mne_nirs.datasets import openneuro_kernel_audio
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm

# %%
# Download and load the data
# --------------------------
#
# We download a single subject's SNIRF file from
# `OpenNeuro ds006545 <https://openneuro.org/datasets/ds006545>`_
# and load it with MNE's SNIRF reader. After loading, we immediately
# resample to 1 Hz to reduce memory usage (~520 MB raw).

snirf_file = openneuro_kernel_audio.data_path()
raw = read_raw_snirf(snirf_file).load_data().resample(1)
sphere = (0.0, -0.02, 0.006, 0.1)

# %%
# Explore TD moment channel types
# --------------------------------
#
# The SNIRF file contains three types of TD moment channels for each
# source-detector pair and wavelength. Let's see what we have.

ch_types = np.array(raw.get_channel_types())
print("Channel types and counts:")
for t in np.unique(ch_types):
    print(f"  {t}: {np.sum(ch_types == t)}")

# %%
#
# Each source-detector pair × wavelength has 3 channels (one per moment),
# and there are two wavelengths (690 nm and 905 nm).
print(f"\nFirst 6 channel names: {raw.ch_names[:6]}")

# %%
# Select 905 nm channels for each moment type
# ---------------------------------------------
#
# We pick 905 nm channels which have better sensitivity to HbO absorption
# changes, and create a separate Raw object for each moment type.

moment_types = {
    "intensity": "fnirs_td_moments_intensity",
    "mean_tof": "fnirs_td_moments_mean",
    "variance": "fnirs_td_moments_variance",
}

raws = {}
for name, ch_type in moment_types.items():
    r = raw.copy().pick(ch_type)
    r.pick([ch for ch in r.ch_names if "905" in ch])
    raws[name] = r
    print(f"  {name}: {len(r.ch_names)} channels")

# %%
# Inspect the events
# ------------------
#
# The SNIRF reader parses the ``BlockType`` columns from the stim groups,
# giving us proper ``Story``, ``Noise``, and ``Silence`` annotations.
# For the GLM, we only need Story and Noise.

raw_ref = raws["mean_tof"]
raw_ref.annotations.to_data_frame()[["onset", "duration", "description"]]

# %%
#
# Drop non-stimulus annotations and plot the events.

for r in raws.values():
    r.annotations.delete(
        np.flatnonzero(
            np.isin(
                r.annotations.description,
                ["Silence", "StartExperiment", "StartRest"],
            )
        )
    )

events, event_id = get_events_from_annotations(raw_ref)
plot_events(events, event_id=event_id, sfreq=raw_ref.info["sfreq"])

# %%
# Visualize mean time of flight time courses
# -------------------------------------------
#
# Apply a bandpass filter and look at representative channels to see
# condition-related modulation of the mean time of flight signal.

raw_filt = raw_ref.copy().filter(
    0.01, 0.1, h_trans_bandwidth=0.01, l_trans_bandwidth=0.01
)

# Compute source-detector distances from the loc field (meters -> mm)
source_positions = np.array([ch["loc"][3:6] for ch in raw_ref.info["chs"]])
detector_positions = np.array([ch["loc"][6:9] for ch in raw_ref.info["chs"]])
sds = np.sqrt(np.sum((source_positions - detector_positions) ** 2, axis=1)) * 1000

idx_good = np.flatnonzero((sds > 15) & (sds < 30))
print(f"{len(idx_good)} channels with 15-30 mm separation")

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
times = raw_filt.times
for ax_i, ch_idx in enumerate(idx_good[:3]):
    ch_name = raw_filt.ch_names[ch_idx]
    data = raw_filt.get_data(picks=[ch_idx])[0]
    axes[ax_i].plot(times, data * 1e12)  # convert s -> ps for display
    axes[ax_i].set_ylabel("Mean ToF (ps)")
    axes[ax_i].set_title(ch_name)
    for ann in raw_filt.annotations:
        if ann["description"] == "Story":
            axes[ax_i].axvspan(
                ann["onset"], ann["onset"] + ann["duration"], alpha=0.15, color="blue"
            )
        elif ann["description"] == "Noise":
            axes[ax_i].axvspan(
                ann["onset"], ann["onset"] + ann["duration"], alpha=0.15, color="red"
            )
axes[-1].set_xlabel("Time (s)")

# %%
# GLM Design Matrix
# -----------------
#
# We set up a first-level GLM design matrix with the Glover HRF model.
# The stimulus duration is 20 seconds (length of each Story/Noise block).

stim_dur = 20.0
design_matrix = make_first_level_design_matrix(
    raw_ref,
    drift_model="polynomial",
    drift_order=1,
    high_pass=0.01,
    hrf_model="glover",
    stim_dur=stim_dur,
)
fig, ax = plt.subplots(figsize=(design_matrix.shape[1] * 0.5, 6), layout="constrained")
plot_design_matrix(design_matrix, axes=ax)

# %%
# Run GLM for each moment type
# -----------------------------
#
# The ``run_glm`` wrapper works with any fNIRS channel type — the
# underlying nilearn GLM is agnostic to the physical quantity being modeled.

glm_results = {}
for name, r in raws.items():
    print(f"Running GLM for {name}...")
    glm_results[name] = run_glm(r, design_matrix, noise_model="auto")

# %%
# Compute Story < Noise contrast for each moment
# ------------------------------------------------

contrast_matrix = np.eye(design_matrix.shape[1])
basic_conts = dict(
    [
        (column, contrast_matrix[i])
        for i, column in enumerate(design_matrix.columns)
        if column in ("Story", "Noise")
    ]
)
contrast_NvS = basic_conts["Noise"] - basic_conts["Story"]

contrasts = {}
for name, glm_est in glm_results.items():
    contrasts[name] = glm_est.compute_contrast(contrast_NvS)

# %%
# Visualize Story < Noise across all three moments
# --------------------------------------------------
#
# We plot the thresholded t-statistic topomaps for the Story < Noise
# contrast, one column per moment type. This allows direct comparison
# of how each statistical moment captures the auditory response.

cmap = plt.get_cmap("Reds").copy()
cmap.set_under((1, 1, 1, 0))
cmap.set_bad((1, 1, 1, 0))

tm_kwargs = dict(
    sensors=False,
    image_interp="linear",
    extrapolate="local",
    contours=0,
    show=False,
    sphere=sphere,
)

moment_labels = {
    "intensity": "Intensity\n(moment 0)",
    "mean_tof": "Mean ToF\n(moment 1)",
    "variance": "Variance\n(moment 2)",
}

thr_p = 0.01
vlim = (0.01, 5)

fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
for ax_i, (name, conest) in enumerate(contrasts.items()):
    ax = axes[ax_i]
    ax.set_facecolor("white")
    t_map = conest.data.stat()
    p_map = conest.data.p_value()
    t_map_masked = t_map.copy()
    t_map_masked[p_map > thr_p] = np.ma.masked
    ch_type = str(np.unique(conest.get_channel_types())[0])
    plot_topomap(
        t_map_masked,
        conest.info,
        axes=ax,
        vlim=vlim,
        ch_type=ch_type,
        cmap=cmap,
        **tm_kwargs,
    )
    ax.set_title(moment_labels[name], fontsize=14)

norm = mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    fraction=0.03,
    shrink=0.5,
    orientation="horizontal",
    label="t-statistic",
)
fig.suptitle(f"Story < Noise — 905 nm (p < {thr_p})", fontsize=16)

# %%
#
# At a more stringent threshold:
thr_p = 0.0001

fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
for ax_i, (name, conest) in enumerate(contrasts.items()):
    ax = axes[ax_i]
    ax.set_facecolor("white")
    t_map = conest.data.stat()
    p_map = conest.data.p_value()
    t_map_masked = t_map.copy()
    t_map_masked[p_map > thr_p] = np.ma.masked
    ch_type = str(np.unique(conest.get_channel_types())[0])
    plot_topomap(
        t_map_masked,
        conest.info,
        axes=ax,
        vlim=vlim,
        ch_type=ch_type,
        cmap=cmap,
        **tm_kwargs,
    )
    ax.set_title(moment_labels[name], fontsize=14)

norm = mcolors.Normalize(vmin=vlim[0], vmax=vlim[1])
fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    fraction=0.03,
    shrink=0.5,
    orientation="horizontal",
    label="t-statistic",
)
fig.suptitle(f"Story < Noise — 905 nm (p < {thr_p})", fontsize=16)

# %%
#
# Comparing the three moments reveals how different aspects of the photon
# travel time distribution capture the hemodynamic response. For TD-fNIRS
# moments, increased cortical absorption causes a *decrease* in photon
# intensity and arrival times, so the effect sign is reversed compared to
# hemoglobin concentration measures — hence we plot Noise minus Story
# (Story < Noise). The mean time of flight (moment 1) is expected to show
# enhanced sensitivity to cortical changes due to its depth selectivity,
# while intensity (moment 0) behaves similarly to continuous-wave
# measurements.
