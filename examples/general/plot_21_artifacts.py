"""
.. _ex-fnirs-artifacts:

==============================
Artifact Correction Techniques
==============================

Here we artificially introduce several artifacts in a
functional near-infrared spectroscopy (fNIRS) measurement and observe
how artifact correction techniques attempt to correct the data.

"""
# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os

import mne
from mne.preprocessing.nirs import (
    optical_density,
    temporal_derivative_distribution_repair,
)

from mne_nirs.preprocessing import (
    detect_motion_artifacts_by_channel,
    motion_correct_spline,
    motion_correct_wavelet,
    motion_detect_and_correct_wavelet,
)

# %%
# Import data
# -----------
#
# Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
# We resample the data to make indexing exact times more convenient.
# We then convert the data to optical density to perform corrections on
# and plot these signals.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = os.path.join(fnirs_data_folder, "Participant-1")
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data().resample(3, npad="auto")
raw_od = optical_density(raw_intensity)
new_annotations = mne.Annotations(
    [31, 187, 317], [8, 8, 8], ["Movement", "Movement", "Movement"]
)
raw_od.set_annotations(new_annotations)
raw_od.plot(n_channels=15, duration=400, show_scrollbars=False)

# %%
# We can see some small artifacts in the above data from movement around 40,
# 190 and 240 seconds. However, this data is relatively clean so we will
# add some additional artifacts below.


# %%
# Add artificial artifacts to data
# --------------------------------
#
# Two common types of artifacts in NIRS data are spikes and baseline shifts.
# Spikes often occur when a person moves and the optode moves relative to the
# scalp and then returns to its original position.
# Baseline shifts occur if the optode moves relative to the scalp and does not
# return to its original position.
# We add a spike type artifact at 100 seconds and a baseline shift at 200
# seconds to the data.

corrupted_data = raw_od.get_data()
corrupted_data[:, 298:302] = corrupted_data[:, 298:302] - 0.06
corrupted_data[:, 450:750] = corrupted_data[:, 450:750] + 0.03
corrupted_od = mne.io.RawArray(
    corrupted_data, raw_od.info, first_samp=raw_od.first_samp
)
new_annotations.append([95, 145, 245], [10, 10, 10], ["Spike", "Baseline", "Baseline"])
corrupted_od.set_annotations(new_annotations)

corrupted_od.plot(n_channels=15, duration=400, show_scrollbars=False)


# %%
# Apply temporal derivative distribution repair
# ---------------------------------------------
#
# This approach corrects baseline shift and spike artifacts without the need
# for any user-supplied parameters :footcite:`FishburnEtAl2019`.

corrected_tddr = temporal_derivative_distribution_repair(corrupted_od)
corrected_tddr.plot(n_channels=15, duration=400, show_scrollbars=False)


# %%
# We can see in the data above that the introduced spikes and shifts are
# largely removed, but some residual smaller artifact remains.
# The same can be said for the artifacts in the original data.


# %%
# Detect motion artifacts
# -----------------------
#
# The remaining techniques are ports of the Homer3 functions
# :footcite:`HuppertEtAl2009` and reproduce their output. They work in two
# stages: the motion artifacts are first detected and the flagged segments
# are then corrected.
# :func:`~mne_nirs.preprocessing.detect_motion_artifacts_by_channel` flags a
# sample when, within a window of ``t_motion`` seconds, the signal changes by
# more than ``stdev_thresh`` times the standard deviation of its first
# derivative, or by more than ``amp_thresh`` optical density units.
# Each flagged sample is padded by ``t_mask`` seconds.
# Both wavelengths of a source-detector pair are evaluated together and share
# one mask. The default thresholds are those of Homer3, but suitable values
# depend on the sampling rate and noise level of the recording, so we use
# stricter values here that pick up the artifacts in this resampled data.
# The returned mask is ``True`` for clean samples and ``False`` for motion.

mask = detect_motion_artifacts_by_channel(
    corrupted_od, stdev_thresh=10, amp_thresh=0.05
)
print(f"{100 * (~mask).mean():.1f}% of samples flagged as motion artifacts")


# %%
# Apply spline interpolation motion correction
# --------------------------------------------
#
# Spline interpolation :footcite:`ScholkmannEtAl2010` fits a smoothing spline
# to each flagged segment, subtracts it, and shifts the segments so that the
# signal is continuous across their boundaries. It is most effective for
# baseline shifts. The ``smoothing`` parameter is the ``p`` of Homer3 and
# MATLAB's ``csaps``; the default of ``0.99`` is the value recommended in the
# literature. The correction is applied channel by channel using the mask
# computed above.
# If no mask is passed, existing ``BAD`` annotations are used, or the
# artifacts are detected with the default parameters.

corrected_spline = motion_correct_spline(corrupted_od, mask=mask)
corrected_spline.plot(n_channels=15, duration=400, show_scrollbars=False)


# %%
# The baseline shift is largely flattened and the spike is reduced.
# Because segments are shifted to match their neighbours, the absolute level
# of the optical density can differ from the original recording.
# This constant offset does not affect the haemoglobin changes computed later.


# %%
# Apply wavelet motion correction
# -------------------------------
#
# Wavelet correction :footcite:`MolaviDumont2012` decomposes each channel
# with a translation-invariant wavelet transform and zeroes the coefficients
# that are outliers relative to the interquartile range, controlled by
# ``iqr``.
# It targets spikes and leaves slow changes such as baseline shifts in place.
# This function requires the ``PyWavelets`` package.

corrected_wavelet = motion_correct_wavelet(corrupted_od, iqr=1.5)
corrected_wavelet.plot(n_channels=15, duration=400, show_scrollbars=False)


# %%
# The spike at 100 seconds is removed while the baseline shift remains,
# so the spline and wavelet methods are complementary and are often applied
# one after the other.


# %%
# Detect and correct in one step
# ------------------------------
#
# :func:`~mne_nirs.preprocessing.motion_detect_and_correct_wavelet` runs the
# detection and the wavelet correction in one call, the usual Homer3
# processing stream. The detected segments are added to the annotations as
# ``BAD_motion`` so that they can be inspected or excluded from later
# analysis.

corrected_auto, mask_global = motion_detect_and_correct_wavelet(
    corrupted_od, stdev_thresh=10, amp_thresh=0.05
)
corrected_auto.plot(n_channels=15, duration=400, show_scrollbars=False)


# %%
# References
# ----------
#
# .. footbibliography::
