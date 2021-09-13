"""
.. _tut-fnirs-freq:

Mayer Wave Parametrisation
==========================

.. sidebar:: Cite the FOOOF authors

   This tutorial is only possible due to the excellent and generous work of
   the FOOOF authors. Please visit their site https://fooof-tools.github.io/fooof.
   Also be sure to cite their various papers:

   Donoghue, Thomas, et al. "Parameterizing neural power spectra into periodic and aperiodic components." Nature neuroscience 23.12 (2020): 1655-1665.

   Donoghue, Thomas, Julio Dominguez, and Bradley Voytek. "Electrophysiological frequency band ratio measures conflate periodic and aperiodic neural activity." Eneuro 7.6 (2020).

This example demonstrates how to use the
Fitting Oscillations & One Over F (FOOOF)
:footcite:`donoghue2020parameterizing`
method to quanitfy Mayer wave parameters in fNIRS data.
This is based on the description provided in
:footcite:`luke2021characterization`.

This tutorial is heavily based on the tutorials provided by the FOOOF
authors over at https://fooof-tools.github.io/fooof/auto_tutorials/index.html.
You should read their excellent documentation. Their work should be considered
the primary resource, and this is just an example of how to apply it to fNIRS
data for the purpose of extracting Mayer waves oscillation parameters.


.. contents:: Page contents
   :local:
   :depth: 2

"""

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import mne_nirs
import numpy as np
import matplotlib.pyplot as plt

from mne.preprocessing.nirs import optical_density, beer_lambert_law
from mne.time_frequency import psd_welch

from mne_nirs.channels import get_long_channels
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.simulation import simulate_nirs_raw
from mne_nirs.preprocessing import quantify_mayer_fooof

from fooof import FOOOF


# %%
# Import and preprocess data
# --------------------------
#
# We read in the data and convert to haemoglobin concentration.
#
# This code is similar to the first sections in the MNE tutorial,
# so will not be described in detail here.
# Please see the
# :ref:`MNE documentation <mne:tut-fnirs-processing>`.
# for a detailed introduction to processing NIRS with MNE.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_raw_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw = mne.io.read_raw_nirx(fnirs_raw_dir, verbose=True).load_data()

raw = optical_density(raw)
raw.resample(1.5)
raw = beer_lambert_law(raw)
raw = raw.pick(picks="hbo")
raw = get_long_channels(raw, min_dist=0.025, max_dist=0.045)
raw


# %%
# Process data with FOOOF
# -----------------------
#
# Next we estimate the power spectral density of the data and pass this to
# the FOOOF algorithm
#
# I recommend using the FOOOF algorithm as provided by the authors rather
# than reimplementation or custom plotting etc. As there code is of excellent
# quality, well maintained, thoroughly documented, and they have considered
# many edge cases.
#
# Below we plot the spectrum of the data and the FOOOF fit of oscillations
# and aperiodic component. Note the bump at 0.1 Hz that reflects the Mayer
# wave activity.

def scale_up_spectra(spectra, freqs):
    """
    FOOOF requires the frequency values to be higher than the fNIRS data
    permits, so we scale the values up by 10 here, and then will scale
    the frequency values down by 10 later.
    """
    freqs = freqs * 10
    return spectra, freqs

# Prepare data for FOOOF
spectra, freqs = psd_welch(raw, fmin=0.001, fmax=1, tmin=0, tmax=None, n_overlap=300, n_fft=600)
spectra, freqs = scale_up_spectra(spectra, freqs)

fm = FOOOF(peak_width_limits=(0.06, 12.0))
# Set the frequency range to fit the model
freq_range = [0.001, 10]

fm.fit(freqs, np.mean(spectra, axis=0), freq_range)

fig, axs = plt.subplots(1, 1, figsize=(10, 5))
fm.plot(plot_peaks=None, data_kwargs={'color': 'orange'}, plt_log=False, ax=axs, plot_aperiodic=True)
# Correct for x10 scaling above
plt.xticks([0, 1, 2, 3, 4, 5, 6], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


# %%
# Use Mayer wave convenience function provided by MNE-NIRS
# --------------------------------------------------------
#
# MNE-NIRS provides a convenient function to estimate the Mayer wave
# parameters that takes care of the frequency scaling and selects the component
# most likely associated with the Mayer wave. It returns this data in a pandas
# dataframe for your convenience.

print(quantify_mayer_fooof(raw.pick("hbo"), extra_df_fields={"Study": "Online tutorial"}))
