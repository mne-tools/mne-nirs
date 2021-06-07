"""
.. _ex-fnirs-quality:

=========================
Signal Quality Evaluation
=========================

This tutorial demonstrates how signal quality can be evaluated using MNE-NIRS.

.. sidebar:: Relevant literature

   Pollonini L et al., “PHOEBE: a method for real time mapping of
   optodes-scalp coupling in functional near-infrared spectroscopy” in
   Biomed. Opt. Express 7, 5104-5119 (2016).

   Hernandez, Samuel Montero, and Luca Pollonini. "NIRSplot: a tool for
   quality assessment of fNIRS scans." Optics and the Brain.
   Optical Society of America, 2020.

Ensuring your data is of high quality is essential to good scientific research.
Evaluating data quality is an essential part of both data collection and
analysis.
For assessing data quality during data acquisition tools such as PHOEBE are
highly recommended, these allow the researcher to address potential problems
immediately and modify their experiment to improve the quality of recorded
data.
It is also important to assess data quality during data analysis.
MNE-Python and MNE-NIRS provides several mechanisms to allow researchers
to evaluate the quality of their data and to include this information in their
downstream processing.
Dedicated tools exist for quality evaluatiuon such as Hernandez (2020).
This tutorial demonstrates methods in MNE-NIRS and MNE-Python for determining
channels with poor signal
quality, and methods for determining time segments of data that are of low
quality in a subset of channels.

Two methods are introduced in this tutorial. The scalp coupling index (SCI)
and peak power (PP) metrics. Both of these methods examine the signal for
the presence of a heart beat signal, which indicates that the sensors
were in contact with the scalp. For further details see the papers listed
in the relevant literature sidebar.

"""
# sphinx_gallery_thumbnail_number = 3

# Authors: Robert Luke <mail@robertluke.net>
#
# License: BSD (3-clause)

import os
import mne
import numpy as np
from itertools import compress
import matplotlib.pyplot as plt

from mne.preprocessing.nirs import optical_density
from mne_nirs.preprocessing import peak_power
from mne_nirs.visualisation import plot_timechannel_quality_metric

###############################################################################
# Import data
# -----------
#
# Here we will work with the :ref:`fNIRS motor data <fnirs-motor-dataset>`.
# We resample the data to make indexing exact times more convenient.
# We then convert the data to optical density and plot the raw signal.

fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
fnirs_cw_amplitude_dir = os.path.join(fnirs_data_folder, 'Participant-1')
raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
raw_intensity.load_data().resample(4.0, npad="auto")
raw_od = optical_density(raw_intensity)
raw_od.plot(n_channels=55, duration=4000, show_scrollbars=False, clipping=None)

###############################################################################
# From the above plot we observe that the data is relatively clean.
# Later we will add some artificial bad sections to demonstrate the quality
# evaluation metrics.


###############################################################################
# Scalp Coupling Index
# --------------------
#
# The scalp coupling index (SCI) from Pollonini (2016) provides a measure of
# the quality of the signal for a channel as determined using the entire
# measurement duration. See Pollonini (2016) for further details of the
# theory and implementation. Here we calculate the SCI for each channel
# and view the distribution of values.

sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)
fig, ax = plt.subplots()
ax.hist(sci)
ax.set(xlabel='Scalp Coupling Index', ylabel='Count', xlim=[0, 1])

###############################################################################
# We observe that most of the channels have a good SCI of 1, but a few channels
# have a poorer score. We can list the channels with an SCI below a threshold.
# And we can mark these as bad in the MNE data. This way the functions
# downstream will know that the data in these channels is bad.
# For this example we set the threshold for a bad channel to be SCI < 0.8.
# We then print a list of the bad channels and observe their are 10 channels
# (five source-detector pairs) that are marked as bad.

raw_od.info['bads'] = list(compress(raw_od.ch_names, sci < 0.7))
print(raw_od.info['bads'])

###############################################################################
# We can plot the time course of the signal again and note that the bad
# channels are now displayed in grey to indicate they are bad.

raw_od.plot(n_channels=55, duration=4000, show_scrollbars=False, clipping=None)

###############################################################################
# Peak Power
# ----------
#
# It may also be informative to view the quality of the signal at a finer
# time resolution. The Peak Power metric provides a quality metric evalauted
# over a 10 second window. This allows the user to view instances where
# a subset of channels may be contaminated by artifacts for a short duration
# of the recording.
# This plot is based on the meg bad channel detection figures
# available in mne-bids-pipeline.

raw_od, scores, times = peak_power(raw_od)
plot_timechannel_quality_metric(raw_od, scores, times, threshold=0.1)


###############################################################################
# Introduced Noise
# ----------------
#
# The above data is quite clean, so here we add some noise to channels to
# demonstrate that the algorithm is able to detect bad segments of data.
# We add a step like signal at two time instances to mimic an artifact.
# We add one artifact to channel S2-D4 at 1000 seconds
# and one artifact to channel S5-D13 at 2000 seconds.

# Add an artifact to channel S2-D4 at time 1000 seconds
raw_od._data[12, 4000:4080] = np.linspace(0, -0.5, 80) + raw_od._data[12, 4000]

# Add an artifact to channel S5-D13 at time 2000 seconds
raw_od._data[34, 8000:8080] = np.linspace(0, 0.5, 80) + raw_od._data[34, 8000]

###############################################################################
# Next we plot just these channels to demonstrate that indeed an artifact
# has been added.

raw_od.copy().pick(picks = [12, 13, 34, 35]).\
    plot(n_channels=55,duration=40000, show_scrollbars=False,
         clipping=None, scalings={'fnirs_od': 0.2})


###############################################################################
# Peak Power Metric
# -----------------
#
# To determine specific time and channel instances where data is of low
# quality the peak power metric can be utilise (Pollonini, 2016).
# Below the peak power is calculated and displayed for all channels.
# The default threshold of 0.1 is applied to the quality scores to determine
# bad segments of data.
# The left facet illustrates the raw quality scores.
# The right facet illustrates the channel and time segments that do not meet
# the threshold criteria and are marked as bad.

raw_od, scores, times = peak_power(raw_od)
plot_timechannel_quality_metric(raw_od, scores, times, threshold=0.1)


###############################################################################
# Annotations
# -----------
#
# Similar to how entire channels were marked as bad above, the peak power
# function annotates the raw data structure to indicate where the bad
# segments of data are. If we replot the subset of channels as above we note
# that the bad segments are marked in red now, indicating this time section
# contained bad data.
# Note that although the red line appears as if the bad marking is present
# for all channels, this is due to the plotting code. Internally the software
# knows that only certain channels had a bad segment, and downstream processing
# will only treat the specified bad channels as bad. I.e. when generating
# epochs for time 1000 s, no epochs would be generated for channel S2-D4, but
# channels would be generated for S5-D13 (as the artifact was only present
# on S2-D4).

raw_od.copy().pick(picks = [12, 13, 34, 35]).\
    plot(n_channels=55,duration=40000, show_scrollbars=False,
         clipping=None, scalings={'fnirs_od': 0.2})

###############################################################################
# These channel and time specific annotations are used by downstream
# processing. For example, when extracting epochs if a specific channel
# has an annotation indicating a bad segment in a specific time window, then
# the epochs will be discarded for that specific channel.
# Finally, we display the entire time signal after both channel and time
# specific quality evaluation. Note that bad channels are now greyed out
# and bad time segments are marked in red.

raw_od.plot(n_channels=55, duration=4000, show_scrollbars=False, clipping=None)

###############################################################################
# Conclusion
# ----------
#
# Two data quality metrics were presented and plotted.
# One metric for determining a bad channel (scalp coupling index).
# And one metric for determining specific time and channel segments that are
# bad.
# It was demonstrated how bad segments are visualised and how bad segments
# are treated in downstream analysis.
