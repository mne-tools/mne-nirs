# -*- coding: utf-8 -*-
r"""
.. _tut-gowerlabs-data:

===================
Read Gowerlabs data
===================

High Density Diffuse Optical Tomography (HD-DOT) results in a greater
number of channels than traditional fNIRS devices.

`Gowerlabs. <https://www.gowerlabs.co.uk>`__
produces the `Lumo. <https://www.gowerlabs.co.uk/lumo>`__,
a HD-DOT device.
This tutorial demonstrates how to load data from Gowerlabs devices,
including how to utilise 3D digitisation information collected with
the HD-DOT measurement.

Data should be collected using the guidelines provided by Gowerlabs.
Once collected, the data should be converted to the SNIRF format using
`the lumomat software. <https://github.com/Gowerlabs/lumomat>`__.


HD-DOT data is often
collected with individual registration of the sensor positions. In this
tutorial we demonstrate how to load HD-DOT data from a Gowerlabs Lumo device,
co-register the channels to a head, and visualise the resulting channel space.

This tutorial uses the 3d graphical functionality provided by MNE-Python,
to ensure you have all the required packages installed we recommend using the
`official MNE installers. <https://mne.tools/stable/install/index.html>`__
"""

# %%

import os.path as op
import mne
from mne.datasets.testing import data_path

from mne.viz import set_3d_view


# sphinx_gallery_thumbnail_number = 1

# %%
# Import Gowerlabs Example File
# -----------------------------
# First, we must instruct the software where the file we wish to import
# resides. In this example we will use a small test file that is
# included in the MNE testing data set. To load your own data, replace
# the path stored in the `fname` variable by
# running `fname = /path/to/data.snirf`.
#
# .. note:: This will be updated to demonstrate how to load data with a
#           greater number of tiles and more meaningful measurement.
#           In the meantime, a small sample file is used.

testing_path = data_path(download=True)
fname = op.join(testing_path, 'SNIRF', 'GowerLabs', 'lumomat-1-1-0.snirf')

# %%
# We can view the path to the data by calling the variable `fname`.

fname


# %%
# To load the file we call the function :func:`mne:mne.io.read_raw_snirf`.

raw = mne.io.read_raw_snirf(fname, preload=True)

# %%
# We can then look at the file metadata by calling the variable `raw`.

raw

# %%
# Visualise Data
# --------------
# Next, we visualise the raw data to visually inspect the data quality.

raw.plot()

# %%
# By looking at the traces above we see that there are no flat channels
# and the signal includes event annotations. Next, we can view the
# annotations to ensure that they match what we expect from our experiment.
# Annotations provide a flexible tool to represent events in your
# experiment. They can also be used to annotate other useful information
# such as bad segments of data, participant movements, etc.

raw.annotations


# %%
# And we observe that there were six `A` annotations, one `Cat` annotation,
# and two `Dog` annotations. We can view the specific data for each annotation
# by converting the annotations to a dataframe.

raw.annotations.to_data_frame()


# %%
# View Optode Positions in 3D Space
# ---------------------------------
# The position of optodes in 3D space is recorded and stored in the SNIRF file.
# These positions are stored in head coordinate frame,
# for a detailed overview of coordinate frames and how they are handled in MNE
# see :ref:`mne:tut-source-alignment`.
# The position of each optode is stored, along with scalp landmarks (“fiducials”).
# These positions are in an arbitrary space, and must be aligned to a scan of
# the participants, or a generic, head.
#
# For this data, we do not have a MRI scan of the participants head.
# Instead, we will align the positions to a generic head created from
# a collection of 40 MRI scans of real brains called
# `fsaverage. <https://mne.tools/stable/auto_tutorials/forward/10_background_freesurfer.html#fsaverage>`__.
#
# First, lets just look at the sensors in arbitrary space.
# Below we see that there are three lumo tiles, each with three sources
# and four detectors.

subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir)

brain = mne.viz.Brain('fsaverage', subjects_dir=subjects_dir, alpha=0.0, cortex='low_contrast', background="w")
brain.add_sensors(raw.info, trans='fsaverage', fnirs=["sources", "detectors"])
brain.show_view(azimuth=130, elevation=80, distance=700)


# %%
# Coregister Optodes to Template Head
# -----------------------------------
# The optode locations displayed above are floating in free space
# and need to be aligned to our chosen head.
# First, lets just look at the fsaverage head we will use.

plot_kwargs = dict(subjects_dir=subjects_dir,
                   surfaces="brain", dig=True, eeg=[],
                   fnirs=['sources', 'detectors'], show_axes=True,
                   coord_frame='head', mri_fiducials=True)

fig = mne.viz.plot_alignment(trans="fsaverage", subject="fsaverage", **plot_kwargs)
set_3d_view(figure=fig, azimuth=90, elevation=0, distance=1)


# %%
# This is what a head model will look like. If you have an MRI from
# the participant you can use freesurfer to generate the required files.
# For further details on generating freesurfer reconstructions see
# :ref:`mne:tut-freesurfer-reconstruction`.
#
# In the figure above you can see the brain in grey. You can also
# see the MRI fiducial positions marked with diamonds.
# The nasion fiducial is marked in green, the left and right
# preauricular points (LPA and RPA) in red and blue respectively.
#
# Next, we can plot the positions of the optodes with the head model.

fig = mne.viz.plot_alignment(raw.info, trans="fsaverage", subject="fsaverage", **plot_kwargs)
set_3d_view(figure=fig, azimuth=90, elevation=0, distance=1)

# %%
# In the figure above we can see the Gowerlabs optode data
# and the fiducials that were digitised with the recording in circles.
# The digitised fiducials are a large distance from fiducials
# on the head. To coregister the

coreg = mne.coreg.Coregistration(raw.info, "fsaverage", subjects_dir, fiducials="estimated")
coreg.fit_fiducials(lpa_weight=1., nasion_weight=1., rpa_weight=1.)
mne.viz.plot_alignment(raw.info, trans=coreg.trans, subject="fsaverage", **plot_kwargs)


# %%
# A

set_3d_view(figure=fig, azimuth=90, elevation=0, distance=1)


# %%
# A

brain = mne.viz.Brain('gowerlabsdemodata', subjects_dir=subjects_dir, background='w', cortex='0.5', alpha=0.3)
brain.add_sensors(raw.info, trans=coreg.trans, fnirs=['sources', 'detectors'])