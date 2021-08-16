:orphan:

###
API
###

This page describes the application programming interface of MNE-NIRS.
The functions and classes of this package are described below.
The description of each interface includes links to the examples
relevant to that code.

This library extends the fNIRS functionality available within
`MNE-Python <https://mne.tools>`_.
When analysing fNIRS data with these tools you are likely to use functions
from both MNE-Python and MNE-NIRS.
As such, documentation is provided below for relevant functions and classes
from both packages.
General neuroimaging functionality provided by MNE-Python such as filtering,
epoching, visualisation, etc is not included here and can be found in the
`MNE-Python API page <https://mne.tools/dev/python_reference.html>`_.


.. currentmodule:: mne_nirs

.. toctree::


Input/Output
------------

MNE-Python
^^^^^^^^^^

.. currentmodule:: mne.io

.. automodule:: mne.io
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   read_raw_hitachi
   read_raw_nirx
   read_raw_snirf
   read_raw_boxy


MNE-NIRS
^^^^^^^^

.. currentmodule:: mne_nirs.io.snirf

.. automodule:: mne_nirs.io.snirf
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   write_raw_snirf


Preprocessing
-------------

MNE-Python
^^^^^^^^^^

.. currentmodule:: mne.preprocessing.nirs

.. automodule:: mne.preprocessing.nirs
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   optical_density
   beer_lambert_law
   source_detector_distances
   short_channels
   scalp_coupling_index
   temporal_derivative_distribution_repair


MNE-NIRS
^^^^^^^^

Signal Enhancement


.. currentmodule:: mne_nirs.signal_enhancement

.. automodule:: mne_nirs.signal_enhancement
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   enhance_negative_correlation
   short_channel_regression


Data quality evaluation.

.. currentmodule:: mne_nirs.preprocessing

.. automodule:: mne_nirs.preprocessing
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   peak_power
   scalp_coupling_index_windowed


Experimental Design
-------------------

.. currentmodule:: mne_nirs.experimental_design

.. automodule:: mne_nirs.experimental_design
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   make_first_level_design_matrix
   create_boxcar


Statistics
----------

First level analysis
^^^^^^^^^^^^^^^^^^^^

Individual (first) level analysis functions.

.. currentmodule:: mne_nirs.statistics

.. automodule:: mne_nirs.statistics
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   run_glm


Individual (first) level result classes.

.. autosummary::
   :toctree: generated/

   RegressionResults
   ContrastResults
   read_glm

Individual (first) level result class methods.
View the class documentation above for a detailed list of methods.

.. currentmodule:: mne_nirs.statistics

.. automodule:: mne_nirs.statistics
   :no-members:
   :no-inherited-members:
   :noindex:

.. autosummary::
   :toctree: generated/

   RegressionResults.compute_contrast
   RegressionResults.to_dataframe
   RegressionResults.to_dataframe_region_of_interest
   RegressionResults.scatter
   RegressionResults.plot_topo
   RegressionResults.surface_projection
   RegressionResults.scatter
   RegressionResults.save

   ContrastResults.to_dataframe
   ContrastResults.plot_topo
   ContrastResults.scatter
   ContrastResults.save


Second level analysis
^^^^^^^^^^^^^^^^^^^^^

Group (second) level analysis functions.

.. currentmodule:: mne_nirs.statistics

.. automodule:: mne_nirs.statistics
   :no-members:
   :no-inherited-members:
   :noindex:

.. autosummary::
   :toctree: generated/

   statsmodels_to_results


Visualisation
------------------

fNIRS specific data visualisation.

.. currentmodule:: mne_nirs.visualisation

.. automodule:: mne_nirs.visualisation
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   plot_nirs_source_detector


GLM result visualisation.

.. currentmodule:: mne_nirs.visualisation

.. automodule:: mne_nirs.visualisation
   :no-members:
   :no-inherited-members:
   :noindex:

.. autosummary::
   :toctree: generated/

   plot_glm_topo
   plot_glm_contrast_topo
   plot_glm_group_topo
   plot_glm_surface_projection


Data quality visualisation.

.. currentmodule:: mne_nirs.visualisation

.. automodule:: mne_nirs.visualisation
   :no-members:
   :no-inherited-members:
   :noindex:

.. autosummary::
   :toctree: generated/

   plot_timechannel_quality_metric


Simulation
----------

.. currentmodule:: mne_nirs.simulation

.. automodule:: mne_nirs.simulation
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   simulate_nirs_raw


Channels
--------

Functions to help with handling channel information.

.. currentmodule:: mne_nirs.channels

.. automodule:: mne_nirs.channels
   :no-members:
   :no-inherited-members:

.. autosummary::
   :toctree: generated/

   get_short_channels
   get_long_channels
   picks_pair_to_idx
