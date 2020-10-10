MNE-NIRS: Near-Infrared Spectroscopy Analysis
=============================================

API UNDER DEVELOPMENT: Feedback is appreciated.
----------------------------------------------------------------------------

.. image:: https://img.shields.io/badge/docs-master-brightgreen
    :target: https://mne.tools/mne-nirs/
    
.. image:: https://travis-ci.com/mne-tools/mne-nirs.svg?branch=master
    :target: https://travis-ci.com/mne-tools/mne-nirs
    
.. image:: https://codecov.io/gh/mne-tools/mne-nirs/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mne-tools/mne-nirs

**MNE-NIRS** is an `MNE <https://mne.tools>`_ compatible near-infrared spectroscopy processing package. 

MNE has support for some common analysis procedures (see `tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/plot_70_fnirs_processing.html>`_), this package adds additional helper functions, algorithms, and plotting. Functionality that is widely used from this package may be merged in to MNE.


Documentation
-------------

Documentation for this project is hosted `here <https://mne-tools.github.io/mne-nirs>`_.


Examples
--------

- MNE examples:
    - `Basic waveform analysis pipeline <https://mne.tools/dev/auto_tutorials/preprocessing/plot_70_fnirs_processing.html#sphx-glr-auto-tutorials-preprocessing-plot-70-fnirs-processing-py>`_.
    - `Artifact rejection discussion <https://mne.tools/dev/auto_examples/preprocessing/plot_fnirs_artifact_removal.html#ex-fnirs-artifacts>`_.
- MNE-NIRS examples:
    - `Simulated haemodynamic response GLM analysis <https://mne.tools/mne-nirs/auto_examples/plot_11_hrf_simulation.html>`_.
    - `Measured haemodynamic response GLM analysis <https://mne.tools/mne-nirs/auto_examples/plot_10_hrf.html>`_.
    - `Group level GLM analysis <https://mne.tools/mne-nirs/auto_examples/plot_12_group_glm.html>`_.
    - `Signal enhancement <https://mne-tools.github.io/mne-nirs/auto_examples/plot_20_cui.html>`_.
    - `Frequency and filtering <https://mne.tools/mne-nirs/auto_examples/plot_30_frequency.html>`_.


Contributing
------------

Contributions are welcome (more than welcome!). Contributions can be feature requests, improved documentation, bug reports, code improvements, new code, etc. Anything will be appreciated. *Note*: this package adheres to the same contribution  `standards as MNE <https://mne.tools/stable/install/contributing.html>`_.


Acknowledgements
----------------

This package is built on top of many other great packages. These should be acknowledged if you use this package.

MNE: https://mne.tools/dev/overview/cite.html

Nilearn: http://nilearn.github.io/authors.html#citing

Please also cite me. There is not currently a paper specifically on MNE-NIRS, so please find a 
`relevant paper of mine to cite from here <https://scholar.google.com/citations?user=LngqH5sAAAAJ&hl=en>`_.
