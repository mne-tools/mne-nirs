MNE-NIRS: Near-Infrared Spectroscopy Analysis
=============================================

.. image:: https://img.shields.io/badge/docs-master-brightgreen
    :target: https://mne.tools/mne-nirs/
    
.. image:: https://github.com/mne-tools/mne-nirs/workflows/linux%20/%20pip/badge.svg
    :target: https://github.com/mne-tools/mne-nirs/actions?query=workflow%3A%22linux+%2F+pip%22
    
.. image:: https://github.com/mne-tools/mne-nirs/workflows/macos%20/%20conda/badge.svg
    :target: https://github.com/mne-tools/mne-nirs/actions?query=workflow%3A%22macos+%2F+conda%22
    
.. image:: https://github.com/mne-tools/mne-nirs/workflows/linux%20/%20conda/badge.svg
    :target: https://github.com/mne-tools/mne-nirs/actions?query=workflow%3A%22linux+%2F+conda%22
    
.. image:: https://codecov.io/gh/mne-tools/mne-nirs/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/mne-tools/mne-nirs
    
.. image:: https://badge.fury.io/py/mne-nirs.svg
    :target: https://badge.fury.io/py/mne-nirs

**MNE-NIRS** is an `MNE-Python <https://mne.tools>`_ compatible near-infrared spectroscopy processing package. 

MNE-Python provides support for a subset of fNIRS waveform analysis, this package extends that functionality and adds additional GLM style analysis, helper functions, algorithms, data quality metrics, and plotting.


Documentation
-------------

Documentation for this project is hosted `here <https://mne-tools.github.io/mne-nirs>`_.

You can find a list of  `examples within the documentation <https://mne.tools/mne-nirs/master/auto_examples/index.html>`_.


Features
--------

MNE-NIRS and MNE-Python provide a wide variety of tools to use when processing NIRS data including:

* Loading data from a `wide variety of devices <https://mne.tools/mne-nirs/master/auto_examples/general/plot_01_data_io.html>`_, including `SNIRF files <https://mne.tools/mne-nirs/master/auto_examples/general/plot_19_snirf.html>`_.
* Standard preprocessing including `optical density calculation and Beer-Lambert Law conversion <https://mne.tools/mne-nirs/master/auto_examples/general/plot_15_waveform.html#id2>`_, filtering, etc.
* Data quality metrics including `scalp coupling index <https://mne.tools/mne-nirs/master/auto_examples/general/plot_15_waveform.html#id3>`_ and `peak power <https://mne.tools/mne-nirs/master/auto_examples/general/plot_22_quality.html#peak-power>`_.
* GLM analysis with a wide variety of cusomisation including `including FIR <https://mne.tools/mne-nirs/master/auto_examples/general/plot_13_fir_glm.html>`_ or canonical HRF analysis, higher order autoregressive noise models, `short channel regression, region of interest analysis <https://mne.tools/mne-nirs/master/auto_examples/general/plot_11_hrf_measured.html>`_, etc.
* Visualisation tools for all stages of processing from raw data to processed waveforms, GLM result visualisation, including both sensor and cortical surface projections.
* Data cleaning functions including popular short channel techniques and negative correlation enhancement.
* Group level analysis using `(robust) linear mixed effects models <https://mne.tools/mne-nirs/master/auto_examples/general/plot_12_group_glm.html>`_ and `waveform averaging <https://mne.tools/mne-nirs/master/auto_examples/general/plot_16_waveform_group.html>`_.
* And much more! Check out the documentation `examples <https://mne.tools/mne-nirs/master/auto_examples/index.html>`_ and the API `for more details <https://mne.tools/mne-nirs/master/api.html>`_.


Contributing
------------

Contributions are welcome (more than welcome!). Contributions can be feature requests, improved documentation, bug reports, code improvements, new code, etc. Anything will be appreciated. *Note*: this package adheres to the same contribution  `standards as MNE <https://mne.tools/stable/install/contributing.html>`_.


Acknowledgements
----------------

This package is built on top of many other great packages. If you use MNE-NIRS you should also acknowledge these packages.

MNE-Python: https://mne.tools/dev/overview/cite.html

Nilearn: http://nilearn.github.io/authors.html#citing

statsmodels: https://www.statsmodels.org/stable/index.html#citation

Until there is a journal article specifically on MNE-NIRS, please cite `this article <https://www.biorxiv.org/content/10.1101/2020.12.22.423886v1>`_.
