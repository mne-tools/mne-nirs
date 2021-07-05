.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


MNE-NIRS
========

This is a library to assist with processing near-infrared spectroscopy data with MNE.


Installation
------------

To install python and MNE follow `these instructions <https://mne.tools/dev/install/mne_python.html>`_.

Run the following code to install MNE-NIRS.

.. code:: bash

    >>> pip install mne-nirs


To load MNE-NIRS add these lines to your script.

.. code:: python

    >>> import mne
    >>> import mne_nirs


Usage
-----

See the `examples <auto_examples/index.html>`_ and `API documentation <api.html>`_.


Features
--------

MNE-NIRS and MNE-Python provide a wide variety of tools to use when processing NIRS data including:

* Loading data from a wide variety of devices, including SNIRF files.
* Apply 3D sensor locations from common digitisation systems such as Polhemus.
* Standard preprocessing including optical density calculation and Beer-Lambert Law conversion, filtering, etc.
* Data quality metrics including scalp coupling index and peak power.
* GLM analysis with a wide variety of cusomisation including including FIR or canonical HRF analysis, higher order autoregressive noise models, short channel regression, region of interest analysis, etc.
* Visualisation tools for all stages of processing from raw data to processed waveforms, GLM result visualisation, including both sensor and cortical surface projections.
* Data cleaning functions including popular short channel techniques and negative correlation enhancement.
* Group level analysis using (robust) linear mixed effects models and waveform averaging.
* And much more! Check out the documentation examples and the API for more details.


Acknowledgements
----------------

This library is built on top of other great packages. If you use MNE-NIRS you should also acknowledge these packages.

MNE: https://mne.tools/dev/overview/cite.html

Nilearn: http://nilearn.github.io/authors.html#citing

statsmodels: https://www.statsmodels.org/stable/index.html#citation

Until there is a journal article specifically on MNE-NIRS, please cite `this article <https://www.biorxiv.org/content/10.1101/2020.12.22.423886v1>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/index
   api
   changelog
