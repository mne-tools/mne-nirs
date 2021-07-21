.. project-template documentation master file, created by
   sphinx-quickstart on Mon Jan 18 14:44:12 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


MNE-NIRS
========

This is a library to assist with processing near-infrared spectroscopy data with MNE.


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


Installation
------------

To install python and MNE follow `these instructions <https://mne.tools/dev/install/mne_python.html>`_.
Some NIRS functionality is only available in the latest version of MNE-Python,
so I suggest installing the latest development version `as described here <https://mne.tools/dev/install/advanced.html#using-the-development-version>`_.


Run the following code to install MNE-NIRS.

.. code:: bash

    >>> pip install mne-nirs


To load MNE-NIRS add these lines to your script.

.. code:: python

    >>> import mne
    >>> import mne_nirs


Alternative installation options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you wish to try using MNE-NIRS without installing a python environment on your computer,
there are two options available.

If you wish to run the examples using a cloud server, you can use the binder links located at the bottom of each example.
Clicking these binder links will spin up an online server instance for you to use.
This will allow you to edit and run code, upload data, visualise results, etc, 
without needing to download and install anything on your computer.
However, as this is a free cloud server the computation may be slower, 
and data on these cloud binder instances will regularly be reset,
so this approach is best used for quickly exploring the capabilities of MNE-NIRS. 

Alternatively, if you wish to run code locally on your own computer with your own data, you can run a 
`docker instance locally <https://docs.docker.com/get-docker/>`_
using the 
`MNE-NIRS-docker <https://github.com/rob-luke/mne-nirs-docker>`_ image.
See the web page for detailed instructions.
Using docker provides a notebook server running on your own computer, 
it comes pre-prepared with MNE-Python, MNE-NIRS, and other useful packages installed.
This approach gets you up and running with a single command, and provides
the greatest flexibility without installing python.


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
