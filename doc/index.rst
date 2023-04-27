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

Before installing MNE-NIRS you must install Python and MNE-Python.
To install Python and MNE-Python follow `these instructions <https://mne.tools/stable/install/index.html>`_.
We recommend using the standalone installer option unless you are a python expert.


Install via Standalone Installer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Download the installer from `the MNE-Python website <https://mne.tools/stable/install/index.html>`_.
2. Run the installer and wait for it to complete.
3. Once the installer is complete you will have an MNE-Python directory in your Applications folder.
4. Double click on the `Prompt (MNE)`
5. Once the prompt has loaded, run `jupyter lab` and it will open Jupyter in your web browser.
6. Now try one of the MNE-NIRS example notebooks by clicking `Download Jupyter notebook` at the bottom of any example.
7. Use the file browser in the jupyter lab web interface (top left of page) to browse to where you downloaded the example to. Double click to open it.

If you need the latest version of MNE-Python or MNE-NIRS you can enter the following at the prompt:

.. code:: console

   $ pip install -U --no-deps git+https://github.com/mne-tools/mne-python.git@main
   $ pip install -U --no-deps git+https://github.com/mne-tools/mne-nirs.git@main


Install via Conda or Pip
~~~~~~~~~~~~~~~~~~~~~~~~

You must use MNE-Python v1.0 or above.
Follow the installation instructions on the
`MNE-Python documentation site <https://mne.tools/dev/install/manual_install.html>`_.

Then run the following code to install MNE-NIRS:

.. code:: console

    $ pip install mne-nirs


Or if you wish to run the latest development version of MNE-NIRS:

.. code:: console

    $ pip install git+https://github.com/mne-tools/mne-nirs.git@main


Upgrading your software version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the
`MNE-Python instructions for how to update <https://mne.tools/dev/install/updating.html>`_
the MNE-Python version.
Similarly, you can update MNE-NIRS to the latest development version by running


.. code:: console

   $ pip install -U --no-deps git+https://github.com/mne-tools/mne-nirs.git@main


Acknowledgements
----------------

This library is built on top of other great packages. If you use MNE-NIRS you should also acknowledge these packages:

MNE: https://mne.tools/dev/overview/cite.html

Nilearn: http://nilearn.github.io/authors.html#citing

statsmodels: https://www.statsmodels.org/stable/index.html#citation

Until there is a journal article specifically on MNE-NIRS, please cite `this article <https://doi.org/10.1117/1.NPh.8.2.025008>`_.

.. toctree::
   :maxdepth: 2
   :hidden:

   auto_examples/index
   api
   changelog
