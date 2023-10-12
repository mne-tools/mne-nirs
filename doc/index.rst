MNE-NIRS
========

This is a library to assist with processing near-infrared spectroscopy data with MNE.


Usage
-----

See the `examples <auto_examples/index.html>`_ and `API documentation <api.html>`_.

Features
--------

.. include:: ../README.rst
   :start-after: .. features-start
   :end-before: .. features-end

Installation
------------

Before installing MNE-NIRS you must install Python and MNE-Python.
To install Python and MNE-Python follow `these instructions <https://mne.tools/stable/install/index.html>`_.
We recommend using the standalone installer option unless you are a python expert.


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
