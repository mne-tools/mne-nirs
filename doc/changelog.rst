:orphan:

.. _whats_new:

#########
Changelog
#########

MNE-NIRS follows `semantic versioning <https://semver.org/>`_.
Such that, version numbers are described as v<MAJOR>.<MINOR>.<PATCH>.
Major version changes indicate incompatible API changes.
Minor version changes indicate new functionality was added in a backwards compatible manner.
Patch version changes indicate backward compatible bug fixes.

To install a specific version of the library you would run ``pip install mne-nirs==0.0.6``, where ``0.0.6`` is the version you wish to install.


v0.1.3 development
------------------

General

* MNE-NIRS now requires the latest MNE-Python development version (main branch).


Enhancements

* Update SNIRF exporter to meet v1.0 validator requirements :meth:`mne_nirs.io.snirf.write_raw_snirf`. By `Robert Luke`_.
* Add ability to provide custom channel weighting in :meth:`mne_nirs.statistics.RegressionResults.to_dataframe_region_of_interest` computation. By `Robert Luke`_.


Fixes

* Fix bug when using no weighting in :meth:`mne_nirs.statistics.RegressionResults.to_dataframe_region_of_interest`. By `Robert Luke`_.


v0.1.2
------

General

* MNE-NIRS now uses the MNE-Python stable version v0.24 and no longer requires the development version.


Enhancements

* Add :meth:`mne_nirs.channels.list_sources`. By `Robert Luke`_.
* Add :meth:`mne_nirs.channels.list_detectors`. By `Robert Luke`_.
* Add :meth:`mne_nirs.channels.drop_sources`. By `Robert Luke`_.
* Add :meth:`mne_nirs.channels.drop_detectors`. By `Robert Luke`_.
* Add :meth:`mne_nirs.channels.pick_sources`. By `Robert Luke`_.
* Add :meth:`mne_nirs.channels.pick_detectors`. By `Robert Luke`_.
* Add :meth:`mne_nirs.preprocessing.quantify_mayer_fooof`. By `Robert Luke`_.
* Add :func:`mne_nirs.io.fold_landmark_specificity`. By `Robert Luke`_.
* Add :func:`mne_nirs.io.fold_channel_specificity`. By `Robert Luke`_.
* Added fetchers for two more publicly available datasets. By `Robert Luke`_.


v0.1.1
------

API changes

* :func:`mne_nirs.channels.get_long_channels` maximum optode distance reduced from 5 to 4.5 cm. By `Robert Luke`_.
* :func:`mne_nirs.experimental_design.create_boxcar` duration reduced from 5 to 1 second. By `Robert Luke`_.


Enhancements

* Add :meth:`mne_nirs.statistics.RegressionResults.save`. By `Robert Luke`_.
* Add :meth:`mne_nirs.statistics.ContrastResults.save`. By `Robert Luke`_.
* Add :func:`mne_nirs.statistics.read_glm`. By `Robert Luke`_.
* Add :func:`mne_nirs.experimental_design.longest_inter_annotation_interval`. By `Robert Luke`_.
* Add :func:`mne_nirs.experimental_design.drift_high_pass`. By `Robert Luke`_.


Fixes

* Fix end values for windowed quality metrics. By `Robert Luke`_.
* Fix snirf writer bug where it required the optional DateOfBirth field. By `Christian Arthur`_, Jeonghoon Choi, Jiazhen Liu, and Juncheng Zhang


v0.1.0
------

API changes

* Add :class:`~mne_nirs.statistics.RegressionResults` and :class:`~mne_nirs.statistics.ContrastResults` classes to store GLM results. By `Robert Luke`_.

Adding a class simplifies user code and common use cases.
To generate results in the new format, use the function ``run_glm`` rather than ``run_GLM``.
This will return a ``RegressionResults`` type that contains all relevant information.
All previous functionality still exists with this new type,
but is now accessible as more succinct methods that handle the relevant information,
this results in less arguments being passed around by the user.
For example, to access the previous ``glm_to_tidy(results)`` functionality use the new ``results.to_dataframe()``.
A full list of replacement methods is provided below.

Enhancements

* :meth:`mne_nirs.statistics.RegressionResults.compute_contrast` replaces ``compute_contrast``.
* :meth:`mne_nirs.statistics.RegressionResults.plot_topo` replaces ``plot_glm_topo``.
* :meth:`mne_nirs.statistics.RegressionResults.to_dataframe` replaces ``glm_to_tidy``.
* :meth:`mne_nirs.statistics.RegressionResults.to_dataframe_region_of_interest` replaces ``glm_region_of_interest``.
* Add :meth:`mne_nirs.statistics.RegressionResults.scatter` to display GLM results as a scatter plot.
* Add :meth:`mne_nirs.statistics.RegressionResults.surface_projection` to display GLM results on a cortical surface.
* Add :meth:`mne_nirs.statistics.ContrastResults.plot_topo`.
* Add :meth:`mne_nirs.statistics.ContrastResults.to_dataframe`.
* Add :meth:`mne_nirs.statistics.ContrastResults.scatter`.


Documentation

* Add an example of how to include second level covariates in the group level GLM tutorial. By `Robert Luke`_.


Bugs

* Fix inconsistencies between files written via :meth:`mne_nirs.io.snirf.write_raw_snirf` and the `current version <https://github.com/fNIRS/snirf/blob/52de9a6724ddd0c9dcd36d8d11007895fed74205/snirf_specification.md>`_ of the official SNIRF spec. By `Darin Erat Sleiter`_.


Infrastructure

* Change the git repository to use `main`, rather than `master` branch. By `Robert Luke`_.


v0.0.6
------

Enhancements

* Added binder functionality to website tutorials. By `Robert Luke`_.

* Added convenience function for projecting GLM estimates to the cortical surface. By `Robert Luke`_.

* Improved Hitachi support. By `Eric Larson`_.


v0.0.5
------

Enhancements

* Added windowed signal quality metrics scalp coupling index and peak power. By `Robert Luke`_.

* Added a finite impulse response (FIR) GLM example. By `Robert Luke`_.

* Added group level waveform example. By `Robert Luke`_.

* Added ability to use auto regressive models of any order in GLM computation. By `Robert Luke`_.


.. _Robert Luke: https://github.com/rob-luke/
.. _Eric Larson: https://github.com/larsoner/
.. _Darin Erat Sleiter: https://github.com/dsleiter
.. _Christian Arthur: https://github.com/chrsthur
