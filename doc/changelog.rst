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

v0.1.0
------

Enhancements

* Add :class:`~mne_nirs.statistics.RegressionResults` and :class:`~mne_nirs.statistics.ContrastResults` classes to store GLM results. These classes have methods that replace the functions compute_contrast and glm_region_of_interest. By `Robert Luke`_.



v0.0.6
------

Enhancements

* Added binder functionality to website tutorials. By `Robert Luke`_.

* Added convenience function for projecting GLM estimates to the cortical surface. By `Robert Luke`_.

* Improved Hitachi support. By `Eric Larson`_.


v0.0.5
------

* Added windowed signal quality metrics scalp coupling index and peak power.

* Added FIR GLM example.

* Added group level waveform example.

* Added ability to use auto regressive models of any order in GLM computation.



.. _Robert Luke: https://github.com/rob-luke/
.. _Eric Larson: https://github.com/larsoner/
