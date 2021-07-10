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

Infrastructure

* Change the git repository to use `main`, rather than `master` branch.


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
