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

API changes

* Add :class:`~mne_nirs.statistics.RegressionResults` and :class:`~mne_nirs.statistics.ContrastResults` classes to store GLM results. By `Robert Luke`_.

Adding a class simplifies user code and common use cases.
To generate results in the new format use the function run_glm rather than run_GLM.
This will return a RegressionResults type that contains all relevant information.
All previous existing functionality still exists with this new type,
but is now accessible as more succinct methods that handle the relevant information,
this results in less arguments being passed around by the user.
To access the previous glm_to_tidy(results) functionality use the new results.to_dataframe().
To access the previous plot_glm_topo(results) functionality use the new results.plot_topo().
A full list of replacement methods is provided below

Enhancements

* Add method RegressionResults.compute_contrast which replaces compute_contrast.
* Add method RegressionResults.plot_glm_topo which replaces plot_topo.
* Add method RegressionResults.to_dataframe which replaces glm_to_tidy.
* Add method RegressionResults.to_dataframe_region_of_interest which replaces glm_region_of_interest.
* Add new method RegressionResults.scatter which illustrates the GLM results as a scatter plot.
* Add new method RegressionResults.surface_projection which illustrates the GLM results as a surface projection.




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
